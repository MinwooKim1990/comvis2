"""
DQN Agent for Active Visual Search.

Implements Deep Q-Network with experience replay and target networks.
"""

import random
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .dinov3_encoder import DINOv3Encoder
from .policy_network import QNetwork, DuelingQNetwork


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores (state, action, reward, next_state, done) transitions.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: Dict[str, torch.Tensor],
        action: int,
        reward: float,
        next_state: Dict[str, torch.Tensor],
        done: bool
    ):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample random batch of experiences.

        Returns:
            Tuple of batched (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)

        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Stack tensors
        batch_obs = torch.stack([s['observation'] for s in states])
        batch_pos = torch.stack([s['position'] for s in states])
        batch_target_class = torch.tensor([s['target_class'] for s in states])

        batch_next_obs = torch.stack([s['observation'] for s in next_states])
        batch_next_pos = torch.stack([s['position'] for s in next_states])

        batch_actions = torch.tensor(actions, dtype=torch.long)
        batch_rewards = torch.tensor(rewards, dtype=torch.float32)
        batch_dones = torch.tensor(dones, dtype=torch.float32)

        return (
            batch_obs, batch_pos, batch_target_class,
            batch_actions, batch_rewards,
            batch_next_obs, batch_next_pos,
            batch_dones
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent with DINOv3 visual encoder.

    Uses frozen DINOv3 for feature extraction and trainable Q-network.

    Args:
        config: Configuration dictionary
        encoder: DINOv3 encoder instance
        device: torch device
    """

    def __init__(
        self,
        config: Dict[str, Any],
        encoder: DINOv3Encoder,
        device: str = 'cuda'
    ):
        self.config = config
        self.encoder = encoder
        self.device = device

        # Network parameters
        self.feature_dim = encoder.feature_dim
        self.num_actions = 5  # up, down, left, right, found

        # Create Q-networks
        self.use_dueling = config['agent'].get('architecture', 'dqn') == 'dueling_dqn'

        NetworkClass = DuelingQNetwork if self.use_dueling else QNetwork

        self.q_network = NetworkClass(
            feature_dim=self.feature_dim,
            position_dim=2,
            num_actions=self.num_actions,
            hidden_dims=config['agent']['hidden_dims'],
            dropout=config['agent']['dropout']
        ).to(device)

        self.target_network = NetworkClass(
            feature_dim=self.feature_dim,
            position_dim=2,
            num_actions=self.num_actions,
            hidden_dims=config['agent']['hidden_dims'],
            dropout=config['agent']['dropout']
        ).to(device)

        # Initialize target network with Q-network weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config['agent']['learning_rate']
        )

        # Training parameters
        self.gamma = config['agent']['gamma']
        self.epsilon = config['agent']['epsilon_start']
        self.epsilon_end = config['agent']['epsilon_end']
        self.epsilon_decay = config['agent']['epsilon_decay']

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config['agent']['replay_buffer_size']
        )
        self.batch_size = config['agent']['batch_size']
        self.min_replay_size = config['agent']['min_replay_size']

        # Target network update
        self.target_update_freq = config['agent']['target_update_freq']
        self.use_soft_update = config['agent'].get('use_soft_update', False)
        self.tau = config['agent'].get('tau', 0.005)
        self.update_counter = 0

        # Metrics
        self.training_step = 0
        self.episode_count = 0

        print(f"DQN Agent initialized:")
        print(f"  Q-Network: {'Dueling' if self.use_dueling else 'Standard'}")
        print(f"  Feature dim: {self.feature_dim}")
        print(f"  Hidden dims: {config['agent']['hidden_dims']}")
        print(f"  Learning rate: {config['agent']['learning_rate']}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Replay buffer: {config['agent']['replay_buffer_size']}")

    def select_action(
        self,
        state: Dict[str, torch.Tensor],
        target_features: torch.Tensor,
        eval_mode: bool = False
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state dictionary
            target_features: Precomputed target class features
            eval_mode: If True, always use greedy action (no exploration)

        Returns:
            action: Integer in [0, 4]
        """
        # Exploration
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        # Exploitation: select best action
        with torch.no_grad():
            obs = state['observation'].unsqueeze(0).to(self.device)
            pos = state['position'].unsqueeze(0).to(self.device)

            # Extract features
            obs_features = self.encoder.encode(obs)

            # Ensure target features has batch dimension
            if target_features.dim() == 1:
                target_features = target_features.unsqueeze(0)
            target_features = target_features.to(self.device)

            # Get Q-values
            q_values = self.q_network(obs_features, pos, target_features)

            # Select action with max Q-value
            action = q_values.argmax(dim=-1).item()

        return action

    def store_transition(
        self,
        state: Dict[str, torch.Tensor],
        action: int,
        reward: float,
        next_state: Dict[str, torch.Tensor],
        done: bool
    ):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step (update Q-network).

        Returns:
            loss: Training loss (None if not enough samples)
        """
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        # Sample batch
        (
            batch_obs, batch_pos, batch_target_class,
            batch_actions, batch_rewards,
            batch_next_obs, batch_next_pos,
            batch_dones
        ) = self.replay_buffer.sample(self.batch_size)

        # Move to device
        batch_obs = batch_obs.to(self.device)
        batch_pos = batch_pos.to(self.device)
        batch_next_obs = batch_next_obs.to(self.device)
        batch_next_pos = batch_next_pos.to(self.device)
        batch_actions = batch_actions.to(self.device)
        batch_rewards = batch_rewards.to(self.device)
        batch_dones = batch_dones.to(self.device)

        # Extract features
        with torch.no_grad():
            obs_features = self.encoder.encode(batch_obs)
            next_obs_features = self.encoder.encode(batch_next_obs)

            # Get target features (use cached features for efficiency)
            # For simplicity, we'll encode a dummy image for each class
            # In practice, you should cache these once
            target_features = obs_features  # Simplified for MVP

        # Current Q-values
        current_q_values = self.q_network(obs_features, batch_pos, target_features)
        current_q_values = current_q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

        # Next Q-values (from target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_obs_features, batch_next_pos, target_features)
            max_next_q_values = next_q_values.max(dim=1)[0]

            # Compute target Q-values
            target_q_values = batch_rewards + self.gamma * max_next_q_values * (1 - batch_dones)

        # Compute loss
        loss = nn.functional.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.use_soft_update:
            self._soft_update_target_network()
        elif self.update_counter % self.target_update_freq == 0:
            self._hard_update_target_network()

        self.training_step += 1

        return loss.item()

    def _soft_update_target_network(self):
        """Soft update: target = tau * online + (1-tau) * target."""
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def _hard_update_target_network(self):
        """Hard update: target = online."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def end_episode(self):
        """Call at end of each episode."""
        self.episode_count += 1
        self.decay_epsilon()

    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': self.config
        }, filepath)
        print(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        print(f"Agent loaded from {filepath}")

    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'replay_buffer_size': len(self.replay_buffer),
            'network_type': 'Dueling DQN' if self.use_dueling else 'DQN'
        }
