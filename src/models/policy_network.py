"""
Policy Networks for Active Visual Search.

Q-Network for DQN agent using DINOv3 features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class QNetwork(nn.Module):
    """
    Q-Network for DQN agent.

    Takes DINOv3 features + position encoding + target embedding
    and outputs Q-values for each action.

    Args:
        feature_dim: Dimension of DINOv3 features (384 for ViT-S)
        position_dim: Dimension of position encoding (2 for normalized x, y)
        num_actions: Number of discrete actions (5: up/down/left/right/found)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        feature_dim: int = 384,
        position_dim: int = 2,
        num_actions: int = 5,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.2
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.position_dim = position_dim
        self.num_actions = num_actions

        # Input: DINOv3 features (384) + position (2) + target embedding (384)
        input_dim = feature_dim + position_dim + feature_dim

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_actions))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        observation_features: torch.Tensor,
        position: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to compute Q-values.

        Args:
            observation_features: DINOv3 features of current observation (B, feature_dim)
            position: Normalized position (B, position_dim)
            target_features: DINOv3 features of target class (B, feature_dim)

        Returns:
            q_values: Q-values for each action (B, num_actions)
        """
        # Concatenate all inputs
        x = torch.cat([observation_features, position, target_features], dim=-1)

        # Forward through network
        q_values = self.network(x)

        return q_values


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture.

    Separates value and advantage streams for better learning.
    V(s) + (A(s,a) - mean(A(s,a)))

    Args:
        feature_dim: Dimension of DINOv3 features
        position_dim: Dimension of position encoding
        num_actions: Number of discrete actions
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
    """

    def __init__(
        self,
        feature_dim: int = 384,
        position_dim: int = 2,
        num_actions: int = 5,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.2
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_actions = num_actions

        input_dim = feature_dim + position_dim + feature_dim

        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], num_actions)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        observation_features: torch.Tensor,
        position: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with dueling architecture.

        Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))
        """
        # Concatenate inputs
        x = torch.cat([observation_features, position, target_features], dim=-1)

        # Shared features
        shared = self.shared_layers(x)

        # Value and advantage
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)

        # Combine using dueling formula
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))

        return q_values


def test_networks():
    """Test Q-Network implementations."""
    print("\n=== Testing Q-Networks ===\n")

    batch_size = 8
    feature_dim = 384
    position_dim = 2
    num_actions = 5

    # Create dummy inputs
    obs_features = torch.randn(batch_size, feature_dim)
    position = torch.randn(batch_size, position_dim)
    target_features = torch.randn(batch_size, feature_dim)

    print(f"Input shapes:")
    print(f"  Observation features: {obs_features.shape}")
    print(f"  Position: {position.shape}")
    print(f"  Target features: {target_features.shape}")

    # Test standard Q-Network
    print("\n--- Standard Q-Network ---")
    q_net = QNetwork(
        feature_dim=feature_dim,
        position_dim=position_dim,
        num_actions=num_actions,
        hidden_dims=[512, 256]
    )

    q_values = q_net(obs_features, position, target_features)
    print(f"Output Q-values shape: {q_values.shape}")
    print(f"Sample Q-values: {q_values[0].detach()}")

    num_params = sum(p.numel() for p in q_net.parameters())
    print(f"Number of parameters: {num_params:,}")

    # Test Dueling Q-Network
    print("\n--- Dueling Q-Network ---")
    dueling_q_net = DuelingQNetwork(
        feature_dim=feature_dim,
        position_dim=position_dim,
        num_actions=num_actions,
        hidden_dims=[512, 256]
    )

    dueling_q_values = dueling_q_net(obs_features, position, target_features)
    print(f"Output Q-values shape: {dueling_q_values.shape}")
    print(f"Sample Q-values: {dueling_q_values[0].detach()}")

    num_params_dueling = sum(p.numel() for p in dueling_q_net.parameters())
    print(f"Number of parameters: {num_params_dueling:,}")

    # Test gradient flow
    print("\n--- Testing Gradient Flow ---")
    loss = q_values.mean()
    loss.backward()
    has_grad = any(p.grad is not None for p in q_net.parameters())
    print(f"Gradients computed: {has_grad}")

    print("\n=== Test Complete ===\n")


if __name__ == '__main__':
    test_networks()
