"""
Training script for Active Visual Search with DINOv3 + DQN.

Main training loop with logging, checkpointing, and evaluation.
"""

import os
import sys
import argparse
import yaml
from datetime import datetime
from typing import Dict, Optional
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.environment import VisualSearchEnv
from src.models.dinov3_encoder import DINOv3Encoder
from src.models.dqn_agent import DQNAgent
from src.utils.data_loader import prepare_datasets, get_class_names
from src.utils.visualization import plot_training_curves, visualize_episode


class Trainer:
    """
    Trainer for Active Visual Search agent.

    Manages training loop, logging, checkpointing, and evaluation.
    """

    def __init__(self, config: Dict, log_dir: Optional[str] = None):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
            log_dir: Directory for logs and checkpoints
        """
        self.config = config
        self.device = config['experiment']['device']

        # Create log directory
        if log_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = os.path.join(
                config['logging']['log_dir'],
                f"run_{timestamp}"
            )
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(self.log_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # TensorBoard
        if config['logging']['tensorboard']:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None

        # Checkpoint directory
        self.checkpoint_dir = config['logging']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Prepare datasets
        print("\n" + "="*60)
        print("ACTIVE VISUAL SEARCH - TRAINING")
        print("="*60)

        self.train_dataset, self.val_dataset, self.test_dataset = prepare_datasets(config)

        # Initialize environment
        print("\nInitializing environment...")
        self.env = VisualSearchEnv(
            config=config,
            dataset=self.train_dataset,
            device=self.device
        )

        # Initialize DINOv3 encoder
        print("\nInitializing DINOv3 encoder...")
        self.encoder = DINOv3Encoder(
            model_name=config['dinov3']['variant'],
            freeze=config['dinov3']['freeze'],
            use_fp16=config['dinov3']['use_fp16'] and self.device == 'cuda',
            device=self.device
        )

        # Precompute target class embeddings
        print("\nPrecomputing target class embeddings...")
        self._precompute_target_embeddings()

        # Initialize DQN agent
        print("\nInitializing DQN agent...")
        self.agent = DQNAgent(
            config=config,
            encoder=self.encoder,
            device=self.device
        )

        # Training state
        self.episode = 0
        self.global_step = 0
        self.best_success_rate = 0.0

        # Metrics tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_success = []
        self.losses = []

        # Rolling metrics for display
        self.recent_rewards = deque(maxlen=100)
        self.recent_success = deque(maxlen=100)
        self.recent_steps = deque(maxlen=100)

        print("\n" + "="*60)
        print("INITIALIZATION COMPLETE")
        print("="*60 + "\n")

    def _precompute_target_embeddings(self):
        """Precompute DINOv3 embeddings for each target class."""
        self.target_embeddings = {}

        for class_id in self.config['dataset']['target_classes']:
            # Find first image of this class
            for img, label in self.train_dataset:
                if label == class_id:
                    img_batch = img.unsqueeze(0).to(self.device)
                    embedding = self.encoder.encode(img_batch)
                    self.target_embeddings[class_id] = embedding.squeeze(0)
                    break

        print(f"Cached embeddings for {len(self.target_embeddings)} classes")

    def train(self, num_episodes: Optional[int] = None):
        """
        Main training loop.

        Args:
            num_episodes: Number of episodes to train (uses config if None)
        """
        if num_episodes is None:
            num_episodes = self.config['training']['num_episodes']

        eval_freq = self.config['training']['eval_frequency']
        save_freq = self.config['training']['save_frequency']
        log_freq = self.config['logging']['log_frequency']

        print(f"Starting training for {num_episodes} episodes...")
        print(f"Device: {self.device}")
        print(f"Log directory: {self.log_dir}\n")

        # Progress bar
        pbar = tqdm(range(num_episodes), desc="Training")

        for episode in pbar:
            self.episode = episode

            # Run episode
            episode_reward, episode_steps, success, info = self._run_episode()

            # Update metrics
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(episode_steps)
            self.episode_success.append(1.0 if success else 0.0)

            self.recent_rewards.append(episode_reward)
            self.recent_success.append(1.0 if success else 0.0)
            self.recent_steps.append(episode_steps)

            # Update progress bar
            avg_reward = np.mean(self.recent_rewards)
            avg_success = np.mean(self.recent_success)
            avg_steps = np.mean(self.recent_steps)

            pbar.set_postfix({
                'reward': f'{avg_reward:.2f}',
                'success': f'{avg_success:.2%}',
                'steps': f'{avg_steps:.1f}',
                'ε': f'{self.agent.epsilon:.3f}'
            })

            # Logging
            if (episode + 1) % log_freq == 0:
                self._log_metrics(episode + 1)

            # Evaluation
            if (episode + 1) % eval_freq == 0:
                self._evaluate()

            # Save checkpoint
            if (episode + 1) % save_freq == 0:
                self._save_checkpoint(f'episode_{episode+1}.pt')

            # Decay epsilon
            self.agent.end_episode()

        # Final save
        self._save_checkpoint('final.pt')

        # Final evaluation
        print("\n\nRunning final evaluation...")
        self._evaluate(num_episodes=100)

        # Save training curves
        self._save_training_curves()

        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\nLogs saved to: {self.log_dir}")
        print(f"Best success rate: {self.best_success_rate:.2%}")

        if self.writer:
            self.writer.close()

    def _run_episode(self) -> tuple:
        """
        Run single training episode.

        Returns:
            (total_reward, num_steps, success, info)
        """
        state = self.env.reset()
        target_class = state['target_class']
        target_features = self.target_embeddings[target_class]

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # Select action
            action = self.agent.select_action(state, target_features, eval_mode=False)

            # Step environment
            next_state, reward, done, info = self.env.step(action)

            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)

            # Train agent
            loss = self.agent.train_step()
            if loss is not None:
                self.losses.append(loss)

                # Log to TensorBoard
                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, self.global_step)

            total_reward += reward
            steps += 1
            self.global_step += 1

            state = next_state

        success = info.get('success', False)

        return total_reward, steps, success, info

    def _evaluate(self, num_episodes: int = 50):
        """
        Evaluate current policy.

        Args:
            num_episodes: Number of evaluation episodes
        """
        print(f"\nEvaluating on {num_episodes} episodes...")

        eval_rewards = []
        eval_steps = []
        eval_success = []

        for _ in range(num_episodes):
            state = self.env.reset()
            target_class = state['target_class']
            target_features = self.target_embeddings[target_class]

            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action = self.agent.select_action(state, target_features, eval_mode=True)
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1

            eval_rewards.append(total_reward)
            eval_steps.append(steps)
            eval_success.append(1.0 if info.get('success', False) else 0.0)

        # Compute metrics
        avg_reward = np.mean(eval_rewards)
        avg_steps = np.mean(eval_steps)
        success_rate = np.mean(eval_success)

        print(f"Evaluation results:")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Avg reward: {avg_reward:.2f}")
        print(f"  Avg steps: {avg_steps:.1f}")

        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('Eval/success_rate', success_rate, self.episode)
            self.writer.add_scalar('Eval/avg_reward', avg_reward, self.episode)
            self.writer.add_scalar('Eval/avg_steps', avg_steps, self.episode)

        # Save best model
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            self._save_checkpoint('best_model.pt')
            print(f"  ✓ New best model saved! (success rate: {success_rate:.2%})")

    def _log_metrics(self, episode: int):
        """Log metrics to TensorBoard."""
        if not self.writer:
            return

        # Recent averages
        if len(self.recent_rewards) > 0:
            self.writer.add_scalar('Train/avg_reward', np.mean(self.recent_rewards), episode)
            self.writer.add_scalar('Train/avg_steps', np.mean(self.recent_steps), episode)
            self.writer.add_scalar('Train/success_rate', np.mean(self.recent_success), episode)

        # Agent state
        self.writer.add_scalar('Agent/epsilon', self.agent.epsilon, episode)
        self.writer.add_scalar('Agent/replay_buffer_size', len(self.agent.replay_buffer), episode)

    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            'episode': self.episode,
            'global_step': self.global_step,
            'best_success_rate': self.best_success_rate,
            'agent_state': self.agent.q_network.state_dict(),
            'target_network_state': self.agent.target_network.state_dict(),
            'optimizer_state': self.agent.optimizer.state_dict(),
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_path)

    def _save_training_curves(self):
        """Save training curves plot."""
        log_data = {
            'episodes': list(range(len(self.episode_rewards))),
            'rewards': self.episode_rewards,
            'steps': self.episode_steps,
            'success_rate': self._compute_rolling_success_rate(),
            'losses': self.losses
        }

        save_path = os.path.join(self.log_dir, 'training_curves.png')
        plot_training_curves(log_data, save_path=save_path, show=False)

    def _compute_rolling_success_rate(self, window: int = 100):
        """Compute rolling success rate."""
        success_rate = []
        for i in range(len(self.episode_success)):
            start = max(0, i - window + 1)
            success_rate.append(np.mean(self.episode_success[start:i+1]))
        return success_rate


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Active Visual Search agent')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to config file')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, overrides config)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config if specified
    if args.device:
        config['experiment']['device'] = args.device

    # Set device
    if config['experiment']['device'] == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config['experiment']['device'] = 'cpu'

    # Create trainer
    trainer = Trainer(config)

    # Train
    try:
        trainer.train(num_episodes=args.episodes)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        trainer._save_checkpoint('interrupted.pt')
        print("Checkpoint saved")


if __name__ == '__main__':
    main()
