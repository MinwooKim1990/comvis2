"""
Evaluation script for trained Active Visual Search agent.

Evaluates performance and creates visualizations.
"""

import os
import sys
import argparse
import yaml
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.environment import VisualSearchEnv
from src.models.dinov3_encoder import DINOv3Encoder
from src.models.dqn_agent import DQNAgent
from src.utils.data_loader import prepare_datasets, get_class_names
from src.utils.visualization import (
    plot_success_comparison,
    visualize_episode,
    plot_episode_grid,
    plot_search_heatmap
)


class Evaluator:
    """
    Evaluator for Active Visual Search agent.

    Loads trained model and evaluates on test set.
    """

    def __init__(self, checkpoint_path: str, config: Optional[Dict] = None):
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration dict (loaded from checkpoint if None)
        """
        self.checkpoint_path = checkpoint_path

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if config is None:
            config = checkpoint['config']
        self.config = config

        self.device = config['experiment']['device']
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.device = 'cpu'

        # Prepare datasets
        print("\nPreparing datasets...")
        _, _, self.test_dataset = prepare_datasets(config)

        # Initialize environment
        print("\nInitializing environment...")
        self.env = VisualSearchEnv(
            config=config,
            dataset=self.test_dataset,
            device=self.device
        )

        # Initialize encoder
        print("\nInitializing DINOv3 encoder...")
        self.encoder = DINOv3Encoder(
            model_name=config['dinov3']['variant'],
            freeze=True,
            use_fp16=config['dinov3']['use_fp16'] and self.device == 'cuda',
            device=self.device
        )

        # Precompute target embeddings
        print("\nPrecomputing target embeddings...")
        self._precompute_target_embeddings()

        # Initialize agent
        print("\nInitializing agent...")
        self.agent = DQNAgent(
            config=config,
            encoder=self.encoder,
            device=self.device
        )

        # Load trained weights
        self.agent.q_network.load_state_dict(checkpoint['agent_state'])
        self.agent.q_network.eval()

        print("\n" + "="*60)
        print("EVALUATOR READY")
        print("="*60 + "\n")

    def _precompute_target_embeddings(self):
        """Precompute target class embeddings."""
        self.target_embeddings = {}

        for class_id in self.config['dataset']['target_classes']:
            for img, label in self.test_dataset:
                if label == class_id:
                    img_batch = img.unsqueeze(0).to(self.device)
                    embedding = self.encoder.encode(img_batch)
                    self.target_embeddings[class_id] = embedding.squeeze(0)
                    break

    def evaluate(
        self,
        num_episodes: int = 100,
        save_videos: bool = False,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Evaluate agent on test episodes.

        Args:
            num_episodes: Number of test episodes
            save_videos: Whether to save episode videos
            save_dir: Directory to save results

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating on {num_episodes} test episodes...")

        rewards = []
        steps = []
        successes = []
        episode_data = []

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            video_dir = os.path.join(save_dir, 'videos')
            os.makedirs(video_dir, exist_ok=True)

        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            state = self.env.reset()
            target_class = state['target_class']
            target_features = self.target_embeddings[target_class]

            done = False
            total_reward = 0.0
            num_steps = 0
            frames = []
            trajectory = []

            while not done:
                # Select action (greedy)
                action = self.agent.select_action(state, target_features, eval_mode=True)

                # Save frame
                if save_videos and episode < 10:  # Only save first 10
                    frames.append(self.env.render())

                # Track trajectory
                trajectory.append(self.env.agent_pos.copy())

                # Step
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                num_steps += 1

            rewards.append(total_reward)
            steps.append(num_steps)
            successes.append(1.0 if info.get('success', False) else 0.0)

            episode_data.append({
                'reward': total_reward,
                'steps': num_steps,
                'success': info.get('success', False),
                'trajectory': trajectory,
                'frames': frames if save_videos and episode < 10 else None
            })

            # Save video
            if save_videos and episode < 10 and len(frames) > 0:
                video_path = os.path.join(video_dir, f'episode_{episode:03d}.gif')
                visualize_episode(frames, save_path=video_path, fps=5)

        # Compute metrics
        results = {
            'num_episodes': num_episodes,
            'success_rate': np.mean(successes),
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_steps': np.mean(steps),
            'std_steps': np.std(steps),
            'avg_steps_success': np.mean([s for s, succ in zip(steps, successes) if succ > 0.5])
                                  if any(successes) else 0,
            'episode_data': episode_data
        }

        # Print results
        self._print_results(results)

        return results

    def compare_with_random(
        self,
        num_episodes: int = 100,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Compare trained agent with random baseline.

        Args:
            num_episodes: Number of episodes for each agent
            save_dir: Directory to save comparison plots

        Returns:
            Comparison results
        """
        print("\n" + "="*60)
        print("COMPARING TRAINED AGENT VS RANDOM BASELINE")
        print("="*60)

        # Evaluate trained agent
        print("\n--- Trained Agent ---")
        trained_results = self.evaluate(num_episodes, save_videos=False)

        # Evaluate random agent
        print("\n--- Random Agent ---")
        random_results = self._evaluate_random(num_episodes)

        # Create comparison
        comparison = {
            'Random Agent': {
                'success_rate': random_results['success_rate'],
                'avg_steps': random_results['avg_steps'],
                'avg_reward': random_results['avg_reward']
            },
            'Trained Agent': {
                'success_rate': trained_results['success_rate'],
                'avg_steps': trained_results['avg_steps'],
                'avg_reward': trained_results['avg_reward']
            }
        }

        # Print comparison
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(f"\nSuccess Rate:")
        print(f"  Random:  {comparison['Random Agent']['success_rate']:.1%}")
        print(f"  Trained: {comparison['Trained Agent']['success_rate']:.1%}")
        print(f"  Improvement: {(comparison['Trained Agent']['success_rate'] - comparison['Random Agent']['success_rate']):.1%}")

        print(f"\nAverage Steps:")
        print(f"  Random:  {comparison['Random Agent']['avg_steps']:.1f}")
        print(f"  Trained: {comparison['Trained Agent']['avg_steps']:.1f}")
        print(f"  Reduction: {comparison['Random Agent']['avg_steps'] - comparison['Trained Agent']['avg_steps']:.1f} steps")

        print(f"\nAverage Reward:")
        print(f"  Random:  {comparison['Random Agent']['avg_reward']:.2f}")
        print(f"  Trained: {comparison['Trained Agent']['avg_reward']:.2f}")
        print(f"  Improvement: {comparison['Trained Agent']['avg_reward'] - comparison['Random Agent']['avg_reward']:.2f}")

        # Save comparison plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(save_dir, 'comparison.png')
            plot_success_comparison(comparison, save_path=plot_path)

        return {
            'comparison': comparison,
            'trained_results': trained_results,
            'random_results': random_results
        }

    def _evaluate_random(self, num_episodes: int) -> Dict:
        """Evaluate random agent."""
        rewards = []
        steps = []
        successes = []

        for _ in tqdm(range(num_episodes), desc="Evaluating random"):
            state = self.env.reset()
            done = False
            total_reward = 0.0
            num_steps = 0

            while not done:
                action = np.random.randint(0, 5)  # Random action
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                num_steps += 1

            rewards.append(total_reward)
            steps.append(num_steps)
            successes.append(1.0 if info.get('success', False) else 0.0)

        return {
            'success_rate': np.mean(successes),
            'avg_reward': np.mean(rewards),
            'avg_steps': np.mean(steps)
        }

    def visualize_examples(
        self,
        num_examples: int = 5,
        save_dir: Optional[str] = None
    ):
        """
        Create visualizations of example episodes.

        Args:
            num_examples: Number of example episodes
            save_dir: Directory to save visualizations
        """
        print(f"\nCreating {num_examples} example visualizations...")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for i in range(num_examples):
            state = self.env.reset()
            target_class = state['target_class']
            target_features = self.target_embeddings[target_class]

            done = False
            frames = []
            trajectory = []

            while not done:
                frames.append(self.env.render())
                trajectory.append(self.env.agent_pos.copy())

                action = self.agent.select_action(state, target_features, eval_mode=True)
                state, reward, done, info = self.env.step(action)

            # Save as GIF
            if save_dir:
                gif_path = os.path.join(save_dir, f'example_{i+1}.gif')
                visualize_episode(
                    frames,
                    title=f"Example {i+1} ({'Success' if info.get('success') else 'Failure'})",
                    save_path=gif_path,
                    fps=5
                )

                # Save frame grid
                grid_path = os.path.join(save_dir, f'example_{i+1}_grid.png')
                plot_episode_grid(frames, num_frames=12, save_path=grid_path)

                # Save heatmap
                heatmap_path = os.path.join(save_dir, f'example_{i+1}_heatmap.png')
                plot_search_heatmap(
                    trajectory,
                    self.env.canvas_size,
                    self.env.target_position,
                    save_path=heatmap_path
                )

        print(f"Visualizations saved to {save_dir}/")

    def _print_results(self, results: Dict):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"\nEpisodes: {results['num_episodes']}")
        print(f"\nSuccess Rate: {results['success_rate']:.2%}")
        print(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Average Steps: {results['avg_steps']:.1f} ± {results['std_steps']:.1f}")
        if results['avg_steps_success'] > 0:
            print(f"Average Steps (Success): {results['avg_steps_success']:.1f}")
        print("="*60 + "\n")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of test episodes')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with random baseline')
    parser.add_argument('--visualize', action='store_true',
                       help='Create example visualizations')
    parser.add_argument('--save-dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    args = parser.parse_args()

    # Create evaluator
    evaluator = Evaluator(args.checkpoint)

    # Evaluate
    if args.compare:
        results = evaluator.compare_with_random(
            num_episodes=args.episodes,
            save_dir=args.save_dir
        )
    else:
        results = evaluator.evaluate(
            num_episodes=args.episodes,
            save_videos=True,
            save_dir=args.save_dir
        )

    # Visualizations
    if args.visualize:
        viz_dir = os.path.join(args.save_dir, 'visualizations')
        evaluator.visualize_examples(num_examples=5, save_dir=viz_dir)

    print(f"\nResults saved to {args.save_dir}/")


if __name__ == '__main__':
    main()
