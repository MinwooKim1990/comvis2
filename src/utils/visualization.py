"""
Visualization utilities for Active Visual Search.

Tools for rendering episodes, plotting metrics, and creating animations.
"""

import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
from PIL import Image
import imageio


def plot_training_curves(
    log_data: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training curves (rewards, success rate, steps).

    Args:
        log_data: Dictionary with 'episodes', 'rewards', 'success_rate', 'steps'
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

    # Episode rewards
    axes[0, 0].plot(log_data['episodes'], log_data['rewards'], alpha=0.6, label='Episode Reward')
    if len(log_data['rewards']) > 10:
        # Rolling average
        window = min(50, len(log_data['rewards']) // 10)
        rolling_avg = np.convolve(log_data['rewards'], np.ones(window)/window, mode='valid')
        axes[0, 0].plot(log_data['episodes'][window-1:], rolling_avg,
                       'r-', linewidth=2, label=f'Rolling Avg ({window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Success rate
    axes[0, 1].plot(log_data['episodes'], log_data['success_rate'], 'g-', linewidth=2)
    axes[0, 1].axhline(y=0.7, color='r', linestyle='--', label='Target (70%)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_title('Success Rate (Rolling)')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Steps per episode
    axes[1, 0].plot(log_data['episodes'], log_data['steps'], alpha=0.6, label='Steps')
    if len(log_data['steps']) > 10:
        window = min(50, len(log_data['steps']) // 10)
        rolling_avg = np.convolve(log_data['steps'], np.ones(window)/window, mode='valid')
        axes[1, 0].plot(log_data['episodes'][window-1:], rolling_avg,
                       'r-', linewidth=2, label=f'Rolling Avg ({window})')
    axes[1, 0].axhline(y=15, color='g', linestyle='--', label='Target (15)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].set_title('Steps per Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Loss (if available)
    if 'losses' in log_data and len(log_data['losses']) > 0:
        axes[1, 1].plot(log_data['losses'], alpha=0.7)
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Loss data not available',
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Training Loss')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_episode(
    frames: List[np.ndarray],
    title: str = "Agent Search Trajectory",
    save_path: Optional[str] = None,
    fps: int = 5
) -> Optional[str]:
    """
    Create animated GIF from episode frames.

    Args:
        frames: List of rendered frames (H, W, 3)
        title: Title for the visualization
        save_path: Path to save GIF
        fps: Frames per second

    Returns:
        Path to saved GIF if save_path provided
    """
    if len(frames) == 0:
        print("No frames to visualize")
        return None

    if save_path:
        # Save as GIF
        imageio.mimsave(save_path, frames, fps=fps)
        print(f"Episode animation saved to {save_path}")
        return save_path
    else:
        # Display in matplotlib
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(title)
        ax.axis('off')

        img_plot = ax.imshow(frames[0])

        def update(frame_idx):
            img_plot.set_array(frames[frame_idx])
            return [img_plot]

        anim = FuncAnimation(fig, update, frames=len(frames),
                           interval=1000//fps, blit=True)
        plt.show()
        return None


def plot_episode_grid(
    frames: List[np.ndarray],
    num_frames: int = 12,
    save_path: Optional[str] = None,
    title: str = "Episode Trajectory"
):
    """
    Plot grid of frames from episode.

    Args:
        frames: List of frames
        num_frames: Number of frames to show
        save_path: Path to save figure
        title: Title for plot
    """
    num_frames = min(num_frames, len(frames))
    indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)

    rows = (num_frames + 3) // 4
    cols = min(4, num_frames)

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    if num_frames == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        axes[i].imshow(frames[idx])
        axes[i].set_title(f'Step {idx+1}')
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Episode grid saved to {save_path}")

    plt.show()


def plot_success_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
):
    """
    Plot comparison between different agents (e.g., random vs trained).

    Args:
        results: Dictionary with agent_name -> metrics dict
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Agent Performance Comparison', fontsize=16, fontweight='bold')

    agent_names = list(results.keys())

    # Success rate
    success_rates = [results[name]['success_rate'] * 100 for name in agent_names]
    axes[0].bar(agent_names, success_rates, color=['red', 'green'], alpha=0.7)
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Success Rate')
    axes[0].set_ylim([0, 100])
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

    # Average steps
    avg_steps = [results[name]['avg_steps'] for name in agent_names]
    axes[1].bar(agent_names, avg_steps, color=['red', 'green'], alpha=0.7)
    axes[1].set_ylabel('Average Steps')
    axes[1].set_title('Average Steps to Success')
    for i, v in enumerate(avg_steps):
        axes[1].text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')

    # Average reward
    avg_rewards = [results[name]['avg_reward'] for name in agent_names]
    axes[2].bar(agent_names, avg_rewards, color=['red', 'green'], alpha=0.7)
    axes[2].set_ylabel('Average Reward')
    axes[2].set_title('Average Episode Reward')
    for i, v in enumerate(avg_rewards):
        axes[2].text(i, v + 0.2, f'{v:.1f}', ha='center', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    plt.show()


def plot_search_heatmap(
    positions: List[Tuple[int, int]],
    canvas_size: Tuple[int, int],
    target_position: Tuple[int, int],
    save_path: Optional[str] = None
):
    """
    Plot heatmap of agent's visited positions.

    Args:
        positions: List of (y, x) positions visited
        canvas_size: Size of canvas
        target_position: Position of target
        save_path: Path to save figure
    """
    # Create heatmap
    heatmap = np.zeros(canvas_size)
    for y, x in positions:
        if 0 <= y < canvas_size[0] and 0 <= x < canvas_size[1]:
            heatmap[y, x] += 1

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')

    # Mark target
    target_y, target_x = target_position
    ax.plot(target_x, target_y, 'b*', markersize=30, label='Target')

    # Mark start
    if len(positions) > 0:
        start_y, start_x = positions[0]
        ax.plot(start_x, start_y, 'g^', markersize=15, label='Start')

    # Mark end
    if len(positions) > 1:
        end_y, end_x = positions[-1]
        ax.plot(end_x, end_y, 'rs', markersize=15, label='End')

    plt.colorbar(im, ax=ax, label='Visit Count')
    ax.set_title('Agent Search Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")

    plt.show()


def create_summary_report(
    train_log: Dict,
    eval_results: Dict,
    save_dir: str
):
    """
    Create comprehensive summary report with all visualizations.

    Args:
        train_log: Training log data
        eval_results: Evaluation results
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n=== Creating Summary Report ===\n")

    # Training curves
    print("Generating training curves...")
    plot_training_curves(
        train_log,
        save_path=os.path.join(save_dir, 'training_curves.png'),
        show=False
    )

    # Performance comparison
    if 'comparison' in eval_results:
        print("Generating performance comparison...")
        plot_success_comparison(
            eval_results['comparison'],
            save_path=os.path.join(save_dir, 'performance_comparison.png')
        )

    print(f"\nSummary report saved to {save_dir}/")
    print("=" * 50)


def test_visualization():
    """Test visualization functions."""
    print("\n=== Testing Visualization Tools ===\n")

    # Create dummy data
    episodes = list(range(100))
    rewards = np.random.randn(100).cumsum() + np.linspace(-10, 5, 100)
    success_rate = 1 / (1 + np.exp(-0.05 * (np.array(episodes) - 50)))
    steps = 50 - 30 * success_rate + np.random.randn(100) * 3

    log_data = {
        'episodes': episodes,
        'rewards': rewards,
        'success_rate': success_rate,
        'steps': steps,
        'losses': []
    }

    # Test training curves
    plot_training_curves(log_data, show=False)
    print("✓ Training curves test passed")

    # Test comparison
    results = {
        'Random': {'success_rate': 0.2, 'avg_steps': 35.0, 'avg_reward': -3.5},
        'Trained': {'success_rate': 0.75, 'avg_steps': 12.5, 'avg_reward': 6.2}
    }
    plot_success_comparison(results, save_path=None)
    print("✓ Comparison plot test passed")

    print("\n=== Test Complete ===\n")


if __name__ == '__main__':
    test_visualization()
