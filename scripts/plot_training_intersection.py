"""
Plot Training Results for Intersection RL Agent.

Purpose:
    Generate training plots from TensorBoard logs.
    
Outputs:
    - Reward progression plot
    - Episode length plot
    - Success rate plot (if available)
    - Saved to assets/plots/intersection/
    
Compliance:
    - Type hints everywhere
    - Clean plotting
    - Reproducible analysis

Run: python scripts/plot_training_intersection.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.intersection_config import INTERSECTION_PATHS

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("❌ TensorBoard not installed. Install with: pip install tensorboard")
    sys.exit(1)


def load_tensorboard_data(log_dir: str, tags: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load data from TensorBoard logs.
    
    Args:
        log_dir: Path to TensorBoard log directory
        tags: List of metric tags to load
        
    Returns:
        Dictionary mapping tag to (steps, values) arrays
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    data = {}
    for tag in tags:
        try:
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            data[tag] = (steps, values)
        except KeyError:
            print(f"⚠️ Tag not found: {tag}")
    
    return data


def plot_training_metrics(log_dir: str, output_dir: str) -> None:
    """
    Generate all training plots.
    
    Args:
        log_dir: Path to TensorBoard log directory
        output_dir: Path to save plots
    """
    print(f"\n📊 Loading data from: {log_dir}")
    
    # Load data
    tags = [
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "train/entropy_loss",
        "train/policy_gradient_loss",
        "train/value_loss",
    ]
    
    data = load_tensorboard_data(log_dir, tags)
    
    if not data:
        print("❌ No data found in TensorBoard logs")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Reward Progression
    if "rollout/ep_rew_mean" in data:
        steps, rewards = data["rollout/ep_rew_mean"]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, rewards, linewidth=2, color='#2E86AB')
        plt.xlabel('Timesteps', fontsize=12)
        plt.ylabel('Mean Episode Reward', fontsize=12)
        plt.title('Intersection RL Agent: Training Reward Progression', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add smoothed line
        if len(rewards) > 10:
            window = min(50, len(rewards) // 10)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(steps[window-1:], smoothed, linewidth=3, color='#A23B72', alpha=0.8, label='Smoothed')
            plt.legend()
        
        plt.tight_layout()
        reward_plot_path = output_path / "intersection_reward_progression.png"
        plt.savefig(reward_plot_path, dpi=300)
        print(f"✅ Saved: {reward_plot_path}")
        plt.close()
    
    # 2. Episode Length
    if "rollout/ep_len_mean" in data:
        steps, lengths = data["rollout/ep_len_mean"]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, lengths, linewidth=2, color='#F18F01')
        plt.xlabel('Timesteps', fontsize=12)
        plt.ylabel('Mean Episode Length', fontsize=12)
        plt.title('Intersection RL Agent: Episode Length Progression', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add smoothed line
        if len(lengths) > 10:
            window = min(50, len(lengths) // 10)
            smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
            plt.plot(steps[window-1:], smoothed, linewidth=3, color='#C73E1D', alpha=0.8, label='Smoothed')
            plt.legend()
        
        plt.tight_layout()
        length_plot_path = output_path / "intersection_episode_length.png"
        plt.savefig(length_plot_path, dpi=300)
        print(f"✅ Saved: {length_plot_path}")
        plt.close()
    
    # 3. Training Losses (combined)
    loss_tags = {
        "train/entropy_loss": "Entropy Loss",
        "train/policy_gradient_loss": "Policy Loss",
        "train/value_loss": "Value Loss",
    }
    
    available_losses = {tag: name for tag, name in loss_tags.items() if tag in data}
    
    if available_losses:
        plt.figure(figsize=(12, 8))
        
        for i, (tag, name) in enumerate(available_losses.items()):
            steps, values = data[tag]
            plt.subplot(len(available_losses), 1, i + 1)
            plt.plot(steps, values, linewidth=1.5)
            plt.ylabel(name, fontsize=10)
            plt.grid(True, alpha=0.3)
            if i == 0:
                plt.title('Intersection RL Agent: Training Losses', fontsize=14, fontweight='bold')
            if i == len(available_losses) - 1:
                plt.xlabel('Timesteps', fontsize=12)
        
        plt.tight_layout()
        loss_plot_path = output_path / "intersection_training_losses.png"
        plt.savefig(loss_plot_path, dpi=300)
        print(f"✅ Saved: {loss_plot_path}")
        plt.close()
    
    print(f"\n✅ All plots saved to: {output_path}")


def main():
    """Main plotting function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot Intersection Training Results")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Path to TensorBoard log directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    # Default paths
    if args.log_dir is None:
        # Find most recent log directory
        logs_root = Path(INTERSECTION_PATHS["logs"])
        if not logs_root.exists():
            print(f"❌ Logs directory not found: {logs_root}")
            sys.exit(1)
        
        log_dirs = sorted(logs_root.glob("intersection_ppo_training*"))
        if not log_dirs:
            print(f"❌ No training logs found in: {logs_root}")
            sys.exit(1)
        
        log_dir = str(log_dirs[-1])  # Most recent
        print(f"📁 Using most recent log: {Path(log_dir).name}")
    else:
        log_dir = args.log_dir
    
    if args.output_dir is None:
        output_dir = INTERSECTION_PATHS["plots"]
    else:
        output_dir = args.output_dir
    
    # Generate plots
    print("\n" + "="*70)
    print("INTERSECTION TRAINING PLOT GENERATION")
    print("="*70)
    
    plot_training_metrics(log_dir, output_dir)
    
    print("\n" + "="*70)
    print("✅ PLOTTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
