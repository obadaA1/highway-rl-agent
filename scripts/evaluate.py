"""
Evaluation Script for Highway RL Agent.

Purpose:
    Load trained checkpoint and evaluate performance over multiple episodes.
    
Outputs:
    - Mean/std reward
    - Episode length statistics
    - Crash rate
    - Lane change frequency
    
Compliance:
    - Type hints everywhere
    - Clean statistics reporting
    - Reproducible evaluation

Run: python scripts/evaluate.py
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.highway_env import make_highway_env
from src.agent.ppo_agent import HighwayPPOAgent
from src.config import CHECKPOINT_CONFIG
import numpy as np


def evaluate_checkpoint(
    checkpoint_path: str,
    n_episodes: int = 100,
    render: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a trained checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint .zip file
        n_episodes: Number of episodes to evaluate
        render: Whether to render episodes (slows down evaluation)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*70)
    print(f"EVALUATING: {Path(checkpoint_path).name}")
    print("="*70)
    print(f"Episodes: {n_episodes}")
    print(f"Rendering: {render}")
    print("="*70 + "\n")
    
    # Create environment
    env = make_highway_env(render_mode="human" if render else None)
    
    # Load agent
    print("📦 Loading checkpoint...")
    # Check if this is the untrained checkpoint
    checkpoint_name = Path(checkpoint_path).stem
    if checkpoint_name.endswith("_0_steps") or checkpoint_name == "highway_ppo_0_steps":
        agent = HighwayPPOAgent(env=env, verbose=0)
        print("✅ Using untrained (random) policy\n")
        use_random = True
    else:
        agent = HighwayPPOAgent.load(checkpoint_path, env=env)
        print("✅ Checkpoint loaded\n")
        use_random = False
    
    # Run evaluation
    print(f"🏃 Running {n_episodes} episodes...")
    
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_crashes: List[int] = []
    episode_lane_changes: List[int] = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        episode_length = 0
        crashed = False
        lane_changes = 0
        previous_action = None
        
        while not (done or truncated):
            if use_random:
                action = env.action_space.sample()
            else:
                action, _ = agent.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Track crashes
            if info.get('crashed', False):
                crashed = True
            
            # Track lane changes
            if previous_action is not None and action in [0, 2]:  # LANE_LEFT or LANE_RIGHT
                lane_changes += 1
            previous_action = action
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_crashes.append(1 if crashed else 0)
        episode_lane_changes.append(lane_changes)
        
        # Progress update every 10 episodes
        if (ep + 1) % 10 == 0:
            print(f"  Progress: {ep + 1}/{n_episodes} episodes completed")
    
    env.close()
    
    # Compute statistics
    stats = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'crash_rate': float(np.mean(episode_crashes)),
        'mean_lane_changes': float(np.mean(episode_lane_changes)),
        'std_lane_changes': float(np.std(episode_lane_changes)),
    }
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\n📊 Reward Statistics:")
    print(f"  Mean: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
    print(f"  Range: [{stats['min_reward']:.3f}, {stats['max_reward']:.3f}]")
    print(f"\n🏁 Episode Length:")
    print(f"  Mean: {stats['mean_length']:.1f} ± {stats['std_length']:.1f} steps")
    print(f"\n💥 Safety Metrics:")
    print(f"  Crash Rate: {stats['crash_rate']*100:.1f}%")
    print(f"\n🚗 Efficiency Metrics:")
    print(f"  Lane Changes: {stats['mean_lane_changes']:.1f} ± {stats['std_lane_changes']:.1f} per episode")
    print("="*70 + "\n")
    
    return stats


def main() -> None:
    """
    Evaluate all three checkpoints (untrained, half-trained, fully-trained).
    """
    checkpoint_dir = Path(CHECKPOINT_CONFIG["save_path"])
    
    # Define checkpoints to evaluate
    checkpoints = [
        checkpoint_dir / "highway_ppo_0_steps.zip",      # Untrained
        checkpoint_dir / "highway_ppo_50000_steps.zip",  # Half-trained
        checkpoint_dir / "highway_ppo_100000_steps.zip", # Fully-trained
    ]
    
    results = {}
    
    for checkpoint_path in checkpoints:
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            print(f"   Run training first: python scripts/train.py\n")
            continue
        
        stats = evaluate_checkpoint(
            checkpoint_path=str(checkpoint_path),
            n_episodes=100,  # 100 episodes for statistical significance
            render=False,    # Set to True to watch agent play
        )
        
        results[checkpoint_path.name] = stats
    
    # Comparative summary
    if len(results) > 1:
        print("\n" + "="*70)
        print("COMPARATIVE SUMMARY")
        print("="*70)
        print(f"\n{'Checkpoint':<35} {'Mean Reward':<15} {'Crash Rate':<15}")
        print("-"*70)
        for name, stats in results.items():
            print(f"{name:<35} {stats['mean_reward']:>10.3f}     {stats['crash_rate']*100:>10.1f}%")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
