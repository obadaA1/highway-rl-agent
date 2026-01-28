"""
Evaluation Script for Intersection RL Agent.

Purpose:
    Load trained checkpoint and evaluate performance over multiple episodes.
    
Outputs:
    - Mean/std reward
    - Episode length statistics
    - Crash rate
    - Goal reached rate
    - Success metrics
    
Compliance:
    - Type hints everywhere
    - Clean statistics reporting
    - Reproducible evaluation

Run: python scripts/evaluate_intersection.py
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.intersection_env_v1 import make_intersection_env_v1
from src.agent.ppo_agent import HighwayPPOAgent
from src.intersection_config import INTERSECTION_CHECKPOINT_CONFIG
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
    env = make_intersection_env_v1(render_mode="human" if render else None)
    
    # Load agent
    print("📦 Loading checkpoint...")
    # Check if this is the untrained checkpoint
    checkpoint_name = Path(checkpoint_path).stem
    if checkpoint_name.endswith("_0_steps") or checkpoint_name == "intersection_ppo_0_steps":
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
    episode_goals_reached: List[int] = []
    episode_action_counts: List[Dict[str, int]] = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        episode_length = 0
        crashed = False
        goal_reached = False
        
        # Track action counts: Intersection typically has fewer actions
        action_counts = {
            'action_0': 0,  # Slower
            'action_1': 0,  # Idle/Coast
            'action_2': 0,  # Faster
        }
        
        while not (done or truncated):
            # Get action
            if use_random:
                action = env.action_space.sample()
            else:
                action, _ = agent.model.predict(obs, deterministic=True)
                if isinstance(action, np.ndarray):
                    action = int(action.item())
            
            # Count action
            if action < 3:
                action_counts[f'action_{action}'] += 1
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Check termination reasons
            if done:
                # Check if crashed or reached goal
                crashed = info.get('crashed', False) or info.get('episode_stats', {}).get('crashes', 0) > 0
                goal_reached = info.get('episode_stats', {}).get('reached_goal', 0) > 0
        
        # Store episode stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_crashes.append(1 if crashed else 0)
        episode_goals_reached.append(1 if goal_reached else 0)
        episode_action_counts.append(action_counts)
        
        # Progress update
        if (ep + 1) % 10 == 0:
            print(f"   Completed {ep + 1}/{n_episodes} episodes...")
    
    env.close()
    
    # Compute statistics
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # Rewards
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    
    print(f"\n📊 Reward Statistics:")
    print(f"   Mean: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"   Min:  {min_reward:.2f}")
    print(f"   Max:  {max_reward:.2f}")
    
    # Episode lengths
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    min_length = np.min(episode_lengths)
    max_length = np.max(episode_lengths)
    
    print(f"\n⏱️ Episode Length Statistics:")
    print(f"   Mean: {mean_length:.1f} ± {std_length:.1f} steps")
    print(f"   Min:  {min_length} steps")
    print(f"   Max:  {max_length} steps")
    
    # Crash rate
    crash_rate = np.mean(episode_crashes) * 100
    print(f"\n💥 Crash Rate: {crash_rate:.1f}%")
    print(f"   Crashes: {np.sum(episode_crashes)}/{n_episodes} episodes")
    
    # Goal success rate
    success_rate = np.mean(episode_goals_reached) * 100
    print(f"\n🎯 Goal Success Rate: {success_rate:.1f}%")
    print(f"   Goals Reached: {np.sum(episode_goals_reached)}/{n_episodes} episodes")
    
    # Action distribution
    total_actions = {f'action_{i}': 0 for i in range(3)}
    for counts in episode_action_counts:
        for action, count in counts.items():
            if action in total_actions:
                total_actions[action] += count
    
    total_action_count = sum(total_actions.values())
    if total_action_count > 0:
        print(f"\n🎮 Action Distribution:")
        action_names = {
            'action_0': 'SLOWER',
            'action_1': 'IDLE',
            'action_2': 'FASTER',
        }
        for action, count in sorted(total_actions.items()):
            percentage = (count / total_action_count) * 100
            name = action_names.get(action, action)
            print(f"   {name}: {percentage:.1f}% ({count:,} times)")
    
    print("\n" + "="*70 + "\n")
    
    # Return metrics dictionary
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "std_length": std_length,
        "crash_rate": crash_rate,
        "success_rate": success_rate,
    }


def main() -> None:
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Intersection RL Agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.zip)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate (default: 100)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes (much slower)"
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        
        # Try to find it in default checkpoint directory
        default_path = Path(INTERSECTION_CHECKPOINT_CONFIG["save_path"]) / checkpoint_path.name
        if default_path.exists():
            print(f"✅ Found in default directory: {default_path}")
            checkpoint_path = default_path
        else:
            sys.exit(1)
    
    # Run evaluation
    evaluate_checkpoint(
        checkpoint_path=str(checkpoint_path),
        n_episodes=args.episodes,
        render=args.render,
    )


if __name__ == "__main__":
    main()
