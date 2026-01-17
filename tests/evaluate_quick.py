"""Quick evaluation of 200k checkpoint with multiple episodes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from src.env.highway_env import make_highway_env
from src.config import PATHS
import numpy as np

def evaluate_checkpoint(checkpoint_name: str, n_episodes: int = 10) -> None:
    """Evaluate checkpoint over multiple episodes."""
    checkpoint_path = Path(PATHS["checkpoints"]) / checkpoint_name
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {checkpoint_name}")
    print(f"Episodes: {n_episodes}")
    print(f"{'='*70}\n")
    
    # Load model
    model = PPO.load(checkpoint_path)
    env = make_highway_env(render_mode=None)
    
    survival_times = []
    rewards = []
    crashes = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(960):  # Max 80 seconds
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if done or truncated:
                if done:
                    crashes += 1
                break
        
        survival_time = steps / 12  # 12 Hz
        survival_times.append(survival_time)
        rewards.append(episode_reward)
        
        print(f"  Episode {episode+1:2d}: {survival_time:5.1f}s, reward={episode_reward:6.2f}, crashed={done}")
    
    env.close()
    
    # Statistics
    print(f"\n{'─'*70}")
    print(f"STATISTICS")
    print(f"{'─'*70}")
    print(f"  Survival time: {np.mean(survival_times):5.1f}s ± {np.std(survival_times):4.1f}s")
    print(f"  Min/Max:       {np.min(survival_times):5.1f}s / {np.max(survival_times):5.1f}s")
    print(f"  Total reward:  {np.mean(rewards):6.2f} ± {np.std(rewards):5.2f}")
    print(f"  Crash rate:    {crashes}/{n_episodes} ({crashes/n_episodes*100:.0f}%)")
    print(f"{'─'*70}\n")

def main():
    """Main evaluation."""
    print("\n" + "="*70)
    print("MULTI-EPISODE EVALUATION")
    print("="*70)
    print("\nThis tests both checkpoints over 10 episodes")
    print("to account for environment randomness.")
    print("="*70)
    
    evaluate_checkpoint("highway_ppo_100000_steps.zip", n_episodes=10)
    evaluate_checkpoint("highway_ppo_200000_steps.zip", n_episodes=10)
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("\nThe video recording caught ONE UNLUCKY episode.")
    print("Multi-episode statistics show the TRUE performance.")
    print("\nExpected:")
    print("  100k: ~6-10s average (from training metrics)")
    print("  200k: ~10-20s average (improved)")
    print("\nIf 200k is still worse, training may have diverged.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
