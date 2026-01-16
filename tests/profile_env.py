"""
Profile highway-env to find the bottleneck.
Uses cProfile to identify which function is slow.

Run: python tests\profile_env.py
"""

import sys
import cProfile
import pstats
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym


def run_environment():
    """Run 50 environment steps."""
    env = gym.make("highway-v0", render_mode=None)
    obs, _ = env.reset(seed=42)
    
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, _ = env.reset()
    
    env.close()


if __name__ == "__main__":
    print("\nProfiling 50 environment steps...")
    print("This will take ~20 seconds...\n")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    run_environment()
    
    profiler.disable()
    
    # Print top 20 slowest functions
    print("\n" + "="*70)
    print("TOP 20 SLOWEST FUNCTIONS")
    print("="*70 + "\n")
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
