#!/usr/bin/env python
"""Test V6 environment."""

from src.env.highway_env_v6 import make_highway_env_v6

env = make_highway_env_v6()
obs, _ = env.reset()
print("V6 Environment created!")

total_r = 0
actions_taken = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

for i in range(100):
    # Test with different actions
    action = i % 5  # Cycle through all actions
    obs, r, d, t, info = env.step(action)
    actions_taken[action] += 1
    total_r += r
    
    if i < 10:
        print(f"Step {i}: action={action}, reward={r:.4f}, base={info.get('base_reward', 0):.4f}")
    
    if d or t:
        print(f"\nEpisode ended at step {i}")
        print(f"Crashed: {info.get('episode_stats', {}).get('crashes', 'N/A')}")
        break

print(f"\nTotal reward over {i+1} steps: {total_r:.3f}")
print(f"Average reward per step: {total_r/(i+1):.4f}")
print(f"Actions: {actions_taken}")
print(f"\nExpected: Rewards in [0, 1] range (mostly positive)")

env.close()
print("\n✅ V6 Test Complete!")
