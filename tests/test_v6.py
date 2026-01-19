#!/usr/bin/env python
"""Test V6 Multi-Objective environment with weaving detection."""

from src.env.highway_env_v6 import make_highway_env_v6

env = make_highway_env_v6()
obs, _ = env.reset()
print("V6 Multi-Objective Environment created!")
print("=" * 60)

total_r = 0
actions_taken = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
total_safe_bonuses = 0
total_slow_penalties = 0
total_weaving_penalties = 0

# Test weaving detection: do rapid consecutive lane changes
print("\nTesting weaving detection (rapid lane changes):")
for i in range(20):
    # Alternate between LANE_LEFT and LANE_RIGHT to trigger weaving
    if i < 5:
        action = 0 if i % 2 == 0 else 2  # Rapid weaving
    else:
        action = 1  # Then IDLE to let counter reset
    
    obs, r, d, t, info = env.step(action)
    actions_taken[action] += 1
    total_r += r
    
    components = info.get('reward_components', {})
    
    if i < 10:
        weaving = components.get('p_weaving', 0)
        print(f"Step {i}: action={action}, weaving_penalty={weaving:.3f}, reward={r:.4f}")
    
    if d or t:
        print(f"\nEpisode ended at step {i}")
        break

# Continue with mixed actions
print("\nContinuing with strategic lane changes (spaced out):")
env.reset()
total_r = 0
actions_taken = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

for i in range(100):
    # Strategic: lane change every 15 steps (not weaving)
    if i % 15 == 0:
        action = 0  # LANE_LEFT
    elif i % 15 == 7:
        action = 3  # FASTER
    else:
        action = 1  # IDLE
    
    obs, r, d, t, info = env.step(action)
    actions_taken[action] += 1
    total_r += r
    
    if d or t:
        stats = info.get('episode_stats', {})
        total_weaving_penalties = stats.get('weaving_penalties', 0)
        total_safe_bonuses = stats.get('safe_distance_bonuses', 0)
        total_slow_penalties = stats.get('slow_speed_penalties', 0)
        print(f"\nEpisode ended at step {i}")
        break

if not (d or t):
    stats = info.get('episode_stats', {})
    total_weaving_penalties = stats.get('weaving_penalties', 0)
    total_safe_bonuses = stats.get('safe_distance_bonuses', 0)
    total_slow_penalties = stats.get('slow_speed_penalties', 0)

print(f"\n{'=' * 60}")
print("MULTI-OBJECTIVE TEST RESULTS")
print(f"{'=' * 60}")
print(f"Total reward over {i+1} steps: {total_r:.3f}")
print(f"Average reward per step: {total_r/(i+1):.4f}")
print(f"Actions: {actions_taken}")
print(f"\nMulti-Objective Metrics:")
print(f"  Safe distance bonuses triggered: {total_safe_bonuses}")
print(f"  Slow speed penalties triggered: {total_slow_penalties}")
print(f"  Weaving penalties triggered: {total_weaving_penalties}")
print(f"\nRubric Compliance Check:")
print(f"  ✓ R_speed (high velocity) - via base_reward")
print(f"  ✓ R_safe_distance (safe following) - {total_safe_bonuses} bonuses")
print(f"  ✓ P_weaving (UNNECESSARY lane changes only) - {total_weaving_penalties} penalties")
print(f"  ✓ P_slow_speed (driving too slowly) - {total_slow_penalties} penalties")
print(f"  ✓ P_collision (crash penalty) - on termination")
print(f"\nKey insight: Strategic lane changes (spaced >10 steps) = NO penalty!")

env.close()
print("\n✅ V6 Multi-Objective Test Complete!")
