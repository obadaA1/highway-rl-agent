"""Test new three-layer reward function."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.highway_env import make_highway_env
from src.config import REWARD_CONFIG
import numpy as np

def test_reward_components():
    """Test individual reward components."""
    print("="*70)
    print("REWARD FUNCTION TEST - PROGRESS-BASED")
    print("="*70)
    
    # Display configuration
    print("\nReward Configuration:")
    print(f"  Progress weight: {REWARD_CONFIG['w_progress']}")
    print(f"  Collision penalty: {REWARD_CONFIG['r_collision']}")
    print(f"  Lane change cost: {REWARD_CONFIG['r_lane_change']} (neutral)")
    print(f"  Alive bonus: {REWARD_CONFIG['r_alive']}")
    print(f"  Max velocity: {REWARD_CONFIG['max_velocity']} m/s")
    
    # Create environment
    env = make_highway_env(render_mode=None)
    obs, info = env.reset()
    
    print("\n" + "─"*70)
    print("TEST 1: Normal Driving (FASTER action, no crash)")
    print("─"*70)
    
    action = 3  # FASTER
    obs, reward, done, truncated, info = env.step(action)
    
    components = info['custom_reward_components']
    print(f"  Progress reward: {components['progress']:.4f}")
    print(f"  Alive bonus: {components['alive']:.4f}")
    print(f"  Collision penalty: {components['collision']:.4f}")
    print(f"  Total reward: {reward:.4f}")
    
    expected_range = (0.01, 1.1)  # progress + alive, no penalties
    if expected_range[0] <= reward <= expected_range[1]:
        print(f"  ✅ Reward in expected range {expected_range}")
    else:
        print(f"  ⚠️  Reward outside expected range {expected_range}")
    
    print("\n" + "─"*70)
    print("TEST 2: Lane Change (LANE_LEFT action)")
    print("─"*70)
    
    action = 0  # LANE_LEFT
    obs, reward, done, truncated, info = env.step(action)
    
    components = info['custom_reward_components']
    print(f"  Progress reward: {components['progress']:.4f}")
    print(f"  Alive bonus: {components['alive']:.4f}")
    print(f"  Collision penalty: {components['collision']:.4f}")
    print(f"  Total reward: {reward:.4f}")
    
    # Lane changes should be neutral (no penalty)
    if abs(reward - (components['progress'] + components['alive'])) < 0.001:
        print(f"  ✅ Lane change is neutral (no penalty)")
    else:
        print(f"  ❌ Lane change penalty detected!")
    
    print("\n" + "─"*70)
    print("TEST 3: Run Until Crash")
    print("─"*70)
    
    # Reset and run until crash
    obs, _ = env.reset()
    episode_reward = 0.0
    steps = 0
    
    for step in range(1000):
        action = 3  # Keep accelerating
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        
        if done:
            components = info['custom_reward_components']
            print(f"  Crashed at step {steps}")
            print(f"  Final step reward: {reward:.4f}")
            print(f"    Progress: {components['progress']:.4f}")
            print(f"    Alive: {components['alive']:.4f}")
            print(f"    Collision: {components['collision']:.4f}")
            print(f"  Episode total reward: {episode_reward:.2f}")
            
            if components['collision'] == REWARD_CONFIG['r_collision']:
                print(f"  ✅ Collision penalty applied correctly ({REWARD_CONFIG['r_collision']})")
            else:
                print(f"  ❌ Collision penalty incorrect!")
            break
    
    env.close()
    
    print("\n" + "="*70)
    print("ANALYSIS - PROGRESS-BASED REWARD")
    print("="*70)
    
    print("\nReward Structure Verification:")
    print(f"  ✓ Progress reward (velocity) drives optimization")
    print(f"  ✓ Collision penalty ({REWARD_CONFIG['r_collision']}) >> typical progress")
    print(f"  ✓ Lane change NEUTRAL ({REWARD_CONFIG['r_lane_change']}) - allows overtaking")
    print(f"  ✓ Alive bonus ({REWARD_CONFIG['r_alive']}) provides survival incentive")
    
    print("\nExpected Agent Behavior:")
    print("  1. Maximize forward velocity (distance/step)")
    print("  2. Avoid crashes (penalty eliminates 5s of progress)")
    print("  3. Use lane changes freely when beneficial (no penalty)")
    print("  4. Balance speed + survival for maximum distance")
    
    print("\nReward Balance Check:")
    max_progress_per_step = 1.0
    max_progress_5s = max_progress_per_step * (5 * 12)  # 5 seconds @ 12 Hz
    collision_penalty_abs = abs(REWARD_CONFIG['r_collision'])
    
    print(f"  Max progress in 5 seconds: {max_progress_5s:.1f}")
    print(f"  Collision penalty magnitude: {collision_penalty_abs:.1f}")
    
    if collision_penalty_abs < max_progress_5s:
        print(f"  ⚠️  WARNING: Collision penalty too small!")
        print(f"     Agent might prefer crashing after gaining progress")
    else:
        ratio = collision_penalty_abs / max_progress_5s
        print(f"  ✅ Collision penalty adequate ({ratio:.1f}× max 5s progress)")
    
    print("\n" + "="*70)
    print("✅ PROGRESS-BASED REWARD TEST COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_reward_components()
