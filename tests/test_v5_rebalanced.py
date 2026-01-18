#!/usr/bin/env python
"""Test V5 rebalanced reward function."""

from src.env.highway_env_v5 import make_highway_env_v5


def test_asymmetric_lane_penalties():
    """Test that LANE_LEFT has smaller penalty than LANE_RIGHT."""
    env = make_highway_env_v5()
    
    # Test LANE_LEFT (action 0)
    obs, _ = env.reset()
    _, _, _, _, info_left = env.step(0)  # LANE_LEFT
    lane_left = info_left["custom_reward_components"]["lane"]
    
    # Test LANE_RIGHT (action 2)
    obs, _ = env.reset()
    _, _, _, _, info_right = env.step(2)  # LANE_RIGHT
    lane_right = info_right["custom_reward_components"]["lane"]
    
    print(f"LANE_LEFT penalty: {lane_left}")
    print(f"LANE_RIGHT penalty: {lane_right}")
    
    # Asymmetric: LEFT should be smaller penalty (closer to 0)
    assert lane_left == -0.01, f"Expected -0.01, got {lane_left}"
    assert lane_right == -0.03, f"Expected -0.03, got {lane_right}"
    assert lane_left > lane_right, "LEFT should be smaller penalty (less negative)"
    
    print("✓ Asymmetric lane penalties working correctly!")
    env.close()


def test_headway_no_empty_lane_bonus():
    """Test that empty lanes don't give bonus (only penalties)."""
    env = make_highway_env_v5()
    
    # Run several steps and check headway component
    obs, _ = env.reset()
    
    # When there's no car ahead, headway should be 0, not +0.10
    for _ in range(10):
        action = 1  # IDLE
        obs, reward, done, truncated, info = env.step(action)
        headway = info["custom_reward_components"]["headway"]
        
        # Headway should be 0 (neutral) or negative (penalty for tailgating)
        # Never positive (no bonus for empty lanes)
        assert headway <= 0.0, f"Headway should be <= 0, got {headway}"
        
        if done or truncated:
            obs, _ = env.reset()
    
    print("✓ No empty lane bonus (headway ≤ 0)!")
    env.close()


def test_proportional_faster_bonus():
    """Test that FASTER bonus is proportional to speed deficit."""
    env = make_highway_env_v5()
    
    # At start, velocity should be high, so FASTER bonus might be 0
    obs, _ = env.reset()
    
    # Use SLOWER action to reduce speed, then test FASTER
    for _ in range(5):
        obs, _, done, truncated, _ = env.step(4)  # SLOWER
        if done or truncated:
            obs, _ = env.reset()
    
    # Now test FASTER at lower speed
    obs, _, done, truncated, info = env.step(3)  # FASTER
    faster_bonus = info["custom_reward_components"]["faster"]
    velocity_ratio = info["custom_reward_components"]["velocity_ratio"]
    
    print(f"Velocity ratio: {velocity_ratio:.3f}")
    print(f"FASTER bonus: {faster_bonus:.4f}")
    
    # If velocity < 90%, should get proportional bonus
    if velocity_ratio < 0.9:
        expected_bonus = 0.15 * (1 - velocity_ratio)
        assert abs(faster_bonus - expected_bonus) < 0.01, \
            f"Expected ~{expected_bonus:.4f}, got {faster_bonus:.4f}"
        print(f"✓ Proportional FASTER bonus: {faster_bonus:.4f}")
    else:
        print(f"  (Velocity at {velocity_ratio:.1%}, bonus is 0)")
    
    env.close()


if __name__ == "__main__":
    test_asymmetric_lane_penalties()
    print()
    test_headway_no_empty_lane_bonus()
    print()
    test_proportional_faster_bonus()
    print("\n✅ All V5 rebalanced tests passed!")
