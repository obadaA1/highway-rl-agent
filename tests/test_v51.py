#!/usr/bin/env python
"""Test V5.1 rebalanced reward function."""

from src.env.highway_env_v5 import make_highway_env_v5


def test_v51_rewards():
    """Test V5.1 reward changes."""
    env = make_highway_env_v5()
    
    print("=== V5.1 Reward Test ===\n")
    
    # Test SLOWER at high speed (should be -0.08 now)
    obs, _ = env.reset()
    obs, _, _, _, info = env.step(4)  # SLOWER
    slow_action = info["custom_reward_components"]["slow_action"]
    print(f"SLOWER at high speed: slow_action = {slow_action}")
    assert slow_action == -0.08, f"Expected -0.08, got {slow_action}"
    
    # Test IDLE at high speed (should be +0.05 now)
    obs, _ = env.reset()
    obs, _, _, _, info = env.step(1)  # IDLE  
    idle_bonus = info["custom_reward_components"]["idle"]
    v_ratio = info["custom_reward_components"]["velocity_ratio"]
    print(f"IDLE at v={v_ratio:.2f}: idle_bonus = {idle_bonus}")
    if v_ratio >= 0.8:
        assert idle_bonus == 0.05, f"Expected 0.05, got {idle_bonus}"
    
    # Test FASTER (should be proportional, max 0.25)
    obs, _ = env.reset()
    for _ in range(10):
        obs, done, _, trunc, _ = env.step(4)  # Slow down first
        if done or trunc:
            obs, _ = env.reset()
    
    obs, _, _, _, info = env.step(3)  # FASTER
    v = info["custom_reward_components"]["velocity_ratio"]
    faster = info["custom_reward_components"]["faster"]
    print(f"FASTER at v={v:.2f}: faster_bonus = {faster:.4f}")
    if v < 0.95:
        expected = 0.25 * (1 - v)
        assert abs(faster - expected) < 0.01, f"Expected ~{expected:.4f}, got {faster:.4f}"
    
    # Test collision penalty (should be -50 now)
    from src.config import REWARD_V5_CONFIG
    collision = REWARD_V5_CONFIG["r_collision"]
    print(f"\nCollision penalty: {collision}")
    assert collision == -50.0, f"Expected -50.0, got {collision}"
    
    print("\n✅ All V5.1 tests passed!")
    env.close()


if __name__ == "__main__":
    test_v51_rewards()
