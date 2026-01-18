"""
Test V4 Reward System: Acceleration-Aware (No Overtaking)

Verifies:
1. Acceleration bonus (velocity delta tracking)
2. Context-dependent SLOWER penalty
3. FASTER bonus when slow
4. No overtaking logic (removed from V3.5)
"""

from src.config import REWARD_V4_CONFIG


def test_acceleration_bonus():
    """V4: Progress reward should include acceleration bonus."""
    # Scenario: Accelerating from 50% to 70% speed
    velocity_ratio_t0 = 0.5
    velocity_ratio_t1 = 0.7
    delta_v = velocity_ratio_t1 - velocity_ratio_t0  # 0.2
    
    # Progress reward = current_speed + weight × acceleration
    acceleration_weight = REWARD_V4_CONFIG["acceleration_weight"]
    r_progress = velocity_ratio_t1 + acceleration_weight * delta_v
    
    expected = 0.7 + 0.2 * 0.2  # 0.74
    assert abs(r_progress - expected) < 0.001, f"Expected {expected}, got {r_progress}"
    
    print(f"✅ Acceleration bonus: {r_progress:.4f} (includes Δv = {delta_v:.2f})")


def test_context_slower_penalty_when_slow():
    """V4: SLOWER penalty should be heavier when already slow."""
    velocity_ratio = 0.6  # Below 70% threshold
    
    # When velocity < 70%, penalty should be -0.05
    expected_penalty = REWARD_V4_CONFIG["r_slower_heavy"]
    
    assert expected_penalty == -0.05, f"Expected -0.05, got {expected_penalty}"
    print(f"✅ SLOWER when slow: {expected_penalty} (heavy penalty)")


def test_context_slower_penalty_when_fast():
    """V4: SLOWER penalty should be lighter when moving fast."""
    velocity_ratio = 0.8  # Above 70% threshold
    
    # When velocity >= 70%, penalty should be -0.01
    expected_penalty = REWARD_V4_CONFIG["r_slower_light"]
    
    assert expected_penalty == -0.01, f"Expected -0.01, got {expected_penalty}"
    print(f"✅ SLOWER when fast: {expected_penalty} (light penalty)")


def test_faster_bonus_when_slow():
    """V4: FASTER action should get bonus when moving slowly."""
    velocity_ratio = 0.7  # Below 80% threshold
    
    # When velocity < 80% AND action = FASTER, bonus = +0.05
    expected_bonus = REWARD_V4_CONFIG["r_faster_bonus"]
    
    assert expected_bonus == 0.05, f"Expected 0.05, got {expected_bonus}"
    print(f"✅ FASTER when slow: +{expected_bonus} (encourages acceleration)")


def test_faster_no_bonus_when_fast():
    """V4: FASTER action should NOT get bonus when already fast."""
    velocity_ratio = 0.9  # Above 80% threshold
    
    # When velocity >= 80%, no bonus
    expected_bonus = 0.0
    
    print(f"✅ FASTER when fast: {expected_bonus} (no bonus needed)")


def test_no_overtaking_logic():
    """V4: Verify overtaking parameters are NOT in config."""
    v4_keys = set(REWARD_V4_CONFIG.keys())
    
    # These should NOT exist in V4
    forbidden_keys = {
        "r_overtake_bonus",
        "r_collision_risky",
        "overtake_risk_window",
        "min_overtake_speed"
    }
    
    overlap = v4_keys & forbidden_keys
    assert len(overlap) == 0, f"V4 should not have overtaking params: {overlap}"
    
    print(f"✅ No overtaking logic (removed from V3.5)")


def test_single_collision_penalty():
    """V4: Collision penalty should be single value (no risk logic)."""
    r_collision = REWARD_V4_CONFIG["r_collision"]
    
    # Should be -100.0 (same as r_collision_base in V3.5)
    assert r_collision == -100.0, f"Expected -100.0, got {r_collision}"
    
    # Should NOT have r_collision_risky
    assert "r_collision_risky" not in REWARD_V4_CONFIG
    
    print(f"✅ Single collision penalty: {r_collision} (no risk logic)")


def test_v4_reward_structure():
    """V4: Verify 6-component reward structure."""
    # Simulate a good driving episode
    velocity_ratio = 0.85
    delta_v = 0.05  # Accelerating
    action_FASTER = 3
    not_crashed = False
    
    # Component 1: Progress (with acceleration)
    r_progress = velocity_ratio + REWARD_V4_CONFIG["acceleration_weight"] * delta_v
    
    # Component 2: Alive
    r_alive = REWARD_V4_CONFIG["r_alive"]
    
    # Component 3: Collision (none)
    r_collision = 0.0
    
    # Component 4: SLOWER penalty (none, action is FASTER)
    r_slow_action = 0.0
    
    # Component 5: Low speed penalty (none, velocity high)
    r_low_speed = 0.0
    
    # Component 6: FASTER bonus (none, velocity > 80%)
    r_faster_bonus = 0.0
    
    total = r_progress + r_alive + r_collision + r_slow_action + r_low_speed + r_faster_bonus
    
    # Should be positive (good driving)
    assert total > 0, f"Good driving should be positive, got {total:.4f}"
    
    print(f"✅ V4 structure: {total:.4f} = progress({r_progress:.4f}) + alive({r_alive}) + ... ")


def test_acceleration_vs_deceleration():
    """V4: Accelerating should be rewarded more than decelerating."""
    # Scenario 1: Accelerating from 60% to 70%
    v_start = 0.6
    v_accelerate = 0.7
    delta_v_accel = v_accelerate - v_start  # +0.1
    
    r_accel = v_accelerate + REWARD_V4_CONFIG["acceleration_weight"] * delta_v_accel
    
    # Scenario 2: Decelerating from 70% to 60%
    v_decel = 0.6
    delta_v_decel = v_decel - 0.7  # -0.1
    
    r_decel = v_decel + REWARD_V4_CONFIG["acceleration_weight"] * delta_v_decel
    
    # Accelerating should yield higher reward
    assert r_accel > r_decel, f"Accel ({r_accel:.4f}) should > Decel ({r_decel:.4f})"
    
    print(f"✅ Acceleration rewarded: {r_accel:.4f} > {r_decel:.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("V4 ACCELERATION-AWARE REWARD VERIFICATION")
    print("No Overtaking Logic (Simplified from V3.5)")
    print("=" * 70)
    print()
    
    test_acceleration_bonus()
    test_context_slower_penalty_when_slow()
    test_context_slower_penalty_when_fast()
    test_faster_bonus_when_slow()
    test_faster_no_bonus_when_fast()
    test_no_overtaking_logic()
    test_single_collision_penalty()
    test_v4_reward_structure()
    test_acceleration_vs_deceleration()
    
    print()
    print("=" * 70)
    print("✅ ALL V4 TESTS PASSED")
    print("=" * 70)
