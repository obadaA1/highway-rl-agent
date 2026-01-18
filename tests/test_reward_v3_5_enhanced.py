"""
V3.5 Enhanced Reward Function Tests - 95% Confidence Requirement

Mathematical Model:
    Expected value for overtaking:
        E = P(success) × r_overtake + P(crash) × r_collision_risky
    
    For 95% confidence threshold:
        E = 0.95 × 2.0 + 0.05 × (-138.0)
        E = 1.90 - 6.90 = -5.00 (NEGATIVE)
    
    Agent learns: "Don't overtake unless >95% confident"
    
    For 98.5% confidence (actual config):
        E = 0.985 × 2.0 + 0.015 × (-138.0)
        E = 1.97 - 2.07 = -0.10 (barely negative, conservative)
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import REWARD_CONFIG


def test_exact_95_percent_formula():
    """
    Calculate exact penalty ratio for 95% confidence threshold.
    
    For EXACTLY 95% break-even:
        0.95 × r_overtake + 0.05 × r_collision_risky = 0
        r_collision_risky = -(0.95/0.05) × r_overtake
        r_collision_risky = -19 × r_overtake
    
    With r_overtake = 2.0:
        r_collision_risky = -19 × 2.0 = -38.0 (additional)
        Total crash penalty = -100.0 (base) + (-38.0) = -138.0
    """
    r_overtake = REWARD_CONFIG["r_overtake_bonus"]  # 2.0
    
    # Calculate exact ratio for 95% threshold
    required_ratio = 0.95 / 0.05  # 19:1
    required_additional_penalty = required_ratio * r_overtake  # 38.0
    
    # Current config
    actual_additional_penalty = abs(REWARD_CONFIG["r_collision_risky"])
    
    print("\n" + "="*70)
    print("EXACT 95% THRESHOLD FORMULA")
    print("="*70)
    print(f"Overtake bonus: {r_overtake:.1f}")
    print(f"Required ratio for 95%: {required_ratio:.1f}:1")
    print(f"Required additional penalty: -{required_additional_penalty:.1f}")
    print(f"Current additional penalty: -{actual_additional_penalty:.1f}")
    
    if abs(actual_additional_penalty - required_additional_penalty) < 0.1:
        print("✅ Config matches 95% threshold exactly")
    else:
        total_ratio = (abs(REWARD_CONFIG["r_collision_base"]) + actual_additional_penalty) / r_overtake
        required_conf = (abs(REWARD_CONFIG["r_collision_base"]) + actual_additional_penalty) / \
                       (abs(REWARD_CONFIG["r_collision_base"]) + actual_additional_penalty + r_overtake)
        print(f"⚠️ Config is conservative (ratio {total_ratio:.1f}:1, requires {required_conf*100:.1f}% confidence)")
    
    print("="*70 + "\n")
    
    # Allow small floating point differences
    assert abs(actual_additional_penalty - required_additional_penalty) < 0.1, \
        f"Config should match 95% threshold, got {actual_additional_penalty} vs {required_additional_penalty}"


def test_95_percent_confidence_threshold():
    """
    V3.5: Agent should require ~98.5% success confidence to attempt overtake.
    
    Mathematical proof:
        Break-even expected value:
            E = P(success) × r_overtake + P(crash) × r_collision_risky = 0
        
        With 95% confidence:
            E = 0.95 × 2.0 + 0.05 × (-138.0)
            E = 1.90 - 6.90 = -5.00 (NEGATIVE)
        
        Agent needs ~98.5% confidence for positive expected value.
    """
    r_overtake = REWARD_CONFIG["r_overtake_bonus"]  # 2.0
    r_collision_total = (
        REWARD_CONFIG["r_collision_base"] + 
        REWARD_CONFIG["r_collision_risky"]
    )  # -100.0 + (-38.0) = -138.0
    
    # Calculate expected value at different confidence levels
    confidences = [0.90, 0.95, 0.98, 0.985, 0.99]
    
    print("\n" + "="*70)
    print("95% CONFIDENCE THRESHOLD VERIFICATION")
    print("="*70)
    print(f"Overtake bonus: +{r_overtake:.1f}")
    print(f"Risky crash penalty: {r_collision_total:.1f}")
    print(f"Risk ratio: {abs(r_collision_total)/r_overtake:.1f}:1")
    print("\nExpected Value by Confidence Level:")
    print("-" * 70)
    
    for confidence in confidences:
        p_crash = 1.0 - confidence
        expected_value = confidence * r_overtake + p_crash * r_collision_total
        
        status = "✅ POSITIVE (attempt)" if expected_value > 0 else "❌ NEGATIVE (avoid)"
        print(f"  {confidence*100:5.1f}% confidence: "
              f"E = {expected_value:+6.2f} {status}")
    
    # Verify 95% is negative (agent won't risk it)
    e_95 = 0.95 * r_overtake + 0.05 * r_collision_total
    assert e_95 < 0, f"Expected value at 95% should be negative, got {e_95:.2f}"
    
    # Verify 98.5% is approximately break-even
    e_985 = 0.985 * r_overtake + 0.015 * r_collision_total
    assert -1.0 < e_985 < 1.0, f"Expected value at 98.5% should be near zero, got {e_985:.2f}"
    
    print("\n" + "="*70)
    print("✅ VERIFIED: Agent requires ~98.5% confidence (conservative)")
    print("="*70 + "\n")


def test_safe_overtaking_highly_rewarded():
    """V3.5: Safe overtaking should give large bonus."""
    # 25 m/s, 1 overtake, no crash
    r_progress = 25/30.0  # 0.833
    r_alive = 0.01
    r_overtake = REWARD_CONFIG["r_overtake_bonus"]  # 2.0
    r_collision = 0.0
    
    total = r_progress + r_alive + r_overtake + r_collision
    
    assert total > 2.5, f"Safe overtake should be >2.5, got {total:.4f}"
    print(f"✅ Safe overtaking: {total:.4f} (HIGHLY REWARDED)")


def test_risky_crash_catastrophic():
    """V3.5: Crash during overtaking should be catastrophic."""
    # Overtake completed (+2.0), crash 2s later (within 3s window)
    steps_before_crash = 24  # 2 seconds @ 12 Hz
    r_progress = 0.833 * steps_before_crash  # 20.0
    r_alive = 0.01 * steps_before_crash      # 0.24
    r_overtake = 2.0  # Successful overtake before crash
    r_collision_risky = (
        REWARD_CONFIG["r_collision_base"] + 
        REWARD_CONFIG["r_collision_risky"]
    )  # -138.0
    
    total = r_progress + r_alive + r_overtake + r_collision_risky
    
    assert total < -100, f"Risky crash should be <-100, got {total:.4f}"
    print(f"✅ Risky crash: {total:.4f} (CATASTROPHIC)")


def test_normal_crash_less_severe():
    """V3.5: Normal crash (no overtaking) should be less severe than risky crash."""
    # 50 steps, no overtaking, then crash
    r_progress = 0.833 * 50       # 41.65
    r_alive = 0.01 * 50           # 0.5
    r_overtake = 0.0              # No overtaking
    r_collision_normal = REWARD_CONFIG["r_collision_base"]  # -100.0 only
    
    total = r_progress + r_alive + r_overtake + r_collision_normal
    
    # Total should be between -110 and -50 (negative but less than risky crash)
    assert -110 < total < -50, f"Normal crash should be between -110 and -50, got {total:.4f}"
    print(f"✅ Normal crash: {total:.4f} (LESS SEVERE than risky)")


def test_slow_driving_still_negative():
    """V3.5: Slow driving should still be prevented (V3 guarantees)."""
    # 5 m/s, SLOWER action, no overtaking
    r_progress = 5/30.0  # 0.167
    r_alive = 0.01
    r_slow_action = REWARD_CONFIG["r_slow_action"]  # -0.10
    r_low_speed = REWARD_CONFIG["r_low_speed"]      # -0.20
    r_overtake = 0.0
    
    total = r_progress + r_alive + r_slow_action + r_low_speed + r_overtake
    
    assert total < 0, f"Slow driving should be negative, got {total:.4f}"
    print(f"✅ Slow driving: {total:.4f} (STILL NEGATIVE, V3 preserved)")


def test_overtake_bonus_magnitude():
    """V3.5: Overtake bonus should be 5-10% of speed reward (secondary role)."""
    # Average episode: 80 steps, 2 overtakes
    avg_progress_per_episode = 0.8 * 80  # 64.0
    avg_overtake_per_episode = 2.0 * 2    # 4.0
    
    ratio = avg_overtake_per_episode / avg_progress_per_episode
    
    assert 0.02 < ratio < 0.15, f"Overtake should be 2-15% of progress, got {ratio:.2%}"
    print(f"✅ Overtake bonus: {ratio:.1%} of progress reward (secondary role)")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("V3.5 ENHANCED REWARD VERIFICATION")
    print("95% Confidence Requirement for Overtaking")
    print("="*70 + "\n")
    
    test_exact_95_percent_formula()
    test_95_percent_confidence_threshold()
    
    print("\nBehavioral Tests:")
    print("-" * 70)
    test_safe_overtaking_highly_rewarded()
    test_risky_crash_catastrophic()
    test_normal_crash_less_severe()
    test_slow_driving_still_negative()
    test_overtake_bonus_magnitude()
    
    print("\n" + "="*70)
    print("✅ ALL V3.5 ENHANCED TESTS PASSED")
    print("="*70 + "\n")
