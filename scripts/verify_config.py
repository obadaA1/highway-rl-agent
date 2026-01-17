"""
Configuration Verification Script

Validates that all settings are correctly configured for 50-vehicle training.
Run this before starting training to catch any misconfigurations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ENV_CONFIG, REWARD_CONFIG, TRAINING_CONFIG
from src.env.highway_env import make_highway_env


def verify_configuration():
    """Verify all configuration parameters."""
    print("\n" + "="*70)
    print("CONFIGURATION VERIFICATION")
    print("="*70 + "\n")
    
    errors = []
    warnings = []
    
    # 1. Check vehicle count
    vehicles = ENV_CONFIG["config"]["vehicles_count"]
    print(f"1. Vehicle Count: {vehicles}")
    if vehicles == 50:
        print("   ✅ Correct: 50 vehicles (upper bound of benchmarks)")
    elif vehicles == 30:
        warnings.append(f"   ⚠️  Still set to 30 vehicles (should be 50)")
    else:
        errors.append(f"   ❌ Unexpected value: {vehicles}")
    print()
    
    # 2. Check policy frequency
    policy_freq = ENV_CONFIG["config"]["policy_frequency"]
    print(f"2. Policy Frequency: {policy_freq} Hz")
    if policy_freq == 12:
        print("   ✅ Correct: 12 Hz (optimized for 50 vehicles)")
        print(f"   → Reaction time: {1000/policy_freq:.1f}ms (ADAS-level)")
    elif policy_freq == 15:
        warnings.append(f"   ⚠️  Still set to 15 Hz (should be 12 Hz for 50 vehicles)")
    elif policy_freq == 1:
        errors.append(f"   ❌ Critical bug: 1 Hz policy (should be 12 Hz)")
    else:
        warnings.append(f"   ⚠️  Unusual value: {policy_freq} Hz")
    print()
    
    # 3. Check simulation frequency (must match policy)
    sim_freq = ENV_CONFIG["config"]["simulation_frequency"]
    print(f"3. Simulation Frequency: {sim_freq} Hz")
    if sim_freq == policy_freq:
        print(f"   ✅ Synchronized with policy ({sim_freq} Hz)")
    else:
        errors.append(f"   ❌ DESYNCHRONIZED: sim={sim_freq} Hz, policy={policy_freq} Hz")
        errors.append(f"      This causes {sim_freq/policy_freq:.1f}× simulation overhead!")
    print()
    
    # 4. Check episode duration
    duration = ENV_CONFIG["config"]["duration"]
    steps_per_ep = duration * policy_freq
    print(f"4. Episode Duration: {duration}s")
    print(f"   → Steps per episode: {steps_per_ep} ({duration}s × {policy_freq} Hz)")
    if duration == 80 and steps_per_ep == 960:
        print("   ✅ Correct: 80s episodes = 960 steps at 12 Hz")
    elif duration == 80:
        print(f"   ⚠️  Duration correct but steps = {steps_per_ep}")
    else:
        warnings.append(f"   ⚠️  Unusual duration: {duration}s")
    print()
    
    # 5. Check traffic density
    density = ENV_CONFIG["config"].get("vehicles_density", 2.0)
    print(f"5. Vehicles Density: {density}")
    if density == 2.5:
        print("   ✅ Correct: 2.5 (very dense spawning)")
    elif density == 2.0:
        warnings.append("   ⚠️  Still set to 2.0 (should be 2.5 for very dense)")
    print()
    
    # 6. Check reward weights
    print(f"6. Reward Configuration:")
    print(f"   w_velocity: {REWARD_CONFIG['w_velocity']}")
    print(f"   w_collision: {REWARD_CONFIG['w_collision']}")
    print(f"   w_lane_change: {REWARD_CONFIG['w_lane_change']}")
    print(f"   w_distance: {REWARD_CONFIG['w_distance']}")
    
    if (REWARD_CONFIG['w_velocity'] == 0.8 and 
        REWARD_CONFIG['w_lane_change'] == 0.02 and
        REWARD_CONFIG['w_distance'] == 0.1):
        print("   ✅ Correct: Aggressive driving rewards")
    else:
        warnings.append("   ⚠️  Reward weights don't match aggressive configuration")
    print()
    
    # 7. Check training settings
    total_steps = TRAINING_CONFIG['total_timesteps']
    print(f"7. Training Configuration:")
    print(f"   Total timesteps: {total_steps:,}")
    expected_time_sec = total_steps / 35  # 35 it/s expected
    expected_time_min = expected_time_sec / 60
    print(f"   Expected time @ 35 it/s: {expected_time_min:.1f} minutes")
    if total_steps == 200_000:
        print("   ✅ Correct: 200k steps (thorough training)")
    elif total_steps == 100_000:
        warnings.append("   ⚠️  Set to 100k (consider 200k for harder 50-vehicle task)")
    print()
    
    # 8. Test environment creation
    print("8. Environment Creation Test:")
    try:
        env = make_highway_env(render_mode=None)
        print("   ✅ Environment created successfully")
        
        # Check actual vehicle count in environment
        obs, _ = env.reset()
        print(f"   → Observation shape: {obs.shape}")
        print(f"   → Can observe: {obs.shape[0]} vehicles")
        
        env.close()
    except Exception as e:
        errors.append(f"   ❌ Environment creation failed: {e}")
    print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    if errors:
        print("\n❌ CRITICAL ERRORS:")
        for error in errors:
            print(error)
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(warning)
    
    if not errors and not warnings:
        print("\n✅ ALL CHECKS PASSED")
        print("\nConfiguration Summary:")
        print(f"  • 50 vehicles (very dense traffic)")
        print(f"  • 12 Hz policy (83ms reactions)")
        print(f"  • 960 steps per episode (80s)")
        print(f"  • 200k total timesteps")
        print(f"  • Expected: ~35 it/s, ~95 min training")
        print("\n🚀 Ready to start training!")
    elif not errors:
        print("\n⚠️  WARNINGS DETECTED (non-critical)")
        print("Training can proceed but configuration may not be optimal.")
    else:
        print("\n❌ CRITICAL ERRORS DETECTED")
        print("Fix configuration before training!")
    
    print("="*70 + "\n")
    
    return len(errors) == 0


if __name__ == "__main__":
    success = verify_configuration()
    sys.exit(0 if success else 1)
