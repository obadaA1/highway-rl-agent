"""
Configuration Verification Script (V2: Speed Control)

Validates that all settings are correctly configured for 40-vehicle training
with 5-component reward function.

Run this before starting training to catch any misconfigurations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ENV_CONFIG, REWARD_CONFIG, TRAINING_CONFIG
from src.env.highway_env import make_highway_env


def verify_configuration():
    """Verify all configuration parameters for V2."""
    print("\n" + "="*70)
    print("CONFIGURATION VERIFICATION (V2: Speed Control)")
    print("="*70 + "\n")
    
    errors = []
    warnings = []
    
    # 1. Check vehicle count
    vehicles = ENV_CONFIG["config"]["vehicles_count"]
    print(f"1. Vehicle Count: {vehicles}")
    if vehicles == 40:
        print("   ✅ Correct: 40 vehicles (V2: reduced for better exploration)")
    elif vehicles == 50:
        warnings.append(f"   ⚠️  Still set to 50 vehicles (V2 should be 40)")
    else:
        errors.append(f"   ❌ Unexpected value: {vehicles}")
    print()
    
    # 2. Check policy frequency
    policy_freq = ENV_CONFIG["config"]["policy_frequency"]
    print(f"2. Policy Frequency: {policy_freq} Hz")
    if policy_freq == 12:
        print("   ✅ Correct: 12 Hz (optimized for 40 vehicles)")
        print(f"   → Reaction time: {1000/policy_freq:.1f}ms (ADAS-level)")
    elif policy_freq == 15:
        warnings.append(f"   ⚠️  Still set to 15 Hz (should be 12 Hz)")
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
    
    # 6. Check reward configuration (V2: 5 components)
    print(f"6. Reward Configuration (V2: 5-component):")
    required_keys = ["w_progress", "r_alive", "r_collision", "r_lane_change",
                     "r_slow_action", "r_low_speed", "min_speed_ratio", "max_velocity"]
    
    missing = [key for key in required_keys if key not in REWARD_CONFIG]
    if missing:
        errors.append(f"   ❌ Missing keys: {missing}")
    
    print(f"   w_progress: {REWARD_CONFIG.get('w_progress', 'MISSING')}")
    print(f"   r_alive: {REWARD_CONFIG.get('r_alive', 'MISSING')}")
    print(f"   r_collision: {REWARD_CONFIG.get('r_collision', 'MISSING')}")
    print(f"   r_lane_change: {REWARD_CONFIG.get('r_lane_change', 'MISSING')}")
    print(f"   r_slow_action: {REWARD_CONFIG.get('r_slow_action', 'MISSING')} ⚠️ NEW")
    print(f"   r_low_speed: {REWARD_CONFIG.get('r_low_speed', 'MISSING')} ⚠️ NEW")
    print(f"   min_speed_ratio: {REWARD_CONFIG.get('min_speed_ratio', 'MISSING')} ⚠️ NEW")
    
    # Validate values
    if (REWARD_CONFIG.get('w_progress') == 1.0 and
        REWARD_CONFIG.get('r_collision') == -80.0 and
        REWARD_CONFIG.get('r_slow_action') == -0.02 and
        REWARD_CONFIG.get('r_low_speed') == -0.01 and
        REWARD_CONFIG.get('min_speed_ratio') == 0.6):
        print("   ✅ Correct: V2 speed control configuration")
    else:
        warnings.append("   ⚠️  Reward values don't match V2 configuration")
    print()
    
    # 7. Check training settings
    total_steps = TRAINING_CONFIG['total_timesteps']
    print(f"7. Training Configuration:")
    print(f"   Total timesteps: {total_steps:,}")
    expected_time_sec = total_steps / 42  # 40-45 it/s expected for 40 vehicles
    expected_time_min = expected_time_sec / 60
    print(f"   Expected time @ 42 it/s: {expected_time_min:.1f} minutes")
    if total_steps == 200_000:
        print("   ✅ Correct: 200k steps (thorough training)")
    elif total_steps == 100_000:
        warnings.append("   ⚠️  Set to 100k (V2 uses 200k)")
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
    print("SUMMARY (V2: Speed Control)")
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
        print("\nConfiguration Summary (V2):")
        print(f"  • 40 vehicles (dense traffic, better exploration)")
        print(f"  • 12 Hz policy (83ms reactions)")
        print(f"  • 960 steps per episode (80s)")
        print(f"  • 200k total timesteps")
        print(f"  • 5-component reward (with speed control)")
        print(f"  • Expected: ~40-45 it/s, ~90 min training")
        print("\n🚀 Ready to start V2 training!")
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
