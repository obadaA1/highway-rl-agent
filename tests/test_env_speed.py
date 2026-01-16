"""
Minimal environment speed test.
Isolates whether the problem is in our wrapper or base highway-env.

Run: python tests/test_env_speed.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym


def test_base_highway_env():
    """Test raw highway-env without our wrapper."""
    print("\n[TEST 1] Raw highway-env (no wrapper)")
    print("-" * 70)
    
    env = gym.make("highway-v0", render_mode=None)
    obs, _ = env.reset(seed=42)
    
    start = time.time()
    num_steps = 100
    
    for i in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, _ = env.reset()
        
        # Progress every 10 steps
        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/100... ", end="", flush=True)
            elapsed = time.time() - start
            print(f"({elapsed:.1f}s elapsed)")
    
    elapsed = time.time() - start
    fps = num_steps / elapsed
    
    print(f"\nRaw highway-env FPS: {fps:.1f} steps/second")
    
    if fps < 10:
        print("❌ CRITICAL: Base highway-env is extremely slow!")
        print("   This indicates a system-level issue, not our code.")
        return False
    elif fps < 100:
        print("⚠️  WARNING: Base highway-env is slow")
        return False
    else:
        print("✅ Base highway-env speed is acceptable")
        return True
    
    env.close()


def test_custom_wrapper():
    """Test our CustomHighwayEnv wrapper."""
    from src.env.highway_env import make_highway_env
    
    print("\n[TEST 2] CustomHighwayEnv (our wrapper)")
    print("-" * 70)
    
    env = make_highway_env(render_mode=None)
    obs, _ = env.reset(seed=42)
    
    start = time.time()
    num_steps = 100
    
    for i in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, _ = env.reset()
        
        # Progress every 10 steps
        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/100... ", end="", flush=True)
            elapsed = time.time() - start
            print(f"({elapsed:.1f}s elapsed)")
    
    elapsed = time.time() - start
    fps = num_steps / elapsed
    
    print(f"\nCustom wrapper FPS: {fps:.1f} steps/second")
    
    if fps < 10:
        print("❌ CRITICAL: Custom wrapper is extremely slow!")
        print("   Problem is in our wrapper code.")
        return False
    elif fps < 100:
        print("⚠️  WARNING: Custom wrapper adds significant overhead")
        return False
    else:
        print("✅ Custom wrapper speed is acceptable")
        return True
    
    env.close()


def main():
    print("\n" + "="*70)
    print("ENVIRONMENT SPEED ISOLATION TEST")
    print("="*70)
    print("\nThis will take ~30-60 seconds if working correctly")
    print("If each step takes >1 second, there's a critical issue")
    print("="*70)
    
    # Test 1: Base environment
    base_ok = test_base_highway_env()
    
    # Test 2: Our wrapper
    if base_ok:
        wrapper_ok = test_custom_wrapper()
    else:
        print("\n⚠️  Skipping wrapper test (base env is too slow)")
        wrapper_ok = False
    
    # Final verdict
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    if not base_ok:
        print("\n❌ Base highway-env is the bottleneck")
        print("\nPossible causes:")
        print("  1. Pygame is initializing graphics (even with render_mode=None)")
        print("  2. highway-env version issue")
        print("  3. NumPy/SciPy installation problem")
        print("  4. Antivirus blocking file access")
        print("\nTry:")
        print("  pip uninstall highway-env")
        print("  pip install highway-env==1.9.1")
    elif not wrapper_ok:
        print("\n❌ Custom wrapper is adding massive overhead")
        print("\nProblem is in: src/env/highway_env.py")
        print("Likely causes:")
        print("  - Expensive computation in reward function")
        print("  - Inefficient observation processing")
    else:
        print("\n✅ Both base env and wrapper are fast")
        print("   The slowness must be elsewhere")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
