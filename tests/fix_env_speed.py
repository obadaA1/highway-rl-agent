"""
Fix highway-env speed by disabling Pygame graphics initialization.

This sets SDL to use a dummy video driver, preventing slow graphics init.

Run: python tests/fix_env_speed.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# CRITICAL FIX: Tell SDL to use dummy video driver (no graphics)
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import time
import gymnasium as gym


def test_with_fix():
    """Test environment speed with SDL dummy driver."""
    print("\n" + "="*70)
    print("TESTING HIGHWAY-ENV WITH SDL DUMMY DRIVER")
    print("="*70)
    print("\nThis should complete in 1-2 seconds (not 47 seconds!)")
    print("-" * 70 + "\n")
    
    env = gym.make("highway-v0", render_mode=None)
    obs, _ = env.reset(seed=42)
    
    start = time.time()
    num_steps = 100
    
    for i in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, _ = env.reset()
        
        # Progress every 25 steps
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            print(f"  Step {i+1}/100 completed in {elapsed:.2f}s")
    
    elapsed = time.time() - start
    fps = num_steps / elapsed
    
    print(f"\n" + "-" * 70)
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Environment FPS: {fps:.1f} steps/second")
    print("-" * 70 + "\n")
    
    if fps > 100:
        print("✅ SUCCESS! Environment is now fast!")
        print(f"   Speed increased from 2.1 FPS to {fps:.1f} FPS")
        print(f"   That's {fps/2.1:.0f}x faster!")
        return True
    elif fps > 20:
        print("⚙️  IMPROVED but still slower than expected")
        print(f"   Expected: 400-700 FPS, Got: {fps:.1f} FPS")
        return False
    else:
        print("❌ FAILED: Still extremely slow")
        print("   There may be another underlying issue")
        return False
    
    env.close()


if __name__ == "__main__":
    success = test_with_fix()
    
    if success:
        print("\n" + "="*70)
        print("HOW TO APPLY THIS FIX PERMANENTLY")
        print("="*70)
        print("\nAdd this line at the TOP of your Python scripts:")
        print('  import os')
        print('  os.environ["SDL_VIDEODRIVER"] = "dummy"')
        print("\nOr set it in Windows environment variables:")
        print("  1. Search 'Environment Variables' in Windows")
        print("  2. Add new User variable:")
        print("     Name: SDL_VIDEODRIVER")
        print("     Value: dummy")
        print("  3. Restart terminal")
        print("="*70 + "\n")
