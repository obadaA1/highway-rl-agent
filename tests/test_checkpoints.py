"""Test if 200k checkpoint loads and predicts correctly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from src.env.highway_env import make_highway_env
from src.config import PATHS, ENV_CONFIG
import numpy as np

def test_checkpoint(checkpoint_name: str) -> None:
    """Test a checkpoint."""
    checkpoint_path = Path(PATHS["checkpoints"]) / checkpoint_name
    
    print(f"\n{'='*70}")
    print(f"Testing: {checkpoint_name}")
    print(f"{'='*70}")
    
    if not checkpoint_path.exists():
        print(f"❌ File not found: {checkpoint_path}")
        return
    
    print(f"✅ File exists: {checkpoint_path.stat().st_size / 1024:.1f} KB")
    
    # Load model
    try:
        model = PPO.load(checkpoint_path)
        print(f"✅ Model loaded successfully")
        print(f"   Device: {model.device}")
        print(f"   Policy: {model.policy.__class__.__name__}")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return
    
    # Create environment
    env = make_highway_env(render_mode=None)
    obs, info = env.reset()
    print(f"✅ Environment created")
    print(f"   Observation shape: {obs.shape}")
    
    # Test prediction
    try:
        action, _ = model.predict(obs, deterministic=True)
        print(f"✅ Prediction works")
        print(f"   Action: {action}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return
    
    # Run short episode
    print(f"\nRunning 10-step test episode...")
    obs, _ = env.reset()
    
    for step in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {step+1}: action={action}, reward={reward:.3f}, done={done}")
        
        if done or truncated:
            print(f"  Episode ended at step {step+1}")
            break
    
    env.close()
    
    if done and step < 5:
        print(f"\n⚠️  WARNING: Agent crashed after only {step+1} steps!")
        print(f"  This checkpoint may not have learned collision avoidance.")
    else:
        print(f"\n✅ Agent survived {step+1} steps without crashing.")

def main():
    """Test all checkpoints."""
    print("\n" + "="*70)
    print("CHECKPOINT VERIFICATION TEST")
    print("="*70)
    
    checkpoints = [
        "highway_ppo_100000_steps.zip",
        "highway_ppo_200000_steps.zip",
    ]
    
    for checkpoint in checkpoints:
        test_checkpoint(checkpoint)
    
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    print("\nIf 200k performs worse than 100k, possible causes:")
    print("  1. Training diverged (value function collapsed)")
    print("  2. Checkpoint saved during eval (random actions)")
    print("  3. Learning rate too high (unstable updates)")
    print("  4. Policy catastrophically forgot (rare)")
    print("\nCheck TensorBoard for:")
    print("  - Reward curve (should be increasing 100k→200k)")
    print("  - Value loss (should be stable)")
    print("  - KL divergence (should be low)")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
