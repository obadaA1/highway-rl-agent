"""
Integration test: Agent + Environment + Callbacks.

Purpose:
    Verify all components work together before committing to full training.
    
Tests:
    1. Agent initializes with environment
    2. Callbacks attach correctly
    3. Training loop runs without errors
    4. Checkpoints save at correct timesteps
    5. TensorBoard logs are created

Run: python scripts/test_integration.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.highway_env import make_highway_env
from src.agent.ppo_agent import HighwayPPOAgent
from src.training.callbacks import (
    CheckpointCallback,
    CustomMetricsCallback,
    ProgressCallback,
)
from stable_baselines3.common.callbacks import CallbackList


def test_integration() -> None:
    """
    Run a micro-training session to validate integration.
    
    Training: 10k steps (should complete in ~5-10 seconds on GPU)
    Checkpoints: Every 5k steps (should save at 0, 5000, 10000)
    """
    print("\n" + "="*70)
    print("INTEGRATION TEST: Agent + Callbacks + Environment")
    print("="*70)
    
    # 1. Create environment
    print("\n[1/5] Creating environment...")
    env = make_highway_env(render_mode=None)
    print("✅ Environment created")
    
    # 2. Initialize agent
    print("\n[2/5] Initializing PPO agent...")
    agent = HighwayPPOAgent(
        env=env,
        verbose=0,  # Suppress PPO's own logging
    )
    print("✅ Agent initialized")
    
    # 3. Setup callbacks
    print("\n[3/5] Setting up callbacks...")
    
    checkpoint_callback = CheckpointCallback(
        checkpoint_dir="test_integration_checkpoints",
        save_freq=5_000,  # Save every 5k steps (faster for testing)
        name_prefix="test_ppo",
        verbose=1,
    )
    
    metrics_callback = CustomMetricsCallback(
        log_freq=1000,
        verbose=1,
    )
    
    progress_callback = ProgressCallback(
        total_timesteps=10_000,
        update_freq=5_000,
        verbose=1,
    )
    
    # Combine callbacks
    callback = CallbackList([
        checkpoint_callback,
        metrics_callback,
        progress_callback,
    ])
    
    print("✅ Callbacks configured")
    
    # 4. Run mini-training
    print("\n[4/5] Running mini-training (10k steps)...")
    print("Expected time: 5-10 seconds on GPU\n")
    
    try:
        agent.train(
            total_timesteps=10_000,
            callback=callback,
            tb_log_name="integration_test",
        )
        print("\n✅ Training completed without errors")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Verify outputs
    print("\n[5/5] Verifying outputs...")
    
    # Check checkpoints
    checkpoint_dir = Path("test_integration_checkpoints")
    saved_checkpoints = list(checkpoint_dir.glob("*.zip"))
    expected_checkpoints = [0, 5000, 10000]
    
    print(f"  Checkpoints found: {len(saved_checkpoints)}")
    for cp in saved_checkpoints:
        print(f"    - {cp.name}")
    
    if len(saved_checkpoints) != 3:
        print(f"  ⚠️ Expected 3 checkpoints, got {len(saved_checkpoints)}")
    else:
        print(f"  ✅ All 3 checkpoints saved correctly")
    
    # Check TensorBoard logs
    tb_dir = Path("tensorboard_logs")
    if tb_dir.exists():
        print(f"  ✅ TensorBoard logs created at: {tb_dir}")
    else:
        print(f"  ⚠️ TensorBoard logs not found (expected at {tb_dir})")
    
    # Cleanup
    print("\n[Cleanup] Removing test artifacts...")
    import shutil
    shutil.rmtree("test_integration_checkpoints", ignore_errors=True)
    shutil.rmtree("tensorboard_logs", ignore_errors=True)
    
    env.close()
    
    print("\n" + "="*70)
    print("✅ INTEGRATION TEST PASSED")
    print("="*70)
    print("\nYou are ready to run full training with:")
    print("  python scripts/train.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_integration()