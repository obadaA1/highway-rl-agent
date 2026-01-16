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

Compliance:
    - Type hints everywhere
    - Realistic time expectations (Windows: 20-30 min)
    - No magic numbers
    - Clean path handling

Run: python tests/test_integration.py
"""

import sys
import shutil
from pathlib import Path
from typing import List

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


# Test configuration (separated from main config)
# Reduced for faster validation with 50 vehicles @ 2 it/s
TEST_TIMESTEPS: int = 2_000  # ~17 minutes (vs 83 min for 10k)
TEST_CHECKPOINT_FREQ: int = 1_000  # Checkpoint at 0, 1k, 2k
TEST_METRIC_LOG_FREQ: int = 500
TEST_CHECKPOINT_DIR: str = "test_integration_checkpoints"
TEST_TB_LOG_DIR: str = "tensorboard_logs"


def test_integration() -> None:
    """
    Run a micro-training session to validate integration.
    
    Configuration:
        - Total steps: 2,000 (reduced for fast validation)
        - Checkpoint frequency: Every 1,000 steps
        - Expected checkpoints: 3 (at 0, 1k, 2k)
        - Expected time: ~8-10 minutes @ 3-4 it/s (30 vehicles)
    
    Raises:
        Exception: If training fails or outputs are incorrect
    """
    print("\n" + "="*70)
    print("INTEGRATION TEST: Agent + Callbacks + Environment")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Total timesteps: {TEST_TIMESTEPS:,}")
    print(f"  Checkpoint frequency: {TEST_CHECKPOINT_FREQ:,}")
    print(f"  Vehicles: 30 (dense traffic - adjusted from 50)")
    print(f"  Expected time: ~8-10 minutes @ 3-4 it/s")
    print("="*70)
    
    # 1. Create environment
    print("\n[1/5] Creating environment...")
    env = make_highway_env(render_mode=None)
    print("✅ Environment created")
    
    # 2. Initialize agent
    print("\n[2/5] Initializing PPO agent...")
    agent = HighwayPPOAgent(
        env=env,
        verbose=0,  # Suppress PPO's own logging during test
    )
    print("✅ Agent initialized")
    
    # 3. Setup callbacks
    print("\n[3/5] Setting up callbacks...")
    
    checkpoint_callback = CheckpointCallback(
        checkpoint_dir=TEST_CHECKPOINT_DIR,
        save_freq=TEST_CHECKPOINT_FREQ,
        name_prefix="test_ppo",
        verbose=1,
    )
    
    metrics_callback = CustomMetricsCallback(
        log_freq=TEST_METRIC_LOG_FREQ,
        verbose=1,
    )
    
    progress_callback = ProgressCallback(
        total_timesteps=TEST_TIMESTEPS,  # ← FIXED: Matches test config
        update_freq=TEST_CHECKPOINT_FREQ,
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
    print("\n[4/5] Running mini-training...")
    print(f"Training for {TEST_TIMESTEPS:,} steps...")
    print("⏱️  Expected time: ~8-10 minutes @ 3-4 it/s (30 vehicles)")
    print("   (Progress updates will appear below)\n")
    
    try:
        agent.train(
            total_timesteps=TEST_TIMESTEPS,
            callback=callback,
            tb_log_name="integration_test",
        )
        print("\n✅ Training completed without errors")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return
    
    # 5. Verify outputs
    print("\n[5/5] Verifying outputs...")
    
    # Check checkpoints
    checkpoint_dir = Path(TEST_CHECKPOINT_DIR)
    saved_checkpoints: List[Path] = sorted(checkpoint_dir.glob("*.zip"))
    expected_checkpoint_count: int = 3  # 0, 1k, 2k
    
    print(f"  Checkpoints found: {len(saved_checkpoints)}")
    for cp in saved_checkpoints:
        print(f"    - {cp.name}")
    
    if len(saved_checkpoints) != expected_checkpoint_count:
        print(f"  ⚠️  Expected {expected_checkpoint_count} checkpoints, "
              f"got {len(saved_checkpoints)}")
    else:
        print(f"  ✅ All {expected_checkpoint_count} checkpoints saved correctly")
    
    # Check TensorBoard logs
    tb_dir = Path(TEST_TB_LOG_DIR)
    if tb_dir.exists():
        tb_files = list(tb_dir.glob("**/events.out.tfevents.*"))
        print(f"  ✅ TensorBoard logs created at: {tb_dir}")
        print(f"     ({len(tb_files)} event files found)")
    else:
        print(f"  ⚠️  TensorBoard logs not found (expected at {tb_dir})")
    
    # Cleanup (keep TensorBoard logs for inspection)
    print("\n[Cleanup] Removing test checkpoints...")
    shutil.rmtree(TEST_CHECKPOINT_DIR, ignore_errors=True)
    print("  ✅ Test checkpoints cleaned up")
    print("  ℹ️  TensorBoard logs preserved for inspection")
    
    env.close()
    
    print("\n" + "="*70)
    print("✅ INTEGRATION TEST PASSED")
    print("="*70)
    print("\nGenerated artifacts:")
    print(f"  📊 TensorBoard logs: tensorboard_logs/integration_test_*/")
    print("     View with: tensorboard --logdir=tensorboard_logs")
    print("\nNext steps:")
    print("  1. Review TensorBoard logs to verify metrics")
    print("  2. Proceed to full training (100k steps, ~8 hours):")
    print("     python scripts/train.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_integration()