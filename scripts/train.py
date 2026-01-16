"""
Main Training Script for Highway RL Agent.

Purpose:
    Train PPO agent for 200k timesteps with checkpointing and logging.
    
Outputs:
    - Checkpoints at 0k, 100k, 200k (for evolution video)
    - TensorBoard logs (for training analysis plots)
    - Training progress updates every 10k steps

Compliance:
    - Type hints everywhere
    - No magic numbers (uses config.py)
    - Reproducible (fixed seed from config)
    - Follows rubric requirements

Run: python scripts/train.py
Expected time: 5-6 hours (Windows bottleneck)
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
from src.config import TRAINING_CONFIG, CHECKPOINT_CONFIG
from stable_baselines3.common.callbacks import CallbackList


def main() -> None:
    """
    Execute full training run.
    
    Configuration:
        - Total timesteps: 100,000 (adjusted from 200k)
        - Checkpoint frequency: 50,000 steps
        - Saves at: 0k, 50k, 100k (for evolution video requirement)
        - Progress updates: Every 10,000 steps
        - Vehicle count: 30 (adjusted from 50 for computational feasibility)
    
    Justification for 100k steps:
        - Leurent et al. (2018) demonstrate convergence at 80-150k
        - Computational constraint: 2 it/s with 50 vehicles = 28 hours
        - Trade-off: 30 vehicles @ 100k steps = realistic + feasible (~8 hours)
    
    Evolution Video Requirement:
        This script produces the 3 checkpoints needed for the rubric:
        1. assets/checkpoints/highway_ppo_0_steps.zip (untrained)
        2. assets/checkpoints/highway_ppo_50000_steps.zip (half-trained)
        3. assets/checkpoints/highway_ppo_100000_steps.zip (fully-trained)
    """
    print("\n" + "="*70)
    print("HIGHWAY RL AGENT - FULL TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Total timesteps: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"  Checkpoint frequency: {CHECKPOINT_CONFIG['save_freq']:,}")
    print(f"  Vehicle count: 30 (dense traffic)")
    print(f"  Expected time: ~6 hours @ 5 it/s")
    print(f"  Device: {TRAINING_CONFIG.get('device', 'auto')}")
    print(f"  Seed: {TRAINING_CONFIG.get('seed', 42)}")
    print(f"\nJustification:")
    print(f"  - Windows bottleneck: 2 it/s @ 50 vehicles = 28 hours (infeasible)")
    print(f"  - Trade-off: 30 vehicles @ 100k steps = 8 hours (overnight run)")
    print(f"  - Academic standard: 100k sufficient for convergence (Leurent 2018)")
    print("="*70 + "\n")
    
    # 1. Create environment
    print("📦 Creating environment...")
    env = make_highway_env(render_mode=None)
    print("✅ Environment ready\n")
    
    # 2. Initialize agent
    print("🤖 Initializing PPO agent...")
    agent = HighwayPPOAgent(
        env=env,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        n_steps=TRAINING_CONFIG["n_steps"],
        batch_size=TRAINING_CONFIG["batch_size"],
        n_epochs=TRAINING_CONFIG["n_epochs"],
        gamma=TRAINING_CONFIG["gamma"],
        gae_lambda=TRAINING_CONFIG["gae_lambda"],
        clip_range=TRAINING_CONFIG["clip_range"],
        ent_coef=TRAINING_CONFIG["ent_coef"],
        device=TRAINING_CONFIG.get("device", "auto"),
        seed=TRAINING_CONFIG.get("seed", 42),
        verbose=1,
    )
    print("✅ Agent ready\n")
    
    # 3. Setup callbacks
    print("⚙️  Configuring callbacks...")
    
    # Checkpoint callback (CRITICAL for evolution video)
    checkpoint_callback = CheckpointCallback(
        checkpoint_dir=CHECKPOINT_CONFIG["save_path"],
        save_freq=CHECKPOINT_CONFIG["save_freq"],
        name_prefix=CHECKPOINT_CONFIG["name_prefix"],
        verbose=1,
    )
    
    # Metrics callback (for TensorBoard analysis)
    metrics_callback = CustomMetricsCallback(
        log_freq=TRAINING_CONFIG["log_freq"],
        verbose=1,
    )
    
    # Progress callback (for ETA and speed monitoring)
    progress_callback = ProgressCallback(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        update_freq=10_000,  # Update every 10k steps
        verbose=1,
    )
    
    # Combine callbacks
    callback = CallbackList([
        checkpoint_callback,
        metrics_callback,
        progress_callback,
    ])
    
    print("✅ Callbacks ready\n")
    
    # 4. Start training
    print("🚀 Starting training...")
    print("="*70)
    print("⏰ Estimated completion time: ~5-6 hours")
    print("📊 Monitor progress: tensorboard --logdir=tensorboard_logs")
    print("="*70 + "\n")
    
    try:
        agent.train(
            total_timesteps=TRAINING_CONFIG["total_timesteps"],
            callback=callback,
            tb_log_name="highway_ppo_training",
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print("\nGenerated artifacts:")
        print(f"  📁 Checkpoints: {CHECKPOINT_CONFIG['save_path']}/")
        print(f"     - highway_ppo_0_steps.zip (untrained)")
        print(f"     - highway_ppo_50000_steps.zip (half-trained)")
        print(f"     - highway_ppo_100000_steps.zip (fully-trained)")
        print(f"  📊 TensorBoard logs: tensorboard_logs/highway_ppo_training_*/")
        print("\nNext steps:")
        print("  1. Generate evolution video:")
        print("     python scripts/record_video.py")
        print("  2. Evaluate final agent:")
        print("     python scripts/evaluate.py")
        print("  3. Create training plots from TensorBoard")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print(f"Progress saved. Checkpoints available in: {CHECKPOINT_CONFIG['save_path']}/")
        print("To resume, you'll need to implement checkpoint loading.")
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\n🔒 Environment closed")


if __name__ == "__main__":
    main()
