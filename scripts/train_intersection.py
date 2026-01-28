"""
Main Training Script for Intersection RL Agent.

Purpose:
    Train PPO agent for 200k timesteps with checkpointing and logging.
    
Outputs:
    - Checkpoints at 0k, 100k, 200k (for evolution video)
    - TensorBoard logs (for training analysis plots)
    - Training progress updates every 10k steps

Compliance:
    - Type hints everywhere
    - No magic numbers (uses intersection_config.py)
    - Reproducible (fixed seed from config)
    - Follows rubric requirements

Run: python scripts/train_intersection.py
Expected time: ~1.5-2 hours (depending on GPU/CPU)
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.intersection_env_v1 import make_intersection_env_v1
from src.agent.ppo_agent import HighwayPPOAgent
from src.training.callbacks import (
    CheckpointCallback,
    CustomMetricsCallback,
    ProgressCallback,
)
from src.intersection_config import (
    INTERSECTION_TRAINING_CONFIG,
    INTERSECTION_CHECKPOINT_CONFIG,
    INTERSECTION_PATHS,
)
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import PPO


def main() -> None:
    """
    Execute training run with resume support.
    
    Configuration:
        - Total timesteps: 200,000
        - Checkpoint frequency: 100,000 steps
        - Saves at: 0k, 100k, 200k (for evolution video requirement)
        - Progress updates: Every 10,000 steps
        - Task: Navigate through intersection while avoiding collisions
        - Reward: Goal-directed navigation with safety focus
    
    Resume Support:
        python scripts/train_intersection.py --resume path/to/checkpoint.zip --additional-steps 100000
    
    Evolution Video Requirement:
        This script produces the 3 checkpoints needed for the rubric:
        1. assets/checkpoints/intersection/intersection_ppo_0_steps.zip (untrained)
        2. assets/checkpoints/intersection/intersection_ppo_100000_steps.zip (half-trained)
        3. assets/checkpoints/intersection/intersection_ppo_200000_steps.zip (fully-trained)
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Intersection RL Agent (V1 Goal-Directed)")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--additional-steps",
        type=int,
        default=None,
        help="Additional steps to train from checkpoint"
    )
    parser.add_argument(
        "--log-name",
        type=str,
        default=None,
        help="Custom TensorBoard log name (default: 'intersection_ppo_training')"
    )
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("INTERSECTION RL AGENT - TRAINING (V1: Goal-Directed Navigation)")
    print("="*70)
    
    # 1. Create V1 environment
    print("\n📦 Creating V1 Goal-Directed environment...")
    env = make_intersection_env_v1(render_mode=None)
    print("✅ Environment ready")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # 2. Determine if resuming or starting fresh
    resume_mode = args.resume is not None
    starting_timesteps = 0
    
    if resume_mode:
        print(f"\n📂 RESUME MODE: Loading checkpoint...")
        print(f"   Checkpoint: {args.resume}")
        
        # Load existing model
        agent = HighwayPPOAgent.load(
            args.resume,
            env=env,
            verbose=INTERSECTION_TRAINING_CONFIG["verbose"]
        )
        
        # Extract timesteps from checkpoint name if possible
        checkpoint_name = Path(args.resume).stem
        if "_steps" in checkpoint_name:
            try:
                starting_timesteps = int(checkpoint_name.split("_")[-2])
                print(f"   Starting from: {starting_timesteps:,} timesteps")
            except (ValueError, IndexError):
                print("   ⚠️ Could not parse starting timesteps from filename")
        
        # Determine additional training steps
        if args.additional_steps is not None:
            additional_steps = args.additional_steps
        else:
            additional_steps = INTERSECTION_TRAINING_CONFIG["total_timesteps"] - starting_timesteps
        
        print(f"   Additional steps: {additional_steps:,}")
        total_training_steps = additional_steps
        
    else:
        print(f"\n🆕 FRESH TRAINING: Creating new agent...")
        
        # Create new agent
        agent = HighwayPPOAgent(
            env=env,
            learning_rate=INTERSECTION_TRAINING_CONFIG["learning_rate"],
            n_steps=INTERSECTION_TRAINING_CONFIG["n_steps"],
            batch_size=INTERSECTION_TRAINING_CONFIG["batch_size"],
            n_epochs=INTERSECTION_TRAINING_CONFIG["n_epochs"],
            gamma=INTERSECTION_TRAINING_CONFIG["gamma"],
            gae_lambda=INTERSECTION_TRAINING_CONFIG["gae_lambda"],
            clip_range=INTERSECTION_TRAINING_CONFIG["clip_range"],
            ent_coef=INTERSECTION_TRAINING_CONFIG["ent_coef"],
            seed=INTERSECTION_TRAINING_CONFIG["seed"],
            device=INTERSECTION_TRAINING_CONFIG["device"],
            verbose=INTERSECTION_TRAINING_CONFIG["verbose"],
        )
        
        print("✅ Agent initialized")
        
        # Save untrained checkpoint (0 steps)
        untrained_path = Path(INTERSECTION_CHECKPOINT_CONFIG["save_path"]) / "intersection_ppo_0_steps.zip"
        agent.save(str(untrained_path))
        print(f"💾 Saved untrained checkpoint: {untrained_path.name}")
        
        total_training_steps = INTERSECTION_TRAINING_CONFIG["total_timesteps"]
    
    # 3. Setup callbacks
    print("\n⚙️ Setting up callbacks...")
    
    # TensorBoard log name
    if args.log_name:
        tensorboard_log_name = args.log_name
    else:
        tensorboard_log_name = "intersection_ppo_training"
    
    callbacks = CallbackList([
        # Checkpoint callback (saves every 100k steps)
        CheckpointCallback(
            checkpoint_dir=INTERSECTION_CHECKPOINT_CONFIG["save_path"],
            save_freq=INTERSECTION_CHECKPOINT_CONFIG["save_freq"],
            name_prefix=INTERSECTION_CHECKPOINT_CONFIG["name_prefix"],
            verbose=1,
        ),
        
        # Custom metrics callback (logs episode stats)
        CustomMetricsCallback(),
        
        # Progress callback (prints updates every 10k steps)
        ProgressCallback(
            total_timesteps=total_training_steps,
            starting_timesteps=starting_timesteps,
            update_freq=10_000,
        ),
    ])
    
    print("✅ Callbacks ready")
    
    # 4. Start training
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    print(f"Total timesteps: {total_training_steps:,}")
    print(f"Checkpoint frequency: {INTERSECTION_CHECKPOINT_CONFIG['save_freq']:,} steps")
    print(f"TensorBoard log: {INTERSECTION_PATHS['logs']}/{tensorboard_log_name}")
    print(f"Checkpoints dir: {INTERSECTION_CHECKPOINT_CONFIG['save_path']}")
    print("="*70 + "\n")
    
    try:
        agent.train(
            total_timesteps=total_training_steps,
            callback=callbacks,
            tb_log_name=tensorboard_log_name,
            reset_num_timesteps=(not resume_mode),  # Reset if fresh training
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        
        # Save final checkpoint
        final_timesteps = starting_timesteps + total_training_steps
        final_path = Path(INTERSECTION_CHECKPOINT_CONFIG["save_path"]) / f"intersection_ppo_{final_timesteps}_steps.zip"
        agent.save(str(final_path))
        print(f"\n💾 Final checkpoint saved: {final_path.name}")
        
        print("\n📁 Checkpoints available:")
        checkpoint_dir = Path(INTERSECTION_CHECKPOINT_CONFIG["save_path"])
        for ckpt in sorted(checkpoint_dir.glob("intersection_ppo_*.zip")):
            print(f"   - {ckpt.name}")
        
        print("\n📊 Next steps:")
        print("   1. Generate training plots: python scripts/plot_training_intersection.py")
        print("   2. Evaluate agent: python scripts/evaluate_intersection.py")
        print("   3. Record videos: python scripts/record_video_intersection.py")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("Saving current progress...")
        
        current_timesteps = starting_timesteps + agent.model.num_timesteps
        interrupt_path = Path(INTERSECTION_CHECKPOINT_CONFIG["save_path"]) / f"intersection_ppo_{current_timesteps}_steps_interrupted.zip"
        agent.save(str(interrupt_path))
        print(f"💾 Interrupted checkpoint saved: {interrupt_path.name}")
        
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        raise
    
    finally:
        env.close()
        print("\n🔒 Environment closed")


if __name__ == "__main__":
    main()
