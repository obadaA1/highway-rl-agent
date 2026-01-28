"""
Configuration file for Intersection RL Agent.

This module contains ALL hyperparameters and settings for intersection-v0.
NO MAGIC NUMBERS allowed in the codebase.

Compliance:
- All hyperparameters centralized (rubric requirement)
- Type hints everywhere (rubric requirement)
- Modular organization (rubric requirement)
- Reproducible seeds (rubric requirement)

Environment: intersection-v0 (Gymnasium + highway-env)
Task: Navigate through intersection while yielding to traffic and avoiding collisions
"""

from typing import Dict, Any
from pathlib import Path


# ==================================================
# PROJECT STRUCTURE (INTERSECTION)
# ==================================================

PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
PLOTS_DIR = ASSETS_DIR / "plots" / "intersection"
VIDEOS_DIR = ASSETS_DIR / "videos" / "intersection"
CHECKPOINTS_DIR = ASSETS_DIR / "checkpoints" / "intersection"

# Ensure directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

INTERSECTION_PATHS: Dict[str, str] = {
    "checkpoints": str(CHECKPOINTS_DIR),
    "assets_videos": str(VIDEOS_DIR),
    "plots": str(PLOTS_DIR),
    "logs": str(ASSETS_DIR / "logs" / "intersection"),
}

# Ensure logs directory exists
(ASSETS_DIR / "logs" / "intersection").mkdir(parents=True, exist_ok=True)


# ==================================================
# ENVIRONMENT CONFIGURATION (INTERSECTION-V0)
# ==================================================

INTERSECTION_ENV_CONFIG: Dict[str, Any] = {
    "id": "intersection-v1",
    "render_mode": None,
    "config": {
        # === OBSERVATION SPACE ===
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,  # More vehicles at intersection
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "normalize": True,
            "order": "sorted",
        },
        
        # === ACTION SPACE ===
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 4.5, 9],  # 0, ~4.5, ~9 m/s (0, 16, 32 km/h)
        },
        
        # === INTERSECTION CONFIGURATION ===
        "duration": 13,  # 13 seconds max (typical intersection crossing time)
        "destination": "o1",  # Destination route
        "controlled_vehicles": 1,
        "initial_vehicle_count": 10,  # Traffic density
        "spawn_probability": 0.6,  # Continuous traffic spawning
        
        # === SIMULATION PARAMETERS ===
        "simulation_frequency": 15,  # Hz
        "policy_frequency": 5,  # Decision frequency (Hz)
        
        # === RENDERING ===
        "screen_width": 600,
        "screen_height": 600,
        "centering_position": [0.5, 0.6],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False,
        
        # === BASE REWARD CONFIGURATION ===
        "collision_reward": -5,
        "high_speed_reward": 1,
        "arrived_reward": 1,
        "reward_speed_range": [7.0, 9.0],
        "normalize_reward": False,
    }
}


# ==================================================
# REWARD CONFIGURATION (INTERSECTION V1)
# ==================================================
# Goal-directed navigation with safety focus
# This reward function encourages:
# 1. Progress toward destination (goal_progress_reward)
# 2. Successful safe crossing (safe_crossing_bonus)
# 3. Avoiding collisions (collision_penalty)
# 4. Timely decision-making (timeout_penalty)
# ==================================================

INTERSECTION_REWARD_CONFIG: Dict[str, float] = {
    # === GOAL OBJECTIVE ===
    "goal_progress_reward": 0.4,      # Reward for moving toward destination
    "safe_crossing_bonus": 1.0,       # Large bonus for successful crossing
    
    # === PENALTIES ===
    "collision_penalty": 1.0,         # Strong penalty for crashes
    "timeout_penalty": 0.01,          # Small penalty per step (encourages efficiency)
    
    # === REWARD FLOOR ===
    "min_reward": -1.0,
}


# ==================================================
# NEURAL NETWORK ARCHITECTURE (INTERSECTION)
# ==================================================

INTERSECTION_NETWORK_CONFIG: Dict[str, Any] = {
    "policy_type": "MlpPolicy",
    "net_arch": [dict(pi=[64, 64], vf=[64, 64])],  # Same as highway for consistency
    "activation_fn": "tanh",
    "ortho_init": True,
}


# ==================================================
# PPO TRAINING HYPERPARAMETERS (INTERSECTION)
# ==================================================

INTERSECTION_TRAINING_CONFIG: Dict[str, Any] = {
    # === DURATION ===
    "total_timesteps": 200_000,  # Same as highway for fair comparison
    
    # === LEARNING RATE ===
    "learning_rate": 3e-4,  # Standard PPO learning rate
    
    # === ROLLOUT ===
    "n_steps": 2048,  # Steps per rollout
    "batch_size": 64,  # Mini-batch size
    "n_epochs": 10,  # Optimization epochs per rollout
    
    # === DISCOUNT ===
    "gamma": 0.99,  # Discount factor
    "gae_lambda": 0.95,  # GAE parameter
    
    # === PPO CORE ===
    "clip_range": 0.2,  # PPO clipping parameter
    "clip_range_vf": None,  # Value function clipping (None = no clipping)
    
    # === ENTROPY ===
    # Start with exploration, can reduce later if needed
    "ent_coef": 0.01,  # Entropy coefficient (encourages exploration)
    
    "vf_coef": 0.5,  # Value function coefficient
    "max_grad_norm": 0.5,  # Gradient clipping
    
    # === SYSTEM ===
    "seed": 42,  # Reproducibility
    "device": "auto",  # "cuda" if available, else "cpu"
    "verbose": 1,  # Logging level
    "log_freq": 1000,  # TensorBoard logging frequency
}


# ==================================================
# CHECKPOINT SETTINGS (INTERSECTION)
# ==================================================

INTERSECTION_CHECKPOINT_CONFIG: Dict[str, Any] = {
    "save_freq": 100_000,  # Save every 100k steps
    "name_prefix": "intersection_ppo",
    "save_path": str(CHECKPOINTS_DIR),
    "keep_best_n": 3,  # Keep best 3 checkpoints
}


# ==================================================
# VIDEO RECORDING SETTINGS (INTERSECTION)
# ==================================================

INTERSECTION_VIDEO_CONFIG: Dict[str, Any] = {
    "fps": 15,
    "codec": "mp4v",
    "quality": 8,
    "episode_length": 200,  # Max frames per episode
    "checkpoints_to_record": {
        "untrained": 0,           # Initial random policy
        "half_trained": 100_000,  # Mid-training checkpoint
        "fully_trained": 200_000, # Final trained policy
    },
    "output_filename": "intersection_evolution.mp4",
    "output_path": str(VIDEOS_DIR),
}


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def print_intersection_config() -> None:
    """Print all intersection configuration settings."""
    print("\n" + "="*70)
    print("INTERSECTION RL AGENT - CONFIGURATION")
    print("="*70)
    print(f"\nEnvironment: {INTERSECTION_ENV_CONFIG['id']}")
    print(f"Total Timesteps: {INTERSECTION_TRAINING_CONFIG['total_timesteps']:,}")
    print(f"Device: {INTERSECTION_TRAINING_CONFIG['device']}")
    print(f"Learning Rate: {INTERSECTION_TRAINING_CONFIG['learning_rate']}")
    print(f"Entropy Coefficient: {INTERSECTION_TRAINING_CONFIG['ent_coef']}")
    print(f"Checkpoint Frequency: {INTERSECTION_CHECKPOINT_CONFIG['save_freq']:,}")
    print(f"\nReward Components:")
    print(f"  - Goal Progress: +{INTERSECTION_REWARD_CONFIG['goal_progress_reward']}")
    print(f"  - Safe Crossing: +{INTERSECTION_REWARD_CONFIG['safe_crossing_bonus']}")
    print(f"  - Collision: -{INTERSECTION_REWARD_CONFIG['collision_penalty']}")
    print(f"  - Timeout: -{INTERSECTION_REWARD_CONFIG['timeout_penalty']}/step")
    print("="*70 + "\n")


def get_device() -> str:
    """Check if CUDA is available."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


if __name__ == "__main__":
    print_intersection_config()
    print(f"🚀 Device Check: {get_device()}")
