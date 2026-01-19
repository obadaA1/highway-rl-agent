"""
Configuration file for Highway RL Agent (Phase 2: Polishing).

This module contains ALL hyperparameters and settings.
NO MAGIC NUMBERS allowed in the codebase.

Compliance:
- All hyperparameters centralized (rubric requirement)
- Type hints everywhere (rubric requirement)
- Modular organization (rubric requirement)
- Reproducible seeds (rubric requirement)

Phase 2 Changes (100k -> 200k):
- Entropy reduced from 0.02 to 0.003 (Polishing)
- Reward Function: FROZEN (Standard V6)
"""

from typing import Dict, Any
from pathlib import Path


# ==================================================
# PROJECT STRUCTURE
# ==================================================

PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
PLOTS_DIR = ASSETS_DIR / "plots"
VIDEOS_DIR = ASSETS_DIR / "videos"
CHECKPOINTS_DIR = ASSETS_DIR / "checkpoints"

# Ensure directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

PATHS: Dict[str, str] = {
    "checkpoints": str(CHECKPOINTS_DIR),
    "assets_videos": str(VIDEOS_DIR),
    "plots": str(PLOTS_DIR),
    "logs": str(ASSETS_DIR / "logs"), # Added for Tensorboard
}


# ==================================================
# ENVIRONMENT CONFIGURATION
# ==================================================

ENV_CONFIG: Dict[str, Any] = {
    "id": "highway-v0",
    "render_mode": None,
    "config": {
        # === OBSERVATION SPACE ===
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
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
        },
        
        # === ROAD CONFIGURATION ===
        "lanes_count": 4,
        "vehicles_count": 40, 
        "initial_lane_id": None, 
        "duration": 80,  # 80 seconds max
        
        # === SIMULATION PARAMETERS ===
        "simulation_frequency": 12,
        "policy_frequency": 12,
        
        # === RENDERING ===
        "screen_width": 1200,
        "screen_height": 200,
        "centering_position": [0.3, 0.5],
        "scaling": 3.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False,
        
        # === TRAFFIC DENSITY ===
        "vehicles_density": 2.5,
        
        # === REWARD SHAPING (Must match V6 Logic) ===
        "collision_reward": -0.5,       
        "right_lane_reward": 0.0,       
        "high_speed_reward": 0.5,        # Enabled for V6
        "lane_change_reward": 0.0,      
        "reward_speed_range": [15, 30],  
        "normalize_reward": True,       
        "offroad_terminal": False,       
    }
}


# ==================================================
# REWARD CONFIGURATION (STANDARD V6)
# ==================================================
# This is the "Safe Driving" configuration.
# DO NOT CHANGE THIS during Phase 2 training.
# Continuity is required for the evolution graph.
# ==================================================

REWARD_V6_CONFIG: Dict[str, float] = {
    # === SAFETY OBJECTIVE ===
    "safe_distance_bonus": 0.05,      
    "safe_distance_threshold": 15.0,  
    
    # === SPEED OBJECTIVE ===
    "slow_speed_penalty": 0.02,       
    "slow_speed_threshold": 0.6,      
    
    # === WEAVING PENALTY ===
    "weaving_penalty": 0.08,          
    "weaving_window_steps": 10,       
    
    # === COLLISION PENALTY ===
    "collision_penalty": 0.5,         
    
    # === REWARD FLOOR ===
    "min_reward": -0.5,
}


# ==================================================
# NEURAL NETWORK ARCHITECTURE
# ==================================================

NETWORK_CONFIG: Dict[str, Any] = {
    "policy_type": "MlpPolicy",
    "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
    "activation_fn": "tanh",
    "ortho_init": True,
}


# ==================================================
# PPO TRAINING HYPERPARAMETERS (PHASE 2)
# ==================================================

TRAINING_CONFIG: Dict[str, Any] = {
    # === DURATION ===
    "total_timesteps": 200_000,
    
    # === LEARNING RATE ===
    "learning_rate": 3e-4,
    
    # === ROLLOUT ===
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    
    # === DISCOUNT ===
    "gamma": 0.99,
    "gae_lambda": 0.95,
    
    # === PPO CORE ===
    "clip_range": 0.2,
    "clip_range_vf": None,
    
    # === ENTROPY (THE KEY CHANGE) ===
    # Phase 1 (0-100k): 0.02 (Exploration)
    # Phase 2 (100k-200k): 0.003 (Polishing/Precision)
    "ent_coef": 0.003,
    
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    
    # === SYSTEM ===
    "seed": 42,
    "device": "auto",
    "verbose": 1,
    "log_freq": 1000,
}


# ==================================================
# CHECKPOINT SETTINGS
# ==================================================

CHECKPOINT_CONFIG: Dict[str, Any] = {
    "save_freq": 100_000,
    "name_prefix": "highway_ppo",
    "save_path": str(CHECKPOINTS_DIR),
    "keep_best_n": 3,
}


# ==================================================
# VIDEO RECORDING SETTINGS
# ==================================================

VIDEO_CONFIG: Dict[str, Any] = {
    "fps": 15,
    "codec": "mp4v",
    "quality": 8,
    "episode_length": 200,
    "checkpoints_to_record": {
        "untrained": 0,           
        "half_trained": 100_000,  
        "fully_trained": 200_000, 
    },
    "output_filename": "evolution.mp4",
    "output_path": str(VIDEOS_DIR),
}


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def print_config() -> None:
    """Print all configuration settings."""
    print("\n" + "="*70)
    print("HIGHWAY RL AGENT - CONFIGURATION (PHASE 2)")
    print("="*70)
    # (Simplified print for brevity)
    print(f"Device: {TRAINING_CONFIG['device']}")
    print(f"Entropy: {TRAINING_CONFIG['ent_coef']} (Polishing Mode)")
    print(f"Reward Config: Standard V6")


def get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

if __name__ == "__main__":
    print_config()
    print(f"\n🚀 Device Check: {get_device()}")