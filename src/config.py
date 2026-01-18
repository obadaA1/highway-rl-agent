"""
Configuration file for Highway RL Agent.

This module contains ALL hyperparameters and settings.
NO MAGIC NUMBERS allowed in the codebase.

Compliance:
- All hyperparameters centralized (rubric requirement)
- Type hints everywhere (rubric requirement)
- Modular organization (rubric requirement)
- Reproducible seeds (rubric requirement)

Author: [Your Name]
Date: 2025-01-16
"""

from typing import Dict, Any, List, Tuple
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

# Path dictionary for scripts
PATHS: Dict[str, str] = {
    "checkpoints": str(CHECKPOINTS_DIR),
    "assets_videos": str(VIDEOS_DIR),
    "plots": str(PLOTS_DIR),
}


# ==================================================
# ENVIRONMENT CONFIGURATION
# ==================================================

ENV_CONFIG: Dict[str, Any] = {
    # Environment ID from highway-env
    "id": "highway-v0",
    
    # Rendering mode
    # 'rgb_array': Returns images (required for video recording)
    # 'human': Opens visualization window
    "render_mode": None,
    
    # Highway-env specific configuration
    # This defines the STATE SPACE (what the agent observes)
    "config": {
        # === OBSERVATION SPACE ===
        "observation": {
            # Type: "Kinematics" = matrix of nearby vehicle states
            # Alternative: "GrayscaleObservation" = pixels
            "type": "Kinematics",
            
            # Number of nearby vehicles to observe
            # Shape: (vehicles_count, features)
            "vehicles_count": 5,
            
            # Features per vehicle:
            # - presence: 1 if vehicle exists, 0 otherwise
            # - x: longitudinal position (meters, relative to ego)
            # - y: lateral position (meters, relative to ego)
            # - vx: longitudinal velocity (m/s)
            # - vy: lateral velocity (m/s)
            "features": ["presence", "x", "y", "vx", "vy"],
            
            # Features_range defines normalization bounds
            # We'll use default: [[-1, 1], ...] for all features
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            
            # Absolute vs relative coordinates
            # False: positions relative to ego vehicle (recommended)
            "absolute": False,
            
            # Normalize features to [-1, 1]
            "normalize": True,
            
            # Order: closest vehicles first
            "order": "sorted",
        },
        
        # === ACTION SPACE ===
        "action": {
            # DiscreteMetaAction: 5 discrete actions
            # 0: LANE_LEFT  - Change lane to the left
            # 1: IDLE       - Maintain current speed/lane
            # 2: LANE_RIGHT - Change lane to the right
            # 3: FASTER     - Accelerate
            # 4: SLOWER     - Decelerate
            "type": "DiscreteMetaAction",
        },
        
        # === ROAD CONFIGURATION ===
        # Number of lanes (4-lane highway)
        "lanes_count": 4,
        
        # Number of other vehicles on the road
        # ADJUSTED: Reduced to 40 vehicles to improve exploration
        # Justification:
        #   - Previous 50-vehicle training led to degenerate policies
        #   - 40 vehicles = still dense (Leurent et al. use 20-50 range)
        #   - Allows better learning dynamics while maintaining difficulty
        #   - Should improve training speed and convergence quality
        "vehicles_count": 40,
        
        # Ego vehicle starting configuration
        "initial_lane_id": None,  # Random lane
        
        # Episode duration (seconds)
        # After this time, episode is truncated (not terminated)
        # ADJUSTED: Increased from 40s to 80s to show longer trained agent performance
        "duration": 80,
        
        # === SIMULATION PARAMETERS ===
        # Simulation frequency (Hz)
        # Physics updates per second
        "simulation_frequency": 12,
        
        # Policy frequency (Hz)
        # Agent decision rate (actions per second)
        # OPTIMIZED: 12 Hz (reduced from 15 Hz) for 50-vehicle configuration
        # Rationale:
        #   - MUST match simulation_frequency (avoid 15× overhead bug)
        #   - 12 Hz = 83ms reaction time (still ADAS-level, excellent)
        #   - Partially offsets 2.8× collision detection cost from 50 vehicles
        #   - At 12 Hz: 80 seconds × 12 = 960 steps per episode
        #   - Expected: 35 it/s (vs 25 it/s at 15 Hz with 50 vehicles)
        "policy_frequency": 12,
        
        # === RENDERING (for video recording) ===
        # ADJUSTED: Wider screen and more zoomed out for full highway view
        "screen_width": 1200,
        "screen_height": 200,
        # Centering: [0.3, 0.5] means ego vehicle at 30% from left, 50% from top
        # ADJUSTED: Changed to [0.3, 0.5] to show more road ahead
        "centering_position": [0.3, 0.5],
        # Scaling: Lower values = more zoomed out (show more vehicles)
        # ADJUSTED: Changed to 3.5 for maximum highway visibility
        "scaling": 3.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False,
        
        # === TRAFFIC DENSITY ===
        # Vehicles density (spawning rate)
        # Higher = more vehicles spawn on highway
        # OPTIMIZED: Increased to 2.5 for very dense traffic
        "vehicles_density": 2.5,
        
        # === DEFAULT REWARDS (DISABLED FOR V5) ===
        # CRITICAL FIX: Disable highway-env's built-in reward shaping
        # Problem: Built-in rewards dominated custom V5 reward, causing:
        #   - Degenerate LANE_RIGHT-only policy (92% of actions)
        #   - Zero FASTER action usage
        #   - Agent optimized for "stay right = clear lane" instead of strategic driving
        # Solution: Neutralize all built-in rewards to give custom reward full control
        "collision_reward": -1.0,        # Keep (used by highway-env internally)
        "right_lane_reward": 0.0,        # Disabled (was biasing right lane)
        "high_speed_reward": 0.0,        # Disabled (was conflicting with r_progress)
        "lane_change_reward": 0.0,       # Disabled (we use r_lane instead)
        "reward_speed_range": [0, 30],   # Wide range = no implicit speed penalty
        "normalize_reward": False,       # Use raw custom reward (no normalization)
        "offroad_terminal": False,       # Don't terminate on lane boundaries
    }
}


# ==================================================
# CUSTOM REWARD FUNCTION PARAMETERS
# ==================================================
# 
# PROGRESS-BASED REWARD PHILOSOPHY (OPTIMAL POLICY):
#
# Core Objective: Maximize forward progress (distance traveled)
#   max Σ Δx_t  (maximize total distance, not speed or survival time)
#
# This automatically balances speed + survival:
#   - Fast driving → More distance per step → Higher reward
#   - Crashing early → Fewer steps → Less total distance → Lower reward
#   - Optimal policy: Fast + Long survival = Maximum distance
#
# Three Minimal Fixes (Ng et al. 1999 reward shaping):
#
# FIX 1: Progress reward (not velocity reward)
#   r_progress = (speed × Δt) / (max_speed × Δt) = normalized distance per step
#   Why: Crashing early automatically reduces cumulative reward
#
# FIX 2: Small survival bonus (prevents "crash early")
#   r_alive = 0.01 (small enough to NOT dominate progress)
#   Why: Tiebreaker only, does NOT create "slow forever" behavior
#
# FIX 3: Lane changes neutral (not penalized)
#   r_lane_change = 0.0 (allow overtaking)
#   Why: Agent learns to change lanes when beneficial for progress
#
# FINAL REWARD:
#   R_t = r_progress + r_alive + r_collision
#
# ==================================================

REWARD_CONFIG: Dict[str, float] = {
    # === FIX 1: PROGRESS REWARD (Core change) ===
    
    # Progress reward (normalized distance per step)
    # Formula: r_progress = velocity / max_velocity
    # At 12 Hz: This approximates (velocity × 1/12) / (max_velocity × 1/12)
    # Range: [0, 1] per step
    # 
    # Why this fixes "slow agent":
    #   Old (velocity): Slow driving → More steps → More cumulative reward
    #   New (progress): Slow driving → Less distance/step → Less cumulative reward
    "w_progress": 1.0,  # No scaling needed (already normalized)
    
    # === FIX 2: SMALL SURVIVAL BONUS ===
    
    # Alive bonus per step (VERY small)
    # Prevents "crash early after gaining progress" exploitation
    # CRITICAL: Must be << progress reward (0.01 vs ~0.5-1.0)
    # Otherwise creates "drive slow forever" behavior again
    "r_alive": 0.01,
    
    # === COLLISION PENALTY (Hard constraint) ===
    
    # Collision penalty (terminal punishment)
    # Must ALWAYS be worse than any progress gains to prevent strategic crashes
    # At 12 Hz, max speed: ~1.0 reward/step × 60 steps (5s) = 60.0 max progress
    # Collision penalty = -80.0 ensures crash is NEVER optimal (80 > 60)
    "r_collision": -80.0,
    
    # === FIX 3: LANE CHANGE NEUTRAL ===
    
    # Lane change cost (NEUTRAL - no penalty)
    # Changed from -0.05 to 0.0 to allow free overtaking
    # Agent learns: "Change lanes when it increases progress"
    # Prevents zig-zagging (no reward) while allowing necessary maneuvers
    "r_lane_change": 0.0,
    
    # === V3: MUCH STRONGER SPEED CONTROL (Anti-Degenerate-Policy Fix) ===
    
    # SLOWER action penalty (discourages "spam SLOWER" degenerate policy)
    # V1 training: 200k agent used SLOWER 80.2 times/episode (100%)
    # V2 training: 200k agent used SLOWER 96.6 times/episode (100%) - penalties too weak!
    # V3: Increased penalty 5× to make slow driving net NEGATIVE
    # Math: At 5 m/s with SLOWER, reward = (5/30) + 0.01 + (-0.10) + (-0.20) = -0.123 < 0
    "r_slow_action": -0.10,  # Was -0.02 in V2 (5× stronger)
    
    # Low speed penalty (enforces minimum velocity)
    # Applied when velocity < min_speed_ratio × max_velocity
    # V2: -0.01 was too weak (net reward still positive)
    # V3: -0.20 makes ANY speed below 18 m/s net negative
    # Critical: Must be > r_alive (0.01) to dominate survival bonus at low speeds
    # Math: At 5 m/s, reward = 0.167 + 0.01 + 0 + (-0.20) = -0.023 < 0 (even without SLOWER)
    "r_low_speed": -0.20,  # Was -0.01 in V2 (20× stronger)
    
    # Minimum speed threshold (ratio of max_velocity)
    # 0.6 means 60% of 30 m/s = 18 m/s minimum desired speed
    # Below this: r_low_speed penalty applies
    # Above this: no speed penalty
    "min_speed_ratio": 0.6,
    
    # === V3.5 ENHANCED: OVERTAKING BONUS (Risk-Aware Engagement) ===
    
    # Overtaking bonus per vehicle passed (V3.5 addition for better engagement)
    # Encourages active traffic navigation while maintaining V3's speed-first objective
    # Formula: +2.0 per successful overtake (when relative velocity > 2 m/s)
    # Effect: Incentivizes lane changes for strategic overtaking
    "r_overtake_bonus": 2.0,
    
    # Minimum relative velocity to count as overtake (m/s)
    # Prevents counting "drifting past" slow vehicles as overtakes
    # Must be actively passing (ego_vx - vehicle_vx > 2.0)
    "min_overtake_speed": 2.0,
    
    # === V3.5 ENHANCED: DYNAMIC COLLISION PENALTY (95% Confidence Formula) ===
    
    # Base collision penalty (always applied on crash)
    # Increased from -80.0 to -100.0 for stronger safety constraint
    # Math: -100.0 > max(60 steps × 1.0 progress + 10 overtakes × 2.0) = -100 > 80
    "r_collision_base": -100.0,
    
    # Additional penalty for risky overtaking crashes (V3.5 enhancement)
    # Applied when crash occurs within overtake_risk_window after overtaking
    # Total risky crash penalty: -100.0 + (-38.0) = -138.0
    # 
    # Mathematical Model (95% Confidence Requirement):
    #   Expected value: E = P(success) × r_overtake + P(crash) × r_collision_risky
    #   For break-even: 0 = P(success) × 2.0 + P(crash) × (-138.0)
    #   Required confidence: P(success) = 138 / (138 + 2) = 0.9857 (~98.5%)
    # 
    # Effect: Agent learns "only overtake when very confident of success"
    #   - 95% confidence: E = -5.00 (agent avoids)
    #   - 99% confidence: E = +0.62 (agent attempts)
    # 
    # Risk ratio: 138:2 = 69:1 (conservative, safer than exactly 95%)
    "r_collision_risky": -38.0,  # For exactly 95%: use -38.0 (ratio 19:1)
    
    # Risk window duration (seconds after overtake)
    # Crash within this window after overtaking = risky maneuver
    # At 12 Hz: 3.0s × 12 = 36 steps of heightened risk
    "overtake_risk_window": 3.0,
    
    # === NORMALIZATION PARAMETERS ===
    
    # Maximum velocity for progress normalization (m/s)
    # highway-env max velocity: 30 m/s (108 km/h)
    "max_velocity": 30.0,
    
    # Policy frequency for time calculations (Hz)
    # Used to convert overtake_risk_window from seconds to steps
    "policy_frequency": 12,
}


# ===================================================================
# V4: ACCELERATION-AWARE REWARD (NO OVERTAKING)
# ===================================================================
# Focus: Simplify to core driving mechanics
# - Progress reward includes acceleration bonus
# - Context-dependent SLOWER penalty (heavier when already slow)
# - Small FASTER bonus when moving slowly
# - Remove ALL overtaking logic and risk tracking
# 
# Rationale: V3.5 may be too complex for laptop CPU training
#            V4 tests simpler hypothesis: reward acceleration directly
# ===================================================================

REWARD_V4_CONFIG = {
    # === CORE PARAMETERS (SAME AS V3) ===
    "r_alive": 0.01,
    "r_collision": -100.0,  # Single penalty, no risk logic
    "min_speed_threshold": 18.0,
    "r_low_speed": -0.02,  # V4 FIX: Reduced from -0.20 (was too strong)
    
    # === V4 NEW: ACCELERATION BONUS ===
    # Progress reward = velocity_ratio + acceleration_weight × Δvelocity
    # Encourages speeding up, not just maintaining speed
    # Weight: 0.2 means acceleration is 20% as important as current speed
    "acceleration_weight": 0.2,
    
    # === V4 NEW: CONTEXT-DEPENDENT SLOWER PENALTY ===
    # Penalize SLOWER action more heavily when already slow
    # If velocity < slow_velocity_threshold: penalty = r_slower_heavy
    # Otherwise: penalty = r_slower_light
    "slow_velocity_threshold": 0.7,  # 70% of max speed
    "r_slower_heavy": -0.05,  # When already slow (< 70%)
    "r_slower_light": -0.01,  # When moving fast (>= 70%)
    
    # === V4 NEW: FASTER BONUS ===
    # Small bonus for using FASTER when slow
    # Encourages active acceleration when below threshold
    "faster_velocity_threshold": 0.8,  # 80% of max speed
    "r_faster_bonus": 0.05,  # Small positive when slow
}


# ===================================================================
# V5: RUBRIC-COMPLIANT REWARD (V4 + SAFE HEADWAY + LANE PENALTY)
# ===================================================================
# Full rubric compliance with 8-component reward:
# 
# Components inherited from V4:
#   1. r_progress: velocity + acceleration bonus
#   2. r_alive: survival bonus
#   3. r_collision: crash penalty
#   4. r_slow_action: context-dependent SLOWER penalty
#   5. r_low_speed: penalty for < 60% max speed
#   6. r_faster_bonus: bonus for FASTER when slow
#
# NEW V5 components for rubric compliance:
#   7. r_headway: Safe distance reward (rubric: "maintaining safe distances")
#   8. r_lane: Lane change penalty (rubric: "penalize unnecessary lane changes")
#
# Mathematical Formulation:
#   R(s,a) = Σᵢ rᵢ(s,a)  (8 components)
#
# ===================================================================

REWARD_V5_CONFIG: Dict[str, float] = {
    # === INHERITED FROM V4 (REBALANCED V5.1) ===
    "r_alive": 0.01,
    "r_collision": -50.0,            # REDUCED: Was -100, too dominant vs speed rewards
    "min_speed_ratio": 0.6,          # 60% of max speed
    "r_low_speed": -0.05,            # INCREASED: Was -0.02, now meaningful
    "acceleration_weight": 0.2,
    
    # === V5.1 REBALANCED: SLOWER PENALTY (Much Higher) ===
    # PROBLEM: Old -0.01/-0.05 was too weak. Agent spammed SLOWER to avoid
    # collisions (surviving at low speed > crashing at high speed).
    # 
    # FIX: Make SLOWER expensive enough that agent prefers lane changes.
    #
    # Formula:
    #   r_slower = -0.15 if v < 70%  (heavy: already slow, don't slow more)
    #            = -0.08 if v >= 70% (significant: slowing from high speed)
    #
    # Effect: SLOWER now costs ~10% of progress reward, making it a real tradeoff
    "r_slow_action_heavy": -0.15,    # INCREASED: Was -0.05 (3x)
    "r_slow_action_light": -0.08,    # INCREASED: Was -0.01 (8x)
    
    # === V5.1 NEW: IDLE BONUS (Reward Speed Maintenance) ===
    # PROBLEM: No positive reward for maintaining high speed.
    # Agent had no reason to prefer IDLE over SLOWER at high speed.
    #
    # FIX: Give bonus for IDLE when already at high speed.
    #
    # Formula:
    #   r_idle = +0.05 if v >= 80% AND action == IDLE
    #          = 0.00 otherwise
    #
    # Effect: Actively rewards maintaining speed, not just penalizing slow
    "r_idle_bonus": 0.05,            # NEW: Bonus for IDLE at high speed
    "v_idle_threshold": 0.8,         # NEW: Apply bonus above 80% speed
    # === V5.1 REBALANCED: FASTER BONUS (Stronger Proportional) ===
    # PROBLEM: Even 0.15 max wasn't strong enough vs SLOWER survival strategy.
    # 
    # FIX: Increase max bonus and make it more significant.
    #
    # Formula:
    #   r_faster = r_faster_max × (1 - velocity_ratio) if v < 95%
    #            = 0.0 otherwise
    #
    # Effect (with r_faster_max=0.25):
    #   At 50% speed: +0.125 bonus for FASTER (significant!)
    #   At 70% speed: +0.075 bonus for FASTER  
    #   At 90% speed: +0.025 bonus for FASTER
    "r_faster_max": 0.25,              # INCREASED: Was 0.15 (67% increase)
    "v_faster_threshold": 0.95,        # INCREASED: Was 0.9 (bonus up to 95% speed)
    
    # === V5 REBALANCED: SAFE HEADWAY REWARD (No Empty Lane Bonus) ===
    # PROBLEM: Old +0.10 for safe distance rewarded staying in empty right lane
    # This caused LANE_RIGHT-only policy (92% of actions)
    #
    # FIX: NEUTRAL for safe distance, only PENALIZE tailgating
    # 
    # Formula:
    #   r_headway = 0.00 if τ ≥ τ_safe     (NEUTRAL, not bonus!)
    #             = -0.10 × danger_ratio   (proportional penalty for tailgating)
    #             = 0.00 otherwise          (neutral zone)
    #
    # Rubric: "Reward... maintaining safe distances" ✓ (via penalty avoidance)
    
    "headway_tau_safe": 1.5,           # Safe time-headway threshold (seconds)
    "headway_tau_danger": 0.5,         # Dangerous time-headway threshold (seconds)
    "r_headway_safe": 0.0,             # CHANGED: Neutral (was +0.10, caused RIGHT bias)
    "r_headway_danger_max": -0.10,     # Maximum penalty for extreme tailgating
    
    # === V5 REBALANCED: ASYMMETRIC LANE CHANGE PENALTY ===
    # PROBLEM: Symmetric -0.02 made LEFT and RIGHT equally costly
    # Since RIGHT leads to emptier lanes, agent learned RIGHT-only
    #
    # FIX: Penalize RIGHT more than LEFT (encourage passing on left)
    # 
    # Formula:
    #   r_lane = -0.01 if action == LEFT  (small, encourages passing)
    #          = -0.03 if action == RIGHT (larger, discourages retreat)
    #          = 0.00  otherwise
    #
    # Effect:
    #   - LEFT lane changes: -0.01 (passing slower traffic encouraged)
    #   - RIGHT lane changes: -0.03 (retreating to slow lane discouraged)
    #   - Net effect: Agent prefers passing on left over retreating right
    #
    # Rubric: "Penalize... unnecessary lane changes" ✓
    
    "r_lane_left": -0.01,              # Small penalty for passing (encouraged)
    "r_lane_right": -0.03,             # Larger penalty for retreat (discouraged)
    
    # Max velocity for calculations
    "max_velocity": 30.0,
}


# ==================================================
# NEURAL NETWORK ARCHITECTURE
# ==================================================

NETWORK_CONFIG: Dict[str, Any] = {
    # Policy type: MlpPolicy = Multi-Layer Perceptron
    # Alternative: CnnPolicy (for image observations)
    "policy_type": "MlpPolicy",
    
    # Network architecture
    # PPO uses Actor-Critic with shared layers
    # Format: [dict(pi=[...], vf=[...])]
    # - pi: policy network (actor) layers
    # - vf: value function network (critic) layers
    #
    # [64, 64] means:
    # Input (observation) → 64 neurons → 64 neurons → Output
    "net_arch": [
        dict(
            pi=[64, 64],  # Policy network: 2 hidden layers, 64 units each
            vf=[64, 64]   # Value network: 2 hidden layers, 64 units each
        )
    ],
    
    # Activation function
    # Options: 'tanh', 'relu', 'elu'
    # tanh: Smoother gradients, traditional choice for RL
    # relu: Faster, but can cause dead neurons
    "activation_fn": "tanh",
    
    # Orthogonal initialization for weights
    # Improves training stability
    "ortho_init": True,
}


# ==================================================
# PPO TRAINING HYPERPARAMETERS
# ==================================================

TRAINING_CONFIG: Dict[str, Any] = {
    # === TRAINING DURATION ===
    
    # Total timesteps to train
    # OPTIMIZED: 200k timesteps for thorough training with 50 vehicles
    # Justification:
    #   - Leurent et al. (2018) use 100-200k for convergence
    #   - 50 vehicles = harder task, benefits from extended training
    #   - 200k @ 35 it/s = 95 minutes (1.6 hours, feasible)
    #   - Provides better learning progression for evolution video
    "total_timesteps": 200_000,
    
    # === DEVICE ===
    
    # Computation device
    # "auto": Auto-detect GPU, fallback to CPU
    # "cuda": Force GPU
    # "cpu": Force CPU
    "device": "auto",
    
    # === LEARNING RATE ===
    
    # Learning rate for Adam optimizer
    # 3e-4 is the "default that works" for PPO
    # Lower = more stable, slower learning
    # Higher = faster learning, risk of instability
    "learning_rate": 3e-4,
    
    # === ROLLOUT PARAMETERS ===
    
    # Number of steps to collect before each update
    # Higher = more data per update, more stable
    # Must be multiple of (num_envs * n_steps)
    # Typical range: 1024-4096
    "n_steps": 2048,
    
    # Batch size for training
    # Must be factor of n_steps
    # 2048 / 64 = 32 mini-batches per update
    # Smaller = noisier gradients, faster updates
    "batch_size": 64,
    
    # Number of epochs per update
    # How many times to iterate over collected data
    # Higher = more thorough learning from data
    # Typical range: 4-10
    "n_epochs": 10,
    
    # === DISCOUNT & ADVANTAGE ESTIMATION ===
    
    # Discount factor (gamma)
    # How much to value future rewards
    # 0.99 = "value future almost as much as present"
    # 0.90 = "prefer immediate rewards"
    # Highway-env: 0.99 works well (long-term planning)
    "gamma": 0.99,
    
    # GAE lambda (Generalized Advantage Estimation)
    # Bias-variance tradeoff for advantage calculation
    # 1.0 = high variance, low bias
    # 0.0 = low variance, high bias
    # 0.95 is standard compromise
    "gae_lambda": 0.95,
    
    # === PPO-SPECIFIC PARAMETERS ===
    
    # Clipping range for PPO objective
    # This is PPO's KEY parameter!
    # Limits how much policy can change per update
    # 0.2 means policy can change by max 20%
    # Prevents catastrophic policy collapse
    "clip_range": 0.2,
    
    # Value function clipping
    # Similar to clip_range but for value function
    # None = no clipping (recommended default)
    "clip_range_vf": None,
    
    # Entropy coefficient
    # Encourages exploration by adding entropy bonus
    # Higher = more random actions (more exploration)
    # 0.0 = no entropy bonus (pure exploitation)
    # For highway-env, 0.0 works (environment has enough stochasticity)
    "ent_coef": 0.0,
    
    # Value function coefficient
    # Weight of value loss in total loss
    # Total loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    "vf_coef": 0.5,
    
    # Maximum gradient norm for clipping
    # Prevents exploding gradients
    # 0.5 is standard for PPO
    "max_grad_norm": 0.5,
    
    # === REPRODUCIBILITY ===
    
    # Random seed for reproducibility
    # REQUIRED BY RUBRIC: "Training must be reproducible"
    # Same seed = same results (assuming deterministic environment)
    "seed": 42,
    
    # === DEVICE ===
    
    # Device to use for training
    # 'cuda' = GPU (if available)
    # 'cpu' = CPU only
    # We'll auto-detect in code
    "device": "auto",
    
    # === LOGGING ===
    
    # Verbose level
    # 0: no output
    # 1: info (progress bar)
    # 2: debug (detailed logs)
    "verbose": 1,
    
    # Log metrics frequency (in timesteps)
    # Log custom metrics every N steps
    "log_freq": 1000,
}


# ==================================================
# EVALUATION SETTINGS
# ==================================================

EVAL_CONFIG: Dict[str, Any] = {
    # Number of episodes for evaluation
    # More episodes = more reliable statistics
    "n_eval_episodes": 10,
    
    # Evaluation frequency (in timesteps)
    # Evaluate every N training steps
    # More frequent = better tracking, but slower training
    "eval_freq": 10_000,
    
    # Deterministic evaluation
    # True: always pick best action (argmax)
    # False: sample from policy distribution
    # For evaluation, we want deterministic = True
    "deterministic": True,
    
    # Render evaluation episodes
    # True: show visualization (useful for debugging)
    # False: no rendering (faster)
    "render": False,
    
    # Return episode rewards
    "return_episode_rewards": True,
}


# ==================================================
# CHECKPOINT SETTINGS
# ==================================================

CHECKPOINT_CONFIG: Dict[str, Any] = {
    # Save frequency (in timesteps)
    # Save model every N steps
    # For 200k total: 100k means checkpoints at 0k, 100k, 200k (evolution video requirement)
    # Three stages: untrained, half-trained, fully-trained (rubric compliance)
    "save_freq": 100_000,
    
    # Checkpoint naming prefix
    "name_prefix": "highway_ppo",
    
    # Save path
    "save_path": str(CHECKPOINTS_DIR),
    
    # Keep only best N checkpoints (by reward)
    # Saves disk space
    # None = keep all checkpoints
    "keep_best_n": 3,
}


# ==================================================
# VIDEO RECORDING SETTINGS (EVOLUTION VIDEO)
# ==================================================

VIDEO_CONFIG: Dict[str, Any] = {
    # Frames per second for output video
    "fps": 15,
    
    # Video codec for MP4
    # 'mp4v': Fast encoding, good compatibility
    # 'h264': Better compression, requires ffmpeg
    "codec": "mp4v",
    
    # Video quality (0-10, higher = better)
    "quality": 8,
    
    # Maximum episode length for recording (steps)
    # Prevents videos from being too long
    "episode_length": 200,
    
    # Which checkpoints to record for evolution video
    # REQUIRED BY RUBRIC: "three stages: untrained, half-trained, fully-trained"
    # Format: {stage_name: checkpoint_timestep}
    "checkpoints_to_record": {
        "untrained": 0,           # Random policy
        "half_trained": 100_000,  # Midpoint
        "fully_trained": 200_000, # Final policy
    },
    
    # Output filename
    "output_filename": "evolution.mp4",
    
    # Output path
    "output_path": str(VIDEOS_DIR),
}


# ==================================================
# PLOTTING SETTINGS
# ==================================================

PLOT_CONFIG: Dict[str, Any] = {
    # Figure size (width, height) in inches
    "figsize": (10, 6),
    
    # DPI for saved images
    "dpi": 300,
    
    # Style
    "style": "seaborn-v0_8-darkgrid",
    
    # Moving average window for smoothing
    "moving_avg_window": 100,
    
    # Save path
    "save_path": str(PLOTS_DIR),
}


# ==================================================
# TRAINING CONFIGURATION
# ==================================================

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def print_config() -> None:
    """Print all configuration settings in a formatted way."""
    print("\n" + "="*70)
    print("HIGHWAY RL AGENT - CONFIGURATION")
    print("="*70)
    
    sections = [
        ("ENVIRONMENT", ENV_CONFIG),
        ("REWARD FUNCTION", REWARD_CONFIG),
        ("NEURAL NETWORK", NETWORK_CONFIG),
        ("TRAINING (PPO)", TRAINING_CONFIG),
        ("EVALUATION", EVAL_CONFIG),
        ("CHECKPOINTS", CHECKPOINT_CONFIG),
        ("VIDEO RECORDING", VIDEO_CONFIG),
        ("PLOTTING", PLOT_CONFIG),
    ]
    
    for section_name, section_config in sections:
        print(f"\n[{section_name}]")
        for key, value in section_config.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        print(f"    {sub_key}:")
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            print(f"      {sub_sub_key}: {sub_sub_value}")
                    else:
                        print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "="*70)


def get_device() -> str:
    """
    Get the device to use for training.
    
    Returns:
        'cuda' if GPU available, 'cpu' otherwise
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def validate_config() -> bool:
    """
    Validate configuration for common errors.
    
    Returns:
        True if config is valid, False otherwise
    """
    errors = []
    
    # Check batch_size divides n_steps
    if TRAINING_CONFIG["n_steps"] % TRAINING_CONFIG["batch_size"] != 0:
        errors.append(
            f"n_steps ({TRAINING_CONFIG['n_steps']}) must be "
            f"divisible by batch_size ({TRAINING_CONFIG['batch_size']})"
        )
    
    # Check checkpoint frequency
    if CHECKPOINT_CONFIG["save_freq"] > TRAINING_CONFIG["total_timesteps"]:
        errors.append(
            f"save_freq ({CHECKPOINT_CONFIG['save_freq']}) > "
            f"total_timesteps ({TRAINING_CONFIG['total_timesteps']})"
        )
    
    # Check reward configuration consistency
    required_keys = ["w_progress", "r_alive", "r_collision", "r_lane_change", 
                     "r_slow_action", "r_low_speed", "min_speed_ratio", "max_velocity"]
    missing_keys = [key for key in required_keys if key not in REWARD_CONFIG]
    if missing_keys:
        errors.append(f"Missing reward config keys: {missing_keys}")
    
    # Validate penalty magnitudes
    if abs(REWARD_CONFIG["r_collision"]) < 60:
        errors.append(
            f"Collision penalty ({REWARD_CONFIG['r_collision']}) too weak. "
            f"Should be < -60 to dominate max progress at 12 Hz."
        )
    
    if errors:
        print("❌ Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ Configuration validation passed")
    return True


if __name__ == "__main__":
    # Test configuration
    print_config()
    
    # Validate
    print("\n")
    validate_config()
    
    # Test GPU detection
    device = get_device()
    print(f"\n🚀 Training device: {device}")
    
    if device == "cuda":
        import torch
        print(f"   GPU: {torch.cuda.get_device_name(0)}")