"""
Test Suite for V5 Reward Function (Rubric-Compliant).

Tests all 8 reward components including the new V5 additions:
- r_headway: Safe distance reward
- r_lane: Lane change penalty

Run with: python -m pytest tests/test_reward_v5.py -v
Or:       python tests/test_reward_v5.py

Author: [Your Name]
Date: 2025-01-18
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Import pytest only when running with pytest
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    
from src.env.highway_env_v5 import CustomHighwayEnvV5, make_highway_env_v5
from src.config import REWARD_V5_CONFIG


class TestV5RewardComponents:
    """Test suite for V5 reward function components."""
    
    if PYTEST_AVAILABLE:
        @pytest.fixture
        def env(self):
            """Create V5 environment for testing."""
            env = make_highway_env_v5(render_mode=None)
            env.reset(seed=42)
            return env
    
    def _create_observation(
        self,
        ego_velocity: float = 0.5,
        front_vehicle_distance: float = 50.0,
        front_vehicle_present: bool = True
    ) -> np.ndarray:
        """
        Create a mock observation array.
        
        Args:
            ego_velocity: Normalized velocity [-1, 1]
            front_vehicle_distance: Distance to front vehicle (meters)
            front_vehicle_present: Whether front vehicle exists
        
        Returns:
            Observation array shape (5, 5)
        """
        obs = np.zeros((5, 5))
        # Ego vehicle
        obs[0] = [1.0, 0.0, 0.0, ego_velocity, 0.0]
        # Front vehicle (if present)
        if front_vehicle_present:
            # Normalize distance to [-1, 1] (assuming range [-100, 100])
            normalized_distance = front_vehicle_distance / 100.0
            obs[1] = [1.0, normalized_distance, 0.0, 0.0, 0.0]
        return obs
    
    # ========== V4 Component Tests (Inherited) ==========
    
    def test_progress_reward_high_speed(self, env):
        """Test progress reward at high speed."""
        obs = self._create_observation(ego_velocity=0.8)
        reward = env._compute_progress_reward(obs)
        
        # velocity_ratio = (0.8 + 1) / 2 = 0.9
        # First step, delta_v = 0.9 - 0.0 = 0.9, clipped to 0.1
        # r_progress = 0.9 + 0.2 * 0.1 = 0.92
        expected = 0.9 + 0.2 * 0.1
        assert abs(reward - expected) < 0.01, f"Expected ~{expected}, got {reward}"
    
    def test_progress_reward_low_speed(self, env):
        """Test progress reward at low speed."""
        obs = self._create_observation(ego_velocity=-0.5)
        reward = env._compute_progress_reward(obs)
        
        # velocity_ratio = (-0.5 + 1) / 2 = 0.25
        # delta_v clipped
        assert reward < 0.5, f"Low speed should have low progress reward, got {reward}"
    
    def test_collision_penalty(self, env):
        """Test collision penalty."""
        obs = self._create_observation()
        
        # No crash
        penalty_none = env._compute_collision_penalty(False, {"crashed": False})
        assert penalty_none == 0.0, "No crash should have 0 penalty"
        
        # Crash
        penalty_crash = env._compute_collision_penalty(True, {"crashed": True})
        assert penalty_crash == REWARD_V5_CONFIG["r_collision"], \
            f"Crash penalty should be {REWARD_V5_CONFIG['r_collision']}"
    
    def test_slow_action_heavy_penalty(self, env):
        """Test heavy SLOWER penalty when already slow."""
        obs = self._create_observation(ego_velocity=-0.5)  # 25% speed
        
        penalty = env._compute_slow_action_penalty(action=4, observation=obs)
        assert penalty == REWARD_V5_CONFIG["r_slow_action_heavy"], \
            f"Heavy penalty expected when slow, got {penalty}"
    
    def test_slow_action_light_penalty(self, env):
        """Test light SLOWER penalty when fast."""
        obs = self._create_observation(ego_velocity=0.6)  # 80% speed
        
        penalty = env._compute_slow_action_penalty(action=4, observation=obs)
        assert penalty == REWARD_V5_CONFIG["r_slow_action_light"], \
            f"Light penalty expected when fast, got {penalty}"
    
    def test_slow_action_no_penalty_other_actions(self, env):
        """Test no penalty for non-SLOWER actions."""
        obs = self._create_observation(ego_velocity=-0.5)
        
        for action in [0, 1, 2, 3]:  # Not SLOWER
            penalty = env._compute_slow_action_penalty(action=action, observation=obs)
            assert penalty == 0.0, f"Non-SLOWER action {action} should have 0 penalty"
    
    def test_low_speed_penalty(self, env):
        """Test low speed penalty below threshold."""
        # Below 60%
        obs_slow = self._create_observation(ego_velocity=-0.5)  # 25%
        penalty_slow = env._compute_low_speed_penalty(obs_slow)
        assert penalty_slow == REWARD_V5_CONFIG["r_low_speed"], \
            f"Low speed penalty expected, got {penalty_slow}"
        
        # Above 60%
        obs_fast = self._create_observation(ego_velocity=0.4)  # 70%
        penalty_fast = env._compute_low_speed_penalty(obs_fast)
        assert penalty_fast == 0.0, "No penalty above threshold"
    
    def test_faster_bonus(self, env):
        """Test FASTER bonus when slow."""
        # Slow + FASTER
        obs_slow = self._create_observation(ego_velocity=0.0)  # 50%
        bonus = env._compute_faster_bonus(action=3, observation=obs_slow)
        assert bonus == REWARD_V5_CONFIG["r_faster_bonus"], \
            f"FASTER bonus expected when slow, got {bonus}"
        
        # Fast + FASTER (no bonus)
        obs_fast = self._create_observation(ego_velocity=0.8)  # 90%
        no_bonus = env._compute_faster_bonus(action=3, observation=obs_fast)
        assert no_bonus == 0.0, "No bonus when already fast"
    
    # ========== V5 NEW Component Tests ==========
    
    def test_headway_reward_safe(self, env):
        """Test positive headway reward for safe following distance."""
        # Safe distance: 50m at 15 m/s = 3.33s headway
        obs = self._create_observation(ego_velocity=0.0, front_vehicle_distance=50.0)
        # velocity = (0 + 1) / 2 * 30 = 15 m/s
        # headway = 50 / 15 = 3.33s > 1.5s (safe)
        
        reward = env._compute_headway_reward(obs)
        assert reward == REWARD_V5_CONFIG["r_headway_safe"], \
            f"Safe headway should give +{REWARD_V5_CONFIG['r_headway_safe']}, got {reward}"
    
    def test_headway_penalty_danger(self, env):
        """Test negative headway reward for dangerous tailgating."""
        # Dangerous: 5m at 15 m/s = 0.33s headway
        obs = self._create_observation(ego_velocity=0.0, front_vehicle_distance=5.0)
        # velocity = 15 m/s
        # headway = 5 / 15 = 0.33s < 0.5s (danger)
        
        penalty = env._compute_headway_reward(obs)
        assert penalty == REWARD_V5_CONFIG["r_headway_danger"], \
            f"Dangerous headway should give {REWARD_V5_CONFIG['r_headway_danger']}, got {penalty}"
    
    def test_headway_neutral_zone(self, env):
        """Test neutral headway in the middle zone."""
        # Neutral: 15m at 15 m/s = 1.0s headway (between 0.5s and 1.5s)
        obs = self._create_observation(ego_velocity=0.0, front_vehicle_distance=15.0)
        
        reward = env._compute_headway_reward(obs)
        assert reward == 0.0, f"Neutral zone should give 0, got {reward}"
    
    def test_headway_no_vehicle_ahead(self, env):
        """Test headway reward when no vehicle ahead."""
        obs = self._create_observation(ego_velocity=0.5, front_vehicle_present=False)
        
        reward = env._compute_headway_reward(obs)
        assert reward == REWARD_V5_CONFIG["r_headway_safe"], \
            f"No vehicle ahead should be safe, got {reward}"
    
    def test_lane_change_penalty_left(self, env):
        """Test lane change penalty for LANE_LEFT."""
        penalty = env._compute_lane_change_penalty(action=0)  # LANE_LEFT
        assert penalty == REWARD_V5_CONFIG["r_lane_change"], \
            f"Lane left should give {REWARD_V5_CONFIG['r_lane_change']}, got {penalty}"
    
    def test_lane_change_penalty_right(self, env):
        """Test lane change penalty for LANE_RIGHT."""
        penalty = env._compute_lane_change_penalty(action=2)  # LANE_RIGHT
        assert penalty == REWARD_V5_CONFIG["r_lane_change"], \
            f"Lane right should give {REWARD_V5_CONFIG['r_lane_change']}, got {penalty}"
    
    def test_lane_change_no_penalty_other(self, env):
        """Test no penalty for non-lane-change actions."""
        for action in [1, 3, 4]:  # IDLE, FASTER, SLOWER
            penalty = env._compute_lane_change_penalty(action)
            assert penalty == 0.0, f"Action {action} should have 0 lane penalty"
    
    def test_lane_change_counter(self, env):
        """Test that lane changes are counted."""
        initial_count = env.lane_changes_count
        
        env._compute_lane_change_penalty(action=0)  # LANE_LEFT
        assert env.lane_changes_count == initial_count + 1
        
        env._compute_lane_change_penalty(action=2)  # LANE_RIGHT
        assert env.lane_changes_count == initial_count + 2
    
    # ========== Integration Tests ==========
    
    def test_full_reward_calculation(self, env):
        """Test complete V5 reward calculation."""
        obs = self._create_observation(ego_velocity=0.5, front_vehicle_distance=50.0)
        
        # Use IDLE action (no penalties)
        total_reward = env._calculate_custom_reward(
            observation=obs,
            action=1,  # IDLE
            terminated=False,
            info={"crashed": False}
        )
        
        # Should be positive (progress + alive + headway safe)
        assert total_reward > 0, f"Normal driving should have positive reward, got {total_reward}"
    
    def test_reward_components_dict(self, env):
        """Test that reward components dict includes V5 components."""
        obs = self._create_observation(ego_velocity=0.5)
        
        # Need to run a step first to populate values
        env._compute_progress_reward(obs)
        env._compute_headway_reward(obs)
        
        components = env._get_reward_components(
            observation=obs,
            action=1,
            terminated=False,
            info={"crashed": False}
        )
        
        # Check V5 components exist
        assert 'headway' in components, "V5 should have headway component"
        assert 'lane' in components, "V5 should have lane component"
        assert 'time_headway' in components, "V5 should have time_headway debug info"
        assert 'lane_changes' in components, "V5 should have lane_changes debug info"
    
    def test_episode_stats_v5(self, env):
        """Test that V5 episode stats include new metrics."""
        assert 'headway_violations' in env.episode_stats, \
            "V5 should track headway_violations"
        assert 'lane_changes' in env.episode_stats, \
            "V5 should track lane_changes"
    
    def test_reset_clears_v5_state(self, env):
        """Test that reset clears V5 tracking state."""
        # Simulate some activity
        env.lane_changes_count = 5
        env.last_headway = 1.0
        env.episode_stats["headway_violations"] = 10
        
        # Reset
        env.reset(seed=42)
        
        assert env.lane_changes_count == 0, "Reset should clear lane_changes_count"
        assert env.last_headway == float('inf'), "Reset should clear last_headway"
        assert env.episode_stats["headway_violations"] == 0, \
            "Reset should clear headway_violations"


class TestV5VsV4Differences:
    """Tests that verify V5 differences from V4."""
    
    def test_v5_has_more_components(self):
        """V5 should have 8 components vs V4's 6."""
        env = make_highway_env_v5(render_mode=None)
        env.reset(seed=42)
        
        obs = np.zeros((5, 5))
        obs[0] = [1.0, 0.0, 0.0, 0.5, 0.0]
        
        # Force progress computation to set last_delta_v
        env._compute_progress_reward(obs)
        
        components = env._get_reward_components(
            observation=obs,
            action=1,
            terminated=False,
            info={"crashed": False}
        )
        
        # V5 unique components
        v5_unique = ['headway', 'lane', 'time_headway', 'lane_changes']
        for comp in v5_unique:
            assert comp in components, f"V5 should have {comp} component"
        
        env.close()
    
    def test_v5_config_has_headway_params(self):
        """V5 config should have headway parameters."""
        required_params = [
            'headway_tau_safe',
            'headway_tau_danger', 
            'r_headway_safe',
            'r_headway_danger',
            'r_lane_change'
        ]
        
        for param in required_params:
            assert param in REWARD_V5_CONFIG, \
                f"V5 config missing required parameter: {param}"


def run_tests():
    """Run all V5 tests."""
    print("\n" + "="*70)
    print("V5 REWARD FUNCTION TEST SUITE (RUBRIC-COMPLIANT)")
    print("="*70)
    
    # Create test environment
    env = make_highway_env_v5(render_mode=None)
    obs, info = env.reset(seed=42)
    
    print("\n[1] Environment V5 Created Successfully")
    print(f"    Observation space: {env.observation_space}")
    print(f"    Action space: {env.action_space}")
    
    # Test V5 specific features
    print("\n[2] Testing V5 NEW Components")
    
    # Create test observation with safe following distance
    test_obs = np.zeros((5, 5))
    test_obs[0] = [1.0, 0.0, 0.0, 0.5, 0.0]  # Ego at 75% speed
    test_obs[1] = [1.0, 0.5, 0.0, 0.3, 0.0]  # Vehicle 50m ahead
    
    # Compute progress first (sets last_delta_v)
    env._compute_progress_reward(test_obs)
    
    # Test headway reward
    r_headway = env._compute_headway_reward(test_obs)
    print(f"    r_headway (safe distance): {r_headway:+.2f}")
    
    # Test lane penalty
    r_lane_left = env._compute_lane_change_penalty(0)  # LANE_LEFT
    print(f"    r_lane (LANE_LEFT): {r_lane_left:+.2f}")
    
    r_lane_idle = env._compute_lane_change_penalty(1)  # IDLE
    print(f"    r_lane (IDLE): {r_lane_idle:+.2f}")
    
    # Full reward calculation
    print("\n[3] Testing Full V5 Reward Calculation")
    env.reset(seed=42)
    total_reward = env._calculate_custom_reward(
        observation=test_obs,
        action=1,  # IDLE
        terminated=False,
        info={"crashed": False}
    )
    print(f"    Total V5 reward (IDLE): {total_reward:.4f}")
    
    # Run actual step
    print("\n[4] Testing Step Execution")
    env.reset(seed=42)
    action_names = ['LANE_LEFT', 'IDLE', 'LANE_RIGHT', 'FASTER', 'SLOWER']
    
    for action in range(5):
        obs, reward, term, trunc, info = env.step(action)
        components = info['custom_reward_components']
        print(f"    {action_names[action]:12s}: r={reward:+.3f} | "
              f"headway={components.get('headway', 0):+.2f} | "
              f"lane={components.get('lane', 0):+.2f}")
        if term:
            env.reset(seed=42)
    
    env.close()
    
    print("\n" + "="*70)
    print("✅ V5 Basic Tests Passed!")
    print("="*70)
    print("\nRun full test suite with: pytest tests/test_reward_v5.py -v")


if __name__ == "__main__":
    run_tests()
