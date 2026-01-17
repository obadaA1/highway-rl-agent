"""
Quick test to visualize trained agent behavior.
Shows real-time driving with speed and action display.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.highway_env import make_highway_env
from src.agent.ppo_agent import HighwayPPOAgent
import time

def test_agent(checkpoint_path: str, episodes: int = 3):
    """
    Test agent with visual rendering and behavior analysis.
    """
    print(f"\n{'='*70}")
    print(f"TESTING AGENT BEHAVIOR")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {episodes}")
    print(f"\nWatch for:")
    print("  - Speed (velocity)")
    print("  - Lane changes (overtaking)")
    print("  - Distance to other vehicles")
    print(f"{'='*70}\n")
    
    # Create environment with human rendering
    env = make_highway_env(render_mode="human")
    
    # Load agent
    agent = HighwayPPOAgent.load(checkpoint_path, env=env)
    
    # Run episodes
    for ep in range(episodes):
        print(f"\n🏁 Episode {ep + 1}/{episodes}")
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0
        steps = 0
        lane_changes = 0
        prev_action = None
        velocities = []
        
        while not (done or truncated):
            # Get action
            action, _ = agent.model.predict(obs, deterministic=True)
            
            # Track lane changes
            if prev_action is not None and action in [0, 2]:
                lane_changes += 1
            prev_action = action
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Track metrics
            total_reward += reward
            steps += 1
            
            # Extract velocity from observation (feature index 3 = vx)
            if hasattr(env, 'unwrapped'):
                base_env = env.unwrapped
                if hasattr(base_env, 'vehicle'):
                    velocity = base_env.vehicle.speed  # m/s
                    velocities.append(velocity)
            
            # Render (already handled by environment)
            time.sleep(0.05)  # Slow down for visibility
        
        # Episode summary
        avg_velocity = sum(velocities) / len(velocities) if velocities else 0
        duration = steps / 15  # 15 Hz
        
        print(f"   Duration: {duration:.1f}s ({steps} steps)")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Lane Changes: {lane_changes}")
        print(f"   Avg Velocity: {avg_velocity:.1f} m/s ({avg_velocity * 3.6:.1f} km/h)")
        print(f"   Termination: {'collision' if done else 'time limit'}")
    
    env.close()
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    # Test the fully trained agent
    checkpoint = "assets/checkpoints/highway_ppo_100000_steps.zip"
    test_agent(checkpoint, episodes=3)
