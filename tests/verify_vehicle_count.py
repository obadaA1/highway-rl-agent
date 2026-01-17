"""
Quick verification that environment has 30 vehicles.

This script creates the environment and checks the actual
number of vehicles in the simulation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.highway_env import make_highway_env

def main():
    print("Creating environment...")
    env = make_highway_env(render_mode=None)
    
    print("Resetting environment...")
    obs, info = env.reset()
    
    # Access the underlying highway-env
    base_env = env.unwrapped
    
    # Get the road and count vehicles
    road = base_env.road
    vehicles = road.vehicles
    
    print(f"\n{'='*70}")
    print(f"VEHICLE COUNT VERIFICATION")
    print(f"{'='*70}")
    print(f"Total vehicles in simulation: {len(vehicles)}")
    print(f"  - Ego vehicle: 1")
    print(f"  - Other vehicles: {len(vehicles) - 1}")
    print(f"\nExpected: 30 other vehicles + 1 ego = 31 total")
    print(f"Actual: {len(vehicles)} total vehicles")
    
    if len(vehicles) == 31:
        print(f"\n✅ CORRECT: Environment has 30 other vehicles as configured")
    else:
        print(f"\n⚠️  MISMATCH: Expected 31 total, got {len(vehicles)}")
    
    print(f"\nNote: Camera view only shows ~5-10 nearby vehicles")
    print(f"      (This is normal - observation space is limited)")
    print(f"{'='*70}\n")
    
    env.close()

if __name__ == "__main__":
    main()
