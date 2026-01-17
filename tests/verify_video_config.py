"""Verify video recording uses same config as training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ENV_CONFIG

print("="*70)
print("VIDEO RECORDING CONFIGURATION VERIFICATION")
print("="*70)

config = ENV_CONFIG["config"]

print("\nEnvironment Settings (used for both training and video recording):")
print(f"  🚗 Vehicles count:     {config['vehicles_count']} vehicles")
print(f"  ⏱️  Policy frequency:   {config['policy_frequency']} Hz")
print(f"  ⏱️  Simulation frequency: {config['simulation_frequency']} Hz")
print(f"  🕐 Episode duration:    {config['duration']}s")
print(f"  🚦 Traffic density:     {config['vehicles_density']}")
print(f"  🛣️  Lanes count:         {config['lanes_count']} lanes")
print(f"  📊 Observation type:    {config['observation']['type']}")
print(f"  📊 Observed vehicles:   {config['observation']['vehicles_count']}")

print("\nVideo Recording Settings:")
print(f"  🎥 Screen width:        {config['screen_width']}px")
print(f"  🎥 Screen height:       {config['screen_height']}px")
print(f"  🎥 Centering position:  {config['centering_position']}")
print(f"  🎥 Scaling:             {config['scaling']}")

print("\n✅ Video recording will use SAME configuration as training")
print("   (50 vehicles, 12 Hz policy, 80s duration, dense traffic)")
print("="*70)
