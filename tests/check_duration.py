import sys
sys.path.insert(0, '.')
from src.env.highway_env import make_highway_env

env = make_highway_env(render_mode=None)
base_env = env.unwrapped

print(f'Duration: {base_env.config["duration"]}s')
print(f'Policy freq: {base_env.config["policy_frequency"]} Hz')
print(f'Expected max steps: {base_env.config["duration"]} * {base_env.config["policy_frequency"]} = {base_env.config["duration"] * base_env.config["policy_frequency"]} steps')
env.close()
