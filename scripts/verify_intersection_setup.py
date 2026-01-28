"""
Verification Script for Intersection-v0 Setup.

Purpose:
    Verify that all intersection-v0 components are properly configured.
    
Run: python scripts/verify_intersection_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_imports() -> bool:
    """Verify all required imports."""
    print("\n" + "="*70)
    print("VERIFYING IMPORTS")
    print("="*70)
    
    required_imports = [
        ("gymnasium", "Gymnasium"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
    ]
    
    all_good = True
    for module, name in required_imports:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            all_good = False
    
    return all_good


def verify_config() -> bool:
    """Verify configuration files."""
    print("\n" + "="*70)
    print("VERIFYING CONFIGURATION")
    print("="*70)
    
    try:
        from src.intersection_config import (
            INTERSECTION_ENV_CONFIG,
            INTERSECTION_REWARD_CONFIG,
            INTERSECTION_TRAINING_CONFIG,
            INTERSECTION_CHECKPOINT_CONFIG,
            INTERSECTION_PATHS,
        )
        print("✅ intersection_config.py loaded successfully")
        
        # Print key settings
        print(f"\n   Environment: {INTERSECTION_ENV_CONFIG['id']}")
        print(f"   Total Timesteps: {INTERSECTION_TRAINING_CONFIG['total_timesteps']:,}")
        print(f"   Learning Rate: {INTERSECTION_TRAINING_CONFIG['learning_rate']}")
        print(f"   Entropy Coefficient: {INTERSECTION_TRAINING_CONFIG['ent_coef']}")
        print(f"   Seed: {INTERSECTION_TRAINING_CONFIG['seed']}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading intersection_config.py: {e}")
        return False


def verify_environment() -> bool:
    """Verify environment wrapper."""
    print("\n" + "="*70)
    print("VERIFYING ENVIRONMENT")
    print("="*70)
    
    try:
        from src.env.intersection_env_v1 import make_intersection_env_v1
        
        env = make_intersection_env_v1(render_mode=None)
        print("✅ Environment created successfully")
        
        print(f"\n   Observation Space: {env.observation_space}")
        print(f"   Action Space: {env.action_space}")
        
        # Test reset and step
        obs, info = env.reset(seed=42)
        print(f"   Observation Shape: {obs.shape}")
        
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"   Step executed successfully")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Error with environment: {e}")
        return False


def verify_agent() -> bool:
    """Verify agent can be created."""
    print("\n" + "="*70)
    print("VERIFYING AGENT")
    print("="*70)
    
    try:
        from src.env.intersection_env_v1 import make_intersection_env_v1
        from src.agent.ppo_agent import HighwayPPOAgent
        from src.intersection_config import INTERSECTION_TRAINING_CONFIG
        
        env = make_intersection_env_v1(render_mode=None)
        
        agent = HighwayPPOAgent(
            env=env,
            learning_rate=INTERSECTION_TRAINING_CONFIG["learning_rate"],
            verbose=0,
        )
        print("✅ Agent created successfully")
        
        # Test predict
        obs, _ = env.reset(seed=42)
        action, _ = agent.model.predict(obs)
        print(f"   Prediction executed successfully")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Error with agent: {e}")
        return False


def verify_directories() -> bool:
    """Verify directory structure."""
    print("\n" + "="*70)
    print("VERIFYING DIRECTORIES")
    print("="*70)
    
    from src.intersection_config import INTERSECTION_PATHS
    
    all_good = True
    for name, path in INTERSECTION_PATHS.items():
        path_obj = Path(path)
        if path_obj.exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"⚠️ {name}: {path} (will be created)")
            path_obj.mkdir(parents=True, exist_ok=True)
    
    return all_good


def verify_scripts() -> bool:
    """Verify scripts exist."""
    print("\n" + "="*70)
    print("VERIFYING SCRIPTS")
    print("="*70)
    
    scripts = [
        "scripts/train_intersection.py",
        "scripts/evaluate_intersection.py",
        "scripts/record_video_intersection.py",
        "scripts/plot_training_intersection.py",
        "scripts/convert_to_gif_intersection.py",
    ]
    
    all_good = True
    for script in scripts:
        script_path = Path(__file__).parent.parent / script
        if script_path.exists():
            print(f"✅ {script}")
        else:
            print(f"❌ {script} - NOT FOUND")
            all_good = False
    
    return all_good


def verify_gpu() -> None:
    """Check GPU availability."""
    print("\n" + "="*70)
    print("GPU INFORMATION")
    print("="*70)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️ No GPU available - training will use CPU (slower)")
    except ImportError:
        print("⚠️ PyTorch not installed")


def main():
    """Run all verification checks."""
    print("\n" + "="*70)
    print("INTERSECTION-V0 SETUP VERIFICATION")
    print("="*70)
    
    checks = [
        ("Imports", verify_imports),
        ("Configuration", verify_config),
        ("Environment", verify_environment),
        ("Agent", verify_agent),
        ("Directories", verify_directories),
        ("Scripts", verify_scripts),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name} check failed with error: {e}")
            results[name] = False
    
    # GPU info (not a pass/fail check)
    verify_gpu()
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    print("\n" + "="*70)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("\nYou're ready to start training:")
        print("  python scripts/train_intersection.py")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
