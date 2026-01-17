"""Analyze video durations and diagnose issues."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from src.config import PATHS, ENV_CONFIG

def analyze_video(video_path: Path) -> dict:
    """Analyze a single video file."""
    if not video_path.exists():
        return {"error": "File not found"}
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return {"error": "Cannot open video"}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    duration_seconds = frame_count / fps if fps > 0 else 0
    
    # Calculate simulation steps (at 12 Hz)
    policy_freq = ENV_CONFIG['config']['policy_frequency']
    sim_steps = int(duration_seconds * policy_freq)
    
    cap.release()
    
    return {
        "fps": fps,
        "frames": frame_count,
        "duration_seconds": duration_seconds,
        "sim_steps": sim_steps,
        "width": width,
        "height": height,
        "file_size_mb": video_path.stat().st_size / (1024 * 1024),
    }

def main():
    """Analyze all videos."""
    print("="*80)
    print("VIDEO ANALYSIS")
    print("="*80)
    
    videos_dir = Path(PATHS["assets_videos"])
    
    videos = [
        ("Untrained (0k)", "highway_ppo_0_steps.mp4"),
        ("Half-Trained (100k)", "highway_ppo_100000_steps.mp4"),
        ("Fully-Trained (200k)", "highway_ppo_200000_steps.mp4"),
        ("Evolution (Combined)", "evolution.mp4"),
    ]
    
    print(f"\nEnvironment configuration:")
    policy_freq = ENV_CONFIG['config']['policy_frequency']
    duration = ENV_CONFIG['config']['duration']
    print(f"  Policy frequency: {policy_freq} Hz")
    print(f"  Episode duration: {duration}s")
    print(f"  Max steps per episode: {duration * policy_freq}")
    print()
    
    for label, filename in videos:
        print(f"\n{label}:")
        print(f"  File: {filename}")
        
        video_path = videos_dir / filename
        info = analyze_video(video_path)
        
        if "error" in info:
            print(f"  ❌ {info['error']}")
            continue
        
        print(f"  Resolution: {info['width']}x{info['height']}")
        print(f"  FPS: {info['fps']}")
        print(f"  Frames: {info['frames']}")
        print(f"  Duration: {info['duration_seconds']:.2f} seconds")
        print(f"  Sim steps: {info['sim_steps']} steps")
        print(f"  File size: {info['file_size_mb']:.2f} MB")
        
        # Diagnosis
        if info['duration_seconds'] < 1.0:
            print(f"  ⚠️  WARNING: Video is < 1 second! Agent crashed immediately.")
        elif info['duration_seconds'] < 3.0:
            print(f"  ⚠️  WARNING: Video is < 3 seconds! Agent crashed very quickly.")
        elif info['duration_seconds'] < 10.0:
            print(f"  ⚠️  Short video. Agent survived {info['duration_seconds']:.1f}s.")
        else:
            print(f"  ✅ Good duration ({info['duration_seconds']:.1f}s).")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("\nExpected durations based on training metrics:")
    print("  Untrained (0k):      0.8-2s  (random crashes)")
    print("  Half-Trained (100k): 5.9s    (from training logs)")
    print("  Fully-Trained (200k): 15-20s (predicted)")
    print("\nIf videos are shorter than expected, possible causes:")
    print("  1. Environment mismatch (training vs recording config)")
    print("  2. Observation normalization mismatch")
    print("  3. Agent hasn't learned collision avoidance yet")
    print("  4. Checkpoint loading failure (reverting to random)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
