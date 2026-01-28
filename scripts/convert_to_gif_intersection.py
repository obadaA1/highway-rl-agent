"""
Convert Intersection MP4 videos to GIFs for README embedding.

Purpose:
    Convert evolution videos to GIF format for GitHub README.
    
Outputs:
    - GIF files from MP4 videos
    - Saved to assets/videos/intersection/
    
Run: python scripts/convert_to_gif_intersection.py
"""

import sys
from pathlib import Path
from typing import List
import imageio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.intersection_config import INTERSECTION_PATHS


def convert_video_to_gif(
    video_path: str,
    gif_path: str,
    fps: int = 10,
    scale: float = 0.5,
    max_frames: int = 300
) -> None:
    """
    Convert MP4 video to optimized GIF.
    
    Args:
        video_path: Path to input MP4 file
        gif_path: Path to output GIF file
        fps: Frame rate for GIF (lower = smaller file)
        scale: Scale factor (0.5 = 50% size, smaller = smaller file)
        max_frames: Maximum frames to include (prevents huge GIFs)
    """
    print(f"\n🎬 Converting: {Path(video_path).name}")
    print(f"   Settings: {fps} fps, {int(scale*100)}% scale, max {max_frames} frames")
    
    # Read video
    reader = imageio.get_reader(video_path)
    video_fps = reader.get_meta_data()['fps']
    
    # Calculate frame skip to achieve desired fps
    frame_skip = max(1, int(video_fps / fps))
    
    frames = []
    frame_count = 0
    
    for i, frame in enumerate(reader):
        if i % frame_skip == 0:
            # Resize frame
            import cv2
            if scale != 1.0:
                new_width = int(frame.shape[1] * scale)
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            frames.append(frame)
            frame_count += 1
            
            if frame_count >= max_frames:
                break
    
    reader.close()
    
    print(f"   Extracted {len(frames)} frames")
    
    # Write GIF
    imageio.mimsave(
        gif_path,
        frames,
        fps=fps,
        loop=0  # Infinite loop
    )
    
    # Check file size
    file_size_mb = Path(gif_path).stat().st_size / (1024 * 1024)
    print(f"   ✅ Saved: {Path(gif_path).name} ({file_size_mb:.2f} MB)")
    
    if file_size_mb > 10:
        print(f"   ⚠️  Warning: GIF is large ({file_size_mb:.2f} MB)")
        print(f"      Consider reducing fps, scale, or max_frames")


def main():
    """Main conversion function."""
    print("\n" + "="*70)
    print("INTERSECTION VIDEO TO GIF CONVERSION")
    print("="*70)
    
    videos_dir = Path(INTERSECTION_PATHS["assets_videos"])
    
    # Target videos for conversion
    targets = [
        "intersection_ppo_0_steps.mp4",
        "intersection_ppo_100000_steps.mp4",
        "intersection_ppo_200000_steps.mp4",
    ]
    
    for video_name in targets:
        mp4_path = videos_dir / video_name
        gif_path = videos_dir / video_name.replace(".mp4", ".gif")
        
        if not mp4_path.exists():
            print(f"\n⚠️ Video not found: {video_name}")
            print(f"   Run: python scripts/record_video_intersection.py")
            continue
        
        if gif_path.exists():
            print(f"\n⚠️ {gif_path.name} already exists. Skipping...")
            continue
        
        # Convert with optimized settings for GitHub
        convert_video_to_gif(
            str(mp4_path),
            str(gif_path),
            fps=10,
            scale=0.5,
            max_frames=200
        )
    
    print("\n" + "="*70)
    print("✅ CONVERSION COMPLETE")
    print("="*70)
    print(f"\nGIFs saved to: {videos_dir}")
    print("\nNext steps:")
    print("  1. Check GIF file sizes (should be < 10MB for GitHub)")
    print("  2. If too large, reduce fps or resize_factor in this script")
    print("  3. Update README.md with GIF links")


if __name__ == "__main__":
    main()
