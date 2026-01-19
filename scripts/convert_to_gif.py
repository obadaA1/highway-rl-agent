"""
Convert MP4 videos to GIF format for README embedding.

Purpose:
    Convert evolution videos from MP4 to GIF for direct GitHub embedding
    GIFs display inline in README, while MP4s require clicking links

Run: python scripts/convert_to_gif.py
"""

import sys
from pathlib import Path
from typing import List
import imageio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PATHS


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
    """
    Convert all evolution videos to GIF format.
    """
    print("\n" + "="*70)
    print("CONVERTING VIDEOS TO GIF FORMAT")
    print("="*70)
    
    videos_dir = Path(PATHS["assets_videos"])
    
    # Videos to convert
    video_files = [
        "highway_ppo_0_steps.mp4",
        "highway_ppo_100000_steps.mp4",
        "highway_ppo_200000_steps.mp4",
    ]
    
    for video_file in video_files:
        video_path = videos_dir / video_file
        
        if not video_path.exists():
            print(f"\n⚠️  Video not found: {video_file}")
            continue
        
        # Output GIF path
        gif_path = videos_dir / video_file.replace('.mp4', '.gif')
        
        # Convert with optimized settings for GitHub
        # - 10 fps (smooth enough, smaller file)
        # - 50% scale (GitHub thumbnails are small anyway)
        # - 200 frames max (~20 seconds at 10 fps)
        convert_video_to_gif(
            str(video_path),
            str(gif_path),
            fps=10,
            scale=0.5,
            max_frames=200
        )
    
    print("\n" + "="*70)
    print("✅ CONVERSION COMPLETE")
    print("="*70)
    print(f"\nGIF files saved to: {videos_dir}")
    print("\nEmbed in README with:")
    print("  ![Untrained Agent](assets/videos/highway_ppo_0_steps.gif)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
