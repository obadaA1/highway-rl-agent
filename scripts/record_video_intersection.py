"""
Evolution Video Generation Script for Intersection RL Agent.

Purpose:
    Record videos of untrained, half-trained, and fully-trained intersection agents.
    
Outputs:
    - MP4 videos of agent behavior at different training stages
    - Saved to assets/videos/intersection/
    
Compliance:
    - Type hints everywhere
    - Clean video recording
    - Reproducible evaluation

Run: python scripts/record_video_intersection.py
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import cv2

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.intersection_env_v1 import make_intersection_env_v1
from src.agent.ppo_agent import HighwayPPOAgent
from src.intersection_config import INTERSECTION_PATHS, INTERSECTION_CHECKPOINT_CONFIG


def record_episode(
    checkpoint_path: str,
    max_frames: int = 600,
    n_attempts: int = 3,
    select_best: bool = True,
    show_display: bool = True,
):
    """
    Record a single episode from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        max_frames: Maximum frames to record
        n_attempts: Number of attempts to make
        select_best: Whether to select best episode
        show_display: Whether to show video while recording
        
    Returns:
        frames: List of RGB frames
        stats: Episode statistics
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_name = checkpoint_path.name
    checkpoint_stem = checkpoint_path.stem

    print(f"\n📹 Recording: {checkpoint_name}")

    # Check if untrained checkpoint
    is_untrained = (checkpoint_stem == "intersection_ppo_0_steps")

    env = make_intersection_env_v1(render_mode="rgb_array")

    if is_untrained:
        print("   ⚠️ Using untrained (random) policy")
        agent = None
        use_random = True
    else:
        print("   ✅ Loading trained policy...")
        agent = HighwayPPOAgent.load(str(checkpoint_path), env=env)
        use_random = False

    best_frames: List[np.ndarray] = []
    best_stats: Dict = {}
    best_score = -float('inf')

    attempts = 1 if not select_best else n_attempts

    for attempt in range(attempts):
        if attempts > 1:
            print(f"   Attempt {attempt + 1}/{attempts}...")

        obs, _ = env.reset()
        frames: List[np.ndarray] = []
        done = False
        truncated = False
        step = 0
        total_reward = 0.0

        action_counts = {0: 0, 1: 0, 2: 0}
        crashed = False
        goal_reached = False

        # Get initial position if available
        start_pos = None
        if hasattr(env.unwrapped, 'vehicle') and env.unwrapped.vehicle:
            start_pos = np.array(env.unwrapped.vehicle.position[:2])

        while not (done or truncated) and step < max_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

                if show_display:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.putText(
                        bgr,
                        f"Step {step}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow(f"Recording {checkpoint_name}", bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # Get action
            if use_random:
                action = env.action_space.sample()
            else:
                action, _ = agent.model.predict(obs, deterministic=True)
                action = int(action.item()) if isinstance(action, np.ndarray) else int(action)

            if action < 3:
                action_counts[action] += 1

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # Check episode end conditions
            if done or truncated:
                crashed = info.get('crashed', False) or info.get('episode_stats', {}).get('crashes', 0) > 0
                goal_reached = info.get('episode_stats', {}).get('reached_goal', 0) > 0

        # Get final position if available
        end_pos = None
        distance = 0.0
        if hasattr(env.unwrapped, 'vehicle') and env.unwrapped.vehicle and start_pos is not None:
            end_pos = np.array(env.unwrapped.vehicle.position[:2])
            distance = np.linalg.norm(end_pos - start_pos)

        duration = len(frames) / 15.0
        
        # Determine result reason
        if crashed:
            reason = "Crash"
        elif goal_reached:
            reason = "Goal Reached"
        elif truncated:
            reason = "Timeout"
        else:
            reason = "Unknown"

        print(
            f"      Result: {duration:.1f}s | Reward: {total_reward:.0f} "
            f"| Dist: {distance:.0f}m | {reason}"
        )
        print(
            f"      Actions: SLOW={action_counts[0]}  IDLE={action_counts[1]}  "
            f"FAST={action_counts[2]}"
        )

        # Score episode (prefer goal reached, then duration, then reward)
        episode_score = 0.0
        if goal_reached:
            episode_score += 1000.0
        episode_score += duration * 10
        episode_score += total_reward

        if episode_score > best_score:
            best_score = episode_score
            best_frames = frames
            best_stats = {
                "duration": duration,
                "reward": total_reward,
                "distance": distance,
                "goal_reached": goal_reached,
                "crashed": crashed,
                "actions": action_counts,
            }
            if attempts > 1:
                print("      ✅ New Best Episode")

    if show_display:
        cv2.destroyAllWindows()
    env.close()

    return best_frames, best_stats


def save_video(frames: List[np.ndarray], output_path: str, fps: int = 15):
    """
    Save frames as MP4 video.
    
    Args:
        frames: List of RGB frames
        output_path: Output file path
        fps: Frames per second
    """
    if not frames:
        print("⚠️ No frames to save")
        return

    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"💾 Saved {output_path}")


def main():
    """Main video recording function."""
    parser = argparse.ArgumentParser(description="Record Intersection RL Agent Videos")
    parser.add_argument(
        "--model", 
        type=str, 
        help="Specific checkpoint to record"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't show video while recording"
    )
    args = parser.parse_args()

    videos_dir = Path(INTERSECTION_PATHS["assets_videos"])
    videos_dir.mkdir(parents=True, exist_ok=True)

    show_display = not args.no_display

    if args.model:
        # Record specific model
        frames, stats = record_episode(args.model, show_display=show_display)
        output_path = str(videos_dir / f"{Path(args.model).stem}.mp4")
        save_video(frames, output_path)
        return

    # Record evolution videos (0, 100k, 200k)
    checkpoints_dir = Path(INTERSECTION_CHECKPOINT_CONFIG["save_path"])
    targets = [
        "intersection_ppo_0_steps.zip",
        "intersection_ppo_100000_steps.zip",
        "intersection_ppo_200000_steps.zip",
    ]

    print("\n" + "="*70)
    print("RECORDING INTERSECTION EVOLUTION VIDEOS")
    print("="*70)

    for fname in targets:
        fpath = checkpoints_dir / fname
        if fpath.exists():
            # For untrained agent (0 steps), use first attempt only (no best-of-3)
            # to show truly random behavior
            is_untrained = "0_steps" in fname
            if is_untrained:
                frames, stats = record_episode(str(fpath), n_attempts=1, select_best=False, show_display=show_display)
            else:
                frames, stats = record_episode(str(fpath), show_display=show_display)
            output_path = str(videos_dir / f"{fpath.stem}.mp4")
            save_video(frames, output_path)
        else:
            print(f"⚠️ Missing checkpoint: {fname}")

    print("\n" + "="*70)
    print("✅ VIDEO RECORDING COMPLETE")
    print("="*70)
    print(f"\nVideos saved to: {videos_dir}")
    print("\nNext steps:")
    print("  1. Convert to GIF: python scripts/convert_to_gif_intersection.py")
    print("  2. Update README.md with video links")


if __name__ == "__main__":
    main()
