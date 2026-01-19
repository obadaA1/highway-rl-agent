"""
Evolution Video Generation Script (Correct Final Version).
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import cv2

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.highway_env_v6 import make_highway_env_v6
from src.agent.ppo_agent import HighwayPPOAgent
from src.config import PATHS


def record_episode(
    checkpoint_path: str,
    max_frames: int = 1200,
    n_attempts: int = 3,
    select_best: bool = True,
    show_display: bool = True,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_name = checkpoint_path.name
    checkpoint_stem = checkpoint_path.stem

    print(f"\n📹 Recording: {checkpoint_name}")

    # ✅ EXACT untrained detection
    is_untrained = (checkpoint_stem == "highway_ppo_0_steps")

    env = make_highway_env_v6(render_mode="rgb_array")

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
    best_duration = -1.0

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

        action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        lane_changes = 0
        last_action = None

        start_x = env.unwrapped.vehicle.position[0]

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

            # ---- ACTION ----
            if use_random:
                action = env.action_space.sample()
            else:
                action, _ = agent.model.predict(obs, deterministic=True)
                action = int(action.item()) if isinstance(action, np.ndarray) else int(action)

            action_counts[action] += 1

            # Correct lane-change detection
            if action in [0, 2] and last_action in [0, 2] and action != last_action:
                lane_changes += 1
            last_action = action

            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            step += 1

        end_x = env.unwrapped.vehicle.position[0]
        distance = end_x - start_x
        duration = len(frames) / 15.0
        reason = "Crash" if done else "Timeout"

        print(
            f"      Result: {duration:.1f}s | Reward: {total_reward:.0f} "
            f"| Dist: {distance:.0f}m | {reason}"
        )
        print(
            f"      Actions: LEFT={action_counts[0]}  IDLE={action_counts[1]}  "
            f"RIGHT={action_counts[2]}  FAST={action_counts[3]}  SLOW={action_counts[4]}"
        )

        if duration > best_duration:
            best_duration = duration
            best_frames = frames
            best_stats = {
                "duration": duration,
                "reward": total_reward,
                "distance": distance,
                "lane_changes": lane_changes,
                "actions": action_counts,
            }
            if attempts > 1:
                print("      ✅ New Best Episode")

    if show_display:
        cv2.destroyAllWindows()
    env.close()

    return best_frames, best_stats


def save_video(frames: List[np.ndarray], output_path: str, fps: int = 15):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Specific checkpoint to record")
    args = parser.parse_args()

    videos_dir = Path(PATHS["assets_videos"])
    videos_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        frames, _ = record_episode(args.model)
        save_video(frames, str(videos_dir / f"{Path(args.model).stem}.mp4"))
        return

    checkpoints_dir = Path(PATHS["checkpoints"])
    targets = [
        "highway_ppo_0_steps.zip",
        "highway_ppo_100000_steps.zip",
        "highway_ppo_200000_steps.zip",
    ]

    for fname in targets:
        fpath = checkpoints_dir / fname
        if fpath.exists():
            frames, _ = record_episode(str(fpath))
            save_video(frames, str(videos_dir / f"{fpath.stem}.mp4"))
        else:
            print(f"⚠️ Missing checkpoint: {fname}")


if __name__ == "__main__":
    main()
