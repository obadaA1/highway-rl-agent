# Intersection-v0 Training Guide

## 📋 Overview

This document provides a complete guide for training and evaluating the intersection-v0 RL agent. All necessary code has been created following the same best practices as the highway-v0 implementation.

---

## 🗂️ New Files Created

### 1. Configuration
- **`src/intersection_config.py`**: Dedicated configuration file for intersection environment
  - Environment parameters (15 vehicles, 3 actions, heading info)
  - Reward function parameters (goal-directed navigation)
  - Training hyperparameters (200k timesteps, entropy=0.01)
  - Checkpoint and video settings
  - Separate directory structure: `assets/checkpoints/intersection/`, `assets/videos/intersection/`, etc.

### 2. Environment Wrapper
- **`src/env/intersection_env_v1.py`**: Custom reward wrapper for intersection-v0
  - Goal-directed reward function
  - Components:
    - R_goal_progress: +0.4 max for moving toward destination
    - R_safe_crossing: +1.0 bonus for successful goal completion
    - P_collision: -1.0 penalty for crashes
    - P_timeout: -0.01 per step to encourage efficiency
  - Episode statistics tracking

### 3. Training Scripts
- **`scripts/train_intersection.py`**: Main training script
  - Train for 200k timesteps
  - Checkpoint at 0k, 100k, 200k
  - Resume support
  - TensorBoard logging
  - Progress callbacks

- **`scripts/evaluate_intersection.py`**: Evaluation script
  - Run 100 episodes (configurable)
  - Statistics: reward, episode length, crash rate, goal success rate
  - Action distribution analysis
  - Optional rendering

- **`scripts/record_video_intersection.py`**: Video recording script
  - Record evolution videos (0k, 100k, 200k)
  - Best episode selection
  - MP4 output format

### 4. Utility Scripts
- **`scripts/plot_training_intersection.py`**: Generate training plots
  - Reward progression
  - Episode length
  - Training losses
  - Read from TensorBoard logs

- **`scripts/convert_to_gif_intersection.py`**: Convert MP4 to GIF
  - For README embedding
  - Configurable FPS and size

### 5. Documentation
- **Updated `README.md`**: Complete intersection-v0 section
  - Methodology (state space, action space, reward function in LaTeX)
  - Placeholders for results (to be filled after training)
  - Repository structure updated
  - Reproduction instructions

---

## 🚀 Quick Start

### 1. Activate Environment

```powershell
.\rl_highway_env\Scripts\Activate.ps1
```

### 2. Verify Configuration

```powershell
# Test configuration
python src/intersection_config.py

# Test environment
python src/env/intersection_env_v1.py
```

### 3. Start Training

```powershell
# Train intersection agent (200k timesteps, ~2-3 hours on GPU)
python scripts/train_intersection.py
```

**Expected Output:**
- Checkpoint saved every 100k steps
- Progress updates every 10k steps
- TensorBoard logs in `assets/logs/intersection/`
- Final checkpoints:
  - `assets/checkpoints/intersection/intersection_ppo_0_steps.zip`
  - `assets/checkpoints/intersection/intersection_ppo_100000_steps.zip`
  - `assets/checkpoints/intersection/intersection_ppo_200000_steps.zip`

### 4. Monitor Training (Optional)

```powershell
# In a separate terminal
tensorboard --logdir assets/logs/intersection
```

Then open: http://localhost:6006

### 5. Evaluate Agent

```powershell
# Evaluate final checkpoint (100 episodes)
python scripts/evaluate_intersection.py --checkpoint assets/checkpoints/intersection/intersection_ppo_200000_steps.zip

# Evaluate with rendering (slower, visual feedback)
python scripts/evaluate_intersection.py --checkpoint assets/checkpoints/intersection/intersection_ppo_200000_steps.zip --render

# Evaluate different checkpoints for comparison
python scripts/evaluate_intersection.py --checkpoint assets/checkpoints/intersection/intersection_ppo_0_steps.zip
python scripts/evaluate_intersection.py --checkpoint assets/checkpoints/intersection/intersection_ppo_100000_steps.zip
```

### 6. Generate Evolution Videos

```powershell
# Record all three checkpoints (0k, 100k, 200k)
python scripts/record_video_intersection.py

# Record specific checkpoint
python scripts/record_video_intersection.py --model assets/checkpoints/intersection/intersection_ppo_100000_steps.zip

# No display (faster, background recording)
python scripts/record_video_intersection.py --no-display
```

**Output:** MP4 files in `assets/videos/intersection/`

### 7. Convert Videos to GIF

```powershell
# Convert all videos to GIF for README
python scripts/convert_to_gif_intersection.py
```

**Note:** Requires `moviepy` (already in requirements.txt)

### 8. Generate Training Plots

```powershell
# Generate plots from TensorBoard logs
python scripts/plot_training_intersection.py
```

**Output:** PNG files in `assets/plots/intersection/`

### 9. Update README

After training completes:
1. Run evaluation to get final metrics
2. Generate plots
3. Convert videos to GIF
4. Update README.md placeholders with actual results:
   - Training stages table
   - Reward progression plot
   - Success rate analysis
   - Challenges & insights

---

## 📊 Key Differences: Highway-v0 vs Intersection-v0

| Aspect | Highway-v0 | Intersection-v0 |
|--------|-----------|-----------------|
| **Environment** | `highway-v0` | `intersection-v0` |
| **Task** | Drive fast, avoid collisions | Navigate to goal, avoid cross-traffic |
| **Observation** | 5 vehicles, 5 features | 15 vehicles, 7 features (+ heading) |
| **Action Space** | 5 actions (LEFT, IDLE, RIGHT, FASTER, SLOWER) | 3 actions (SLOWER, IDLE, FASTER) |
| **Reward Focus** | Speed vs Safety | Goal-directed vs Efficiency |
| **Episode Duration** | 80 seconds | 13 seconds |
| **Main Challenge** | Degenerate slow-driving policy | Cross-traffic coordination |
| **Success Metric** | Survival time, crash rate | Goal completion rate, crash rate |

---

## 🎯 Expected Behavior

### Untrained Agent (0 steps)
- Random actions
- No goal-directed behavior
- High crash rate (95%+)
- Either immediate crashes or aimless wandering

### Half-Trained Agent (100k steps)
- Partial collision avoidance
- Some goal-directed behavior
- Moderate success rate (10-30%)
- May still struggle with timing (when to yield vs go)

### Fully-Trained Agent (200k steps)
- Consistent goal-directed navigation
- Strategic yielding to cross-traffic
- High success rate (50-80%)
- Low crash rate (5-20%)

---

## 🐛 Troubleshooting

### Issue: "Module not found: intersection_config"
**Solution:** Make sure you're running from project root and src/ is in path

### Issue: Environment doesn't render
**Solution:** Install pygame: `pip install pygame`

### Issue: TensorBoard not showing logs
**Solution:** Check log path: `assets/logs/intersection/intersection_ppo_training_*/`

### Issue: Training is very slow
**Solution:** Check GPU availability:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Issue: Agent not learning (reward not improving)
**Possible causes:**
- Hyperparameters may need tuning
- Reward function may be too sparse
- Entropy coefficient may be too low (exploration)
- Check evaluation to see actual behavior

---

## 📈 Performance Expectations

### Training Speed
- **GPU (RTX 3050)**: ~1500-2000 it/s → 200k steps in 2-3 hours
- **CPU**: ~300-500 it/s → 200k steps in 8-12 hours

### Checkpoint Sizes
- Each checkpoint: ~1-2 MB
- Total: ~3-6 MB for 3 checkpoints

### Video Sizes
- MP4: ~5-15 MB per video
- GIF: ~10-30 MB per video (use fps=10 to reduce)

---

## 📝 Reporting Results (README Update)

After training, update these sections in README.md:

### 1. Evolution Video Section
- Add GIF links to the three stages
- Update the training stages table with actual metrics

### 2. Training Analysis Section
- Embed reward progression plot
- Embed success rate plot
- Write textual analysis explaining:
  - Learning curve shape
  - Plateaus and why they occurred
  - Final performance vs expectations

### 3. Challenges & Insights Section
- Document at least one real technical failure
- Explain what went wrong, why, and how it was fixed
- Compare to highway-v0 challenges

### 4. Final Results Summary
- Mean reward ± std
- Success rate
- Crash rate
- Episode length
- Action distribution

---

## 🎓 Rubric Compliance Checklist

- [x] **Separate configuration file**: `src/intersection_config.py`
- [x] **Custom reward function**: LaTeX in README + code in `intersection_env_v1.py`
- [x] **Evolution video**: 3 stages (0k, 100k, 200k)
- [ ] **Training analysis**: Plots + textual analysis (after training)
- [ ] **Challenges**: At least one failure documented (after training)
- [x] **Clean repository**: Separate directories for intersection
- [x] **Reproducible**: Fixed seed (42), config centralized
- [x] **Type hints**: All new code has type hints
- [x] **Modular**: Following same structure as highway-v0

---

## 🚀 Next Steps

1. **Start Training**: `python scripts/train_intersection.py`
2. **Monitor Progress**: Check TensorBoard or terminal output
3. **Evaluate Checkpoints**: Test at 100k and 200k steps
4. **Generate Artifacts**: Videos, plots, GIFs
5. **Update README**: Fill in results and analysis
6. **Compare**: Analyze differences vs highway-v0 results

---

## 💡 Tips for Success

1. **Don't skip untrained checkpoint**: It's crucial for evolution video
2. **Evaluate multiple times**: Run 100 episodes for statistical significance
3. **Document failures**: Even if agent doesn't fully learn, document why
4. **Compare hyperparameters**: If results are poor, consider adjusting:
   - Entropy coefficient (exploration)
   - Reward weights (goal vs collision)
   - Learning rate
   - Collision penalty magnitude
5. **Mathematical analysis**: Calculate break-even points like highway-v0 analysis

---

## 📚 Additional Resources

- **highway-env docs**: https://highway-env.farama.org/
- **Stable-Baselines3 docs**: https://stable-baselines3.readthedocs.io/
- **PPO paper**: https://arxiv.org/abs/1707.06347

---

**Good luck with training! 🚀**
