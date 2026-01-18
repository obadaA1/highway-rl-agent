# V4 vs V5 Reward Function Comparison

> **Theoretical Analysis & Empirical Comparison**

This document provides a detailed comparison between the V4 (Acceleration-Aware) and V5 (Rubric-Compliant) reward function implementations for the Highway-RL agent.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Reward Component Comparison](#reward-component-comparison)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Theoretical Analysis](#theoretical-analysis)
5. [Rubric Compliance Matrix](#rubric-compliance-matrix)
6. [Expected Behavior Differences](#expected-behavior-differences)
7. [Empirical Results](#empirical-results-placeholder)
8. [Conclusion](#conclusion)

---

## Executive Summary

| Aspect | V4 (Acceleration-Aware) | V5 (Rubric-Compliant) |
|--------|-------------------------|------------------------|
| **Components** | 6 | 8 |
| **Focus** | Speed + Acceleration | Speed + Safety + Efficiency |
| **Complexity** | Medium | Medium-High |
| **Rubric Compliance** | Partial (4/5) | Full (5/5) |
| **Training Time** | ~90 min (200k steps) | ~90 min (200k steps) |
| **Key Strength** | Simple, fast learning | Balanced, comprehensive |

---

## Reward Component Comparison

### Component Overview Table

| # | Component | V4 Value | V5 Value | Purpose |
|---|-----------|----------|----------|---------|
| 1 | `r_progress` | v + 0.2Δv | v + 0.2Δv | Forward progress (speed + acceleration) |
| 2 | `r_alive` | +0.01 | +0.01 | Survival incentive |
| 3 | `r_collision` | -100.0 | -100.0 | Crash penalty |
| 4 | `r_slow_action` | -0.05/-0.01 | -0.05/-0.01 | Context-dependent SLOWER penalty |
| 5 | `r_low_speed` | -0.02 | -0.02 | Low speed penalty (< 60%) |
| 6 | `r_faster_bonus` | +0.05 | +0.05 | FASTER action bonus when slow |
| 7 | `r_headway` | ❌ N/A | +0.10/-0.10 | **Safe distance reward (V5 NEW)** |
| 8 | `r_lane` | ❌ N/A | -0.02 | **Lane change penalty (V5 NEW)** |

### Visual Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    V4: 6-Component Reward                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐ ┌──────────┐ ┌────────────────┐              │
│  │ r_progress   │ │ r_alive  │ │ r_collision    │              │
│  │ v + 0.2Δv    │ │ +0.01    │ │ -100.0         │              │
│  └──────────────┘ └──────────┘ └────────────────┘              │
│  ┌──────────────┐ ┌────────────┐ ┌──────────────┐              │
│  │ r_slow_action│ │ r_low_speed│ │ r_faster     │              │
│  │ -0.05/-0.01  │ │ -0.02      │ │ +0.05        │              │
│  └──────────────┘ └────────────┘ └──────────────┘              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    V5: 8-Component Reward                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐ ┌──────────┐ ┌────────────────┐              │
│  │ r_progress   │ │ r_alive  │ │ r_collision    │              │
│  │ v + 0.2Δv    │ │ +0.01    │ │ -100.0         │              │
│  └──────────────┘ └──────────┘ └────────────────┘              │
│  ┌──────────────┐ ┌────────────┐ ┌──────────────┐              │
│  │ r_slow_action│ │ r_low_speed│ │ r_faster     │              │
│  │ -0.05/-0.01  │ │ -0.02      │ │ +0.05        │              │
│  └──────────────┘ └────────────┘ └──────────────┘              │
│  ╔══════════════╗ ╔══════════════╗  ← V5 ADDITIONS             │
│  ║ r_headway    ║ ║ r_lane       ║                              │
│  ║ +0.10/-0.10  ║ ║ -0.02        ║                              │
│  ╚══════════════╝ ╚══════════════╝                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Formulation

### V4 Reward Function

$$R_{V4}(s, a) = r_{progress} + r_{alive} + r_{collision} + r_{slow} + r_{low} + r_{faster}$$

Where:

$$r_{progress} = \frac{v_{ego}}{v_{max}} + 0.2 \cdot \Delta v$$

$$\Delta v = \text{clip}\left(\frac{v_t - v_{t-1}}{v_{max}}, -0.05, 0.1\right)$$

$$r_{alive} = 0.01$$

$$r_{collision} = \begin{cases} -100.0 & \text{if crashed} \\ 0 & \text{otherwise} \end{cases}$$

$$r_{slow\_action} = \begin{cases} -0.05 & \text{if } a = \text{SLOWER} \land v < 0.7 \cdot v_{max} \\ -0.01 & \text{if } a = \text{SLOWER} \land v \geq 0.7 \cdot v_{max} \\ 0 & \text{otherwise} \end{cases}$$

$$r_{low\_speed} = \begin{cases} -0.02 & \text{if } v < 0.6 \cdot v_{max} \\ 0 & \text{otherwise} \end{cases}$$

$$r_{faster\_bonus} = \begin{cases} +0.05 & \text{if } a = \text{FASTER} \land v < 0.8 \cdot v_{max} \\ 0 & \text{otherwise} \end{cases}$$

---

### V5 Reward Function (Rubric-Compliant)

$$R_{V5}(s, a) = R_{V4}(s, a) + r_{headway} + r_{lane}$$

**New Components:**

$$r_{headway} = \begin{cases} +0.10 & \text{if } \tau \geq \tau_{safe} = 1.5s \\ -0.10 & \text{if } \tau < \tau_{danger} = 0.5s \\ 0 & \text{otherwise} \end{cases}$$

Where time-headway $\tau$ is:

$$\tau = \frac{d_{front}}{v_{ego}}$$

And $d_{front}$ is the distance to the closest vehicle ahead.

$$r_{lane} = \begin{cases} -0.02 & \text{if } a \in \{\text{LANE\_LEFT}, \text{LANE\_RIGHT}\} \\ 0 & \text{otherwise} \end{cases}$$

---

## Theoretical Analysis

### V4: Acceleration-Focused Philosophy

**Core Idea:** Reward forward progress with an acceleration bonus to encourage active speed management.

**Strengths:**
1. **Simplicity:** 6 components are easier to debug and tune
2. **Clear Gradient:** Acceleration bonus provides clear signal for "speed up when slow"
3. **Fast Convergence:** Fewer objectives = simpler optimization landscape
4. **CPU-Friendly:** Less computation per step

**Weaknesses:**
1. **No Distance Awareness:** Agent may tailgate without penalty
2. **No Lane Discipline:** Agent may zig-zag excessively
3. **Rubric Incomplete:** Missing 2/5 rubric requirements

**Expected Behavior:**
- Aggressive driving style
- Frequent acceleration/deceleration cycles
- Potential tailgating at high speeds
- High lane change frequency

---

### V5: Balanced Multi-Objective Philosophy

**Core Idea:** Comprehensive reward addressing all rubric requirements with balanced trade-offs.

**Strengths:**
1. **Full Rubric Compliance:** All 5 requirements explicitly addressed
2. **Safer Driving:** Headway reward encourages safe following distance
3. **Smoother Trajectories:** Lane penalty reduces unnecessary maneuvers
4. **Better Generalization:** More realistic driving constraints

**Weaknesses:**
1. **Increased Complexity:** 8 components may slow learning
2. **Potential Conflicts:** Safe distance vs. speed objectives may conflict
3. **Tuning Difficulty:** More hyperparameters to balance

**Expected Behavior:**
- More cautious at high speeds (maintains distance)
- Strategic lane changes (only when beneficial)
- Balanced speed vs. safety trade-off
- Lower crash rate, potentially slower average speed

---

## Rubric Compliance Matrix

The project rubric requires the reward function to:

| Requirement | V4 Status | V5 Status | Implementation |
|-------------|-----------|-----------|----------------|
| **"Reward high forward velocity"** | ✅ Full | ✅ Full | `r_progress = v/v_max + 0.2Δv` |
| **"Penalize collisions"** | ✅ Full | ✅ Full | `r_collision = -100.0` |
| **"Penalize driving too slowly"** | ✅ Full | ✅ Full | `r_low_speed = -0.02`, `r_slow_action` |
| **"Maintaining safe distances"** | ❌ Missing | ✅ Full | `r_headway = ±0.10` based on τ |
| **"Penalize unnecessary lane changes"** | ❌ Missing | ✅ Full | `r_lane = -0.02` per change |

**Compliance Score:**
- **V4:** 3/5 (60%) - Missing safe distance and lane change requirements
- **V5:** 5/5 (100%) - Full rubric compliance

---

## Expected Behavior Differences

### Scenario Analysis

#### Scenario 1: Approaching Slower Vehicle

| Aspect | V4 Behavior | V5 Behavior |
|--------|-------------|-------------|
| Following Distance | Close (no penalty) | Maintains 1.5s headway |
| Lane Change Decision | Quick (no cost) | Deliberate (weighs -0.02) |
| Risk Level | Higher | Lower |

#### Scenario 2: Dense Traffic

| Aspect | V4 Behavior | V5 Behavior |
|--------|-------------|-------------|
| Lane Changes/Min | High (unrestricted) | Moderate (penalized) |
| Average Speed | Higher (aggressive) | Balanced (safe + fast) |
| Crash Probability | Higher | Lower |

#### Scenario 3: Open Highway

| Aspect | V4 Behavior | V5 Behavior |
|--------|-------------|-------------|
| Acceleration | Aggressive | Similar |
| Headway Bonus | None | +0.10/step |
| Expected Reward | ~0.9-1.0/step | ~1.0-1.1/step |

### Reward Budget Comparison

Typical step reward breakdown:

```
V4 (Cruising at 80% speed):
  r_progress:    0.80
  r_alive:       0.01
  r_collision:   0.00
  r_slow_action: 0.00
  r_low_speed:   0.00
  r_faster:      0.00
  ─────────────────────
  TOTAL:         0.81

V5 (Cruising at 80% speed, safe distance):
  r_progress:    0.80
  r_alive:       0.01
  r_collision:   0.00
  r_slow_action: 0.00
  r_low_speed:   0.00
  r_faster:      0.00
  r_headway:     0.10  ← V5 bonus
  r_lane:        0.00
  ─────────────────────
  TOTAL:         0.91  (+12% vs V4)
```

---

## Empirical Results (Placeholder)

> **⚠️ This section will be populated after training both V4 and V5 agents.**

### Training Metrics Comparison

| Metric | V4 (200k steps) | V5 (200k steps) |
|--------|-----------------|-----------------|
| Final Mean Reward | *TBD* | *TBD* |
| Crash Rate (100 ep) | *TBD* | *TBD* |
| Avg Episode Length | *TBD* | *TBD* |
| Avg Velocity (m/s) | *TBD* | *TBD* |
| Lane Changes/Episode | *TBD* | *TBD* |
| Headway Violations | N/A | *TBD* |
| Training Time | *TBD* | *TBD* |

### Learning Curves

*[Placeholder for side-by-side training curves]*

```
V4 Reward vs Episodes         V5 Reward vs Episodes
        │                             │
    100 ┤                         100 ┤
        │                             │
     50 ┤                          50 ┤
        │                             │
      0 ┼───────────────            0 ┼───────────────
        0     100k    200k            0     100k    200k
```

### Action Distribution Comparison

*[Placeholder for action distribution pie charts]*

| Action | V4 (%) | V5 (%) |
|--------|--------|--------|
| LANE_LEFT | *TBD* | *TBD* |
| IDLE | *TBD* | *TBD* |
| LANE_RIGHT | *TBD* | *TBD* |
| FASTER | *TBD* | *TBD* |
| SLOWER | *TBD* | *TBD* |

### Qualitative Behavior Analysis

*[Placeholder for video analysis notes]*

**V4 Agent Observations:**
- *TBD after training*

**V5 Agent Observations:**
- *TBD after training*

---

## Conclusion

### Summary

| Criterion | Winner | Reason |
|-----------|--------|--------|
| **Rubric Compliance** | V5 | Full 5/5 vs V4's 3/5 |
| **Training Simplicity** | V4 | Fewer components, faster tuning |
| **Expected Safety** | V5 | Explicit safe distance reward |
| **Expected Speed** | V4* | No lane penalty overhead |
| **Academic Rigor** | V5 | Addresses all requirements |

*\*Pending empirical verification*

### Recommendation

For **rubric compliance** and **academic grading**, **V5 is recommended** as it explicitly addresses all 5 reward function requirements specified in the rubric:

1. ✅ High forward velocity (r_progress)
2. ✅ Collision penalty (r_collision)
3. ✅ Slow driving penalty (r_low_speed, r_slow_action)
4. ✅ Safe distances (r_headway) - **V5 only**
5. ✅ Lane change penalty (r_lane) - **V5 only**

For **research exploration** or **time-constrained training**, V4 provides a simpler baseline that may train faster.

### Next Steps

1. [ ] Train V4 agent (200k timesteps)
2. [ ] Train V5 agent (200k timesteps)
3. [ ] Evaluate both on 100 episodes
4. [ ] Generate comparison videos
5. [ ] Update this document with empirical results
6. [ ] Select final model for README showcase

---

## Appendix: File Locations

| Component | V4 Path | V5 Path |
|-----------|---------|---------|
| Environment | `src/env/highway_env.py` | `src/env/highway_env_v5.py` |
| Config | `src/config.py` (REWARD_V4_CONFIG) | `src/config.py` (REWARD_V5_CONFIG) |
| Tests | `tests/test_reward_v4.py` | `tests/test_reward_v5.py` |

---

*Document Version: 1.0*  
*Created: 2025-01-18*  
*Last Updated: 2025-01-18*
