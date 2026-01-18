# V4: Acceleration-Aware Reward (No Overtaking)

**Created:** January 18, 2026  
**Status:** ✅ Implemented & Tested  
**Training Status:** Not yet started (V3.5 currently training)

---

## Overview

V4 is a **simplified alternative** to V3.5 Enhanced that removes all overtaking logic and focuses on core driving mechanics with acceleration awareness.

### Rationale

- V3.5's 95% confidence overtaking may be too complex for CPU-limited training
- V4 tests simpler hypothesis: **reward acceleration directly**
- Removes 3 complex methods, ~200 lines of overtaking tracking code
- Easier to debug and understand agent behavior

---

## Changes from V3.5

### ✅ Added (New in V4)

1. **Acceleration Bonus**
   - Track previous velocity: `self.prev_velocity`
   - Calculate velocity delta: `Δv = velocity_ratio - prev_velocity`
   - Include in progress reward: `r_progress = velocity_ratio + 0.2 × Δv`
   - **Effect:** Rewards speeding up, not just maintaining speed

2. **Context-Dependent SLOWER Penalty**
   ```python
   if velocity < 70%: penalty = -0.05  # Heavy when already slow
   else: penalty = -0.01               # Light when moving fast
   ```
   - **Effect:** Discourages slowing down when already slow

3. **FASTER Bonus** (when slow)
   ```python
   if action == FASTER and velocity < 80%: bonus = +0.05
   else: bonus = 0.0
   ```
   - **Effect:** Encourages active acceleration when below threshold

### ❌ Removed (From V3.5)

1. **Overtaking tracking**
   - `self.previous_vehicles_behind`
   - `self.recent_overtake_times`
   - `self.current_step`

2. **Overtaking bonus**
   - `r_overtake_bonus` (+2.0 per vehicle passed)
   - `min_overtake_speed` (2.0 m/s threshold)

3. **Risk logic**
   - `r_collision_risky` (-38.0 additional penalty)
   - `overtake_risk_window` (3-second risk window)
   - `_is_in_risky_overtaking_period()`
   - `_detect_overtakes()`

4. **Dynamic collision penalty**
   - Simplified to single value: `-100.0` (always)

---

## Mathematical Model

### V4 Reward Function (6 components)

$$
R = r_{\text{progress}}(v, \Delta v) + r_{\text{alive}} + r_{\text{collision}} + r_{\text{slow}}(\text{context}) + r_{\text{low\_speed}} + r_{\text{faster}}
$$

### Component Breakdown

| Component | Formula | Value | Condition |
|-----------|---------|-------|-----------|
| **Progress** | $v_{\text{ratio}} + 0.2 \times \Delta v$ | Variable | Always |
| **Alive** | Constant | +0.01 | Always |
| **Collision** | Penalty | -100.0 | If crashed |
| **SLOWER** | Context-dependent | -0.05 or -0.01 | If action=SLOWER |
| **Low Speed** | Penalty | -0.20 | If $v < 18$ m/s |
| **FASTER** | Bonus | +0.05 | If action=FASTER & $v < 80\%$ |

### Progress Reward Detail

$$
r_{\text{progress}} = \frac{v_{\text{ego}} + 1}{2} + 0.2 \times \left(\frac{v_t + 1}{2} - \frac{v_{t-1} + 1}{2}\right)
$$

Where:
- $v_t$: Normalized velocity at current step $\in [-1, 1]$
- $v_{t-1}$: Normalized velocity at previous step
- Acceleration weight: **0.2** (20% as important as current speed)

---

## Test Results

All 9 tests passed:

```
✅ Acceleration bonus: 0.7400 (includes Δv = 0.20)
✅ SLOWER when slow: -0.05 (heavy penalty)
✅ SLOWER when fast: -0.01 (light penalty)
✅ FASTER when slow: +0.05 (encourages acceleration)
✅ FASTER when fast: 0.0 (no bonus needed)
✅ No overtaking logic (removed from V3.5)
✅ Single collision penalty: -100.0 (no risk logic)
✅ V4 structure: 0.8700 = progress(0.8600) + alive(0.01) + ...
✅ Acceleration rewarded: 0.7200 > 0.5800
```

---

## Expected Behavior

### Positive Feedback Loop
1. Agent uses **FASTER** when slow → gets +0.05 bonus
2. Velocity increases → $\Delta v > 0$ → higher progress reward
3. Agent maintains speed → continues getting high progress reward
4. **Result:** Learns "accelerate early, maintain speed"

### Negative Feedback (Safety)
1. Agent crashes → -100.0 penalty (same as V3.5 normal crash)
2. Agent uses SLOWER when already slow → -0.05 (heavy penalty)
3. Agent drives below 18 m/s → -0.20 penalty
4. **Result:** Learns "stay fast, avoid crashes"

---

## Implementation Files

### Created
- [`src/env/highway_env_v4.py`](../src/env/highway_env_v4.py) - V4 environment wrapper (364 lines)
- [`tests/test_reward_v4.py`](../tests/test_reward_v4.py) - V4 test suite (150 lines)

### Modified
- [`src/config.py`](../src/config.py) - Added `REWARD_V4_CONFIG` section (lines 323-373)

### NOT Modified (Training)
- `scripts/train.py` - Still uses V3.5 Enhanced
- To train V4: Import `HighwayEnvV4` and use `REWARD_V4_CONFIG`

---

## Configuration Parameters

```python
REWARD_V4_CONFIG = {
    # === CORE (SAME AS V3) ===
    "r_alive": 0.01,
    "r_collision": -100.0,
    "min_speed_threshold": 18.0,
    "r_low_speed": -0.20,
    
    # === V4 NEW ===
    "acceleration_weight": 0.2,          # Acceleration bonus weight
    "slow_velocity_threshold": 0.7,      # 70% for SLOWER penalty
    "r_slower_heavy": -0.05,             # When already slow
    "r_slower_light": -0.01,             # When moving fast
    "faster_velocity_threshold": 0.8,    # 80% for FASTER bonus
    "r_faster_bonus": 0.05,              # Small positive when slow
}
```

---

## Decision Tree: When to Use V4

```
Is V3.5 training successful?
├─ YES (agent learns strategic overtaking)
│  └─ Use V3.5 for deliverables, document V4 as alternative
│
└─ NO (agent still passive or crashes often)
   └─ Is overtaking logic causing issues?
      ├─ YES (too complex, training unstable)
      │  └─ **Train V4** (simpler, acceleration-focused)
      │
      └─ NO (other issues)
         └─ Debug action space or hyperparameters
```

---

## Advantages vs V3.5

1. **Simpler:** 364 lines vs 712 lines (-49%)
2. **Fewer tracking variables:** 1 vs 3 (prev_velocity only)
3. **No complex methods:** Removed `_detect_overtakes()`, `_is_in_risky_overtaking_period()`
4. **Easier debugging:** Direct acceleration feedback, no risk window logic
5. **CPU-friendly:** Less computation per step

---

## Disadvantages vs V3.5

1. **No overtaking awareness:** Agent won't learn strategic passing
2. **Less sophisticated:** Doesn't model risk-aware decision making
3. **May encourage risky acceleration:** No penalty for speeding up near vehicles
4. **No secondary objective:** V3.5 has "speed (primary) + overtaking (bonus)"

---

## Next Steps

1. **Monitor V3.5 training** (currently at 26k/200k steps)
2. **If V3.5 succeeds:** Document V4 as alternative, keep V3.5 for submission
3. **If V3.5 fails:** Train V4 as backup plan (~90 minutes)
4. **Document in README:** Add V4 to "Challenges & Failures" section if unused

---

## Summary

V4 tests the hypothesis: **"Direct acceleration reward is simpler and more effective than overtaking bonuses."**

- **If true:** V4 becomes primary submission
- **If false:** V3.5 proves sophisticated risk modeling works

Both are valid approaches with different trade-offs. V3.5 = sophisticated risk-aware, V4 = simple acceleration-focused.
