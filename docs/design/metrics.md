# Test Framework & Evaluation Metrics Design

**Owner:** Member 5
**Week:** 1 — Project Setup & Design
**Status:** Draft — to be reviewed with the team once module interfaces (Weeks 1–2) are finalized

---

## 1. Testing Approach

### 1.1 Test Structure

```
tests/
  unit/            # test one module/class in isolation, mock dependencies
  integration/     # test two or more modules working together
```

| Layer | What it tests | Owner (writes tests) | When |
|---|---|---|---|
| `tests/unit/` | Individual classes/functions (e.g. `WarehouseEnv`, `AStarPlanner`, `QNetwork`) | Each member tests their own module; Member 5 sets conventions and writes shared/core tests | Week 2 |
| `tests/integration/` | Cross-module behavior (Env+A*, Env+DDQN, full hybrid system) | Member 5 | Weeks 3 & 5 |

### 1.2 Conventions
- **Naming:** `test_<module>_<behavior>.py`, functions named `test_<what_is_being_verified>()`
- **Framework:** `pytest`, run via `./run_tests.sh` or `pytest tests/ -v`
- **Fixtures:** shared fixtures live in `tests/conftest.py` — e.g. a standard small grid (5×5) for fast unit tests, and the full 12×12 grid for integration tests
- **Test doubles:** unit tests for one module should not depend on another team member's incomplete code — use stub/mock objects (e.g. a fake `AStarPlanner` that returns a fixed path) so tests can be written before all modules are done
- **Markers:** use `@pytest.mark.slow` for anything involving actual training, so fast tests can be run separately (`pytest -m "not slow"`)

### 1.3 Planned Fixtures (`tests/conftest.py`)
- `small_grid_env` — 5×5 grid, no obstacles, for fast sanity checks
- `standard_grid_env` — 12×12 grid, default config, for realistic tests
- `dense_obstacle_env`, `narrow_aisle_env` — pre-configured scenario fixtures (aligned with Week 6 test scenarios)
- `sample_waypoints` — a canned waypoint list for testing hybrid/DDQN code without depending on A* being finished

### 1.4 Minimum Bar per Week
- Week 2: >15 unit tests passing (environment + A*)
- Week 3: integration tests for Env+A* and Env+DDQN passing
- Week 5: end-to-end hybrid system tests passing across all three context types (Aisle/Open/Dense)

---

## 2. Evaluation Metrics

These are the metrics used to judge every experiment run (baseline, ablations, full hybrid) from Week 4 onward.

### 2.1 Primary Metrics

| Metric | Definition | Why it matters |
|---|---|---|
| **Success rate** | % of episodes where the robot reaches the goal within the step limit | Core measure of task completion |
| **Collision rate** | Avg. collisions per episode (with static shelving or dynamic obstacles) | Safety — central to comparing reward strategies |
| **Steps-to-goal** | Avg. number of steps taken in successful episodes | Efficiency |
| **Path efficiency ratio** | executed path length ÷ A* optimal path length | How closely the agent follows the "ideal" route |

### 2.2 Learning Metrics
- **Episode reward** (raw, and a moving average, e.g. window=50) — the standard learning curve
- **Convergence episode** — first episode after which the moving-average reward stabilizes within a tolerance band (exact threshold to be set with Member 3 once training logs exist)
- **Q-value trend** — mean predicted Q-value over training (Member 3 owns this analysis; Member 5 provides the logging hook)

### 2.3 Statistical Reporting
- **Seeds:** 2 random seeds per configuration (per revised plan)
- **Episodes:** ~600–800 per run
- Report **mean ± std** across seeds for all primary metrics
- Use a simple significance check (e.g. paired comparison) between adaptive vs. static reward results, since that's the project's central claim — not a full hypothesis-testing pipeline, just enough to support the comparison honestly

### 2.4 Data to Log per Run
Each experiment script (Week 4/6) should output a standard results file so `analysis/` code can consume it consistently:
```
experiments/results/<run_name>_seed<N>.json
{
  "config": {...},
  "episode_rewards": [...],
  "success": [...],       # per-episode boolean
  "collisions": [...],    # per-episode count
  "steps": [...],         # per-episode count
  "path_efficiency": [...]
}
```
This schema should be agreed with Member 3 (who runs training) and Member 4 (hybrid agent) before Week 4.

---

## 3. Benchmark Plan

### 3.1 Configurations to Compare (Week 6)
1. Pure A* (no learning, re-planning only) — Member 2
2. Pure DDQN (no A* guidance) — baseline, Week 4
3. Hybrid, static reward — comparison baseline for the core ablation
4. Hybrid, adaptive reward — the project's main contribution
5. **Ablation A:** adaptive vs. static reward (isolates the reward mechanism's effect)
6. **Ablation B:** adaptive reward, no A* guidance (isolates A*'s contribution)

### 3.2 Test Scenarios (built by Member 1, Week 6)
- Narrow Aisle
- Wide Open
- Mixed
- Dynamic-Heavy

Each configuration above should be run across all applicable scenarios where feasible, budget permitting (2 seeds × ~700 episodes × 6 configs × 4 scenarios is a lot — may need to prioritize; flag this with the team once Week 4 baseline timing is known).

### 3.3 Dependencies / Open Questions for Team Sync
- [ ] Confirm `env.step()` return signature with Member 1 (needed for test fixtures)
- [ ] Confirm A* output format (waypoint list structure) with Member 2
- [ ] Agree on results JSON schema with Member 3 and Member 4 before Week 4
- [ ] Confirm convergence-threshold definition with Member 3 once first training curves exist

---

*This document will be revisited at the end of Week 2 once module interfaces are implemented, to confirm fixtures and mocks match real APIs.*