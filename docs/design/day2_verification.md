# Day 2 — Dynamic Obstacle Verification

## Member

Member 1

## Objective

Implement and verify the dynamic obstacle entities required for the warehouse robot navigation environment.

The Day 2 implementation covers:

* Worker dynamic obstacle
* Predefined patrol route
* Random patrol route
* Configurable speed
* Direction changes
* Boundary handling
* Optional stopping/pausing
* Dynamic Robot
* Linear movement
* Forward/backward movement
* Path reversal
* Configurable speed
* Collision detection

---

## 1. DynamicObstacle Base Class

The `DynamicObstacle` base class provides the common structure for dynamic warehouse entities.

It contains:

* `position`
* `previous_position`
* `movement_pattern`
* `speed`
* `active` state

It also provides:

* Position update functionality
* Activation functionality
* Deactivation functionality
* Current position access
* Previous position access

### Verification

The base class was verified using unit tests covering:

* Initialization
* Position updates
* Activation
* Deactivation

**Status: PASS**

---

## 2. Worker

The `Worker` class extends `DynamicObstacle` and represents a moving warehouse worker.

### 2.1 Predefined Patrol Route

The worker supports movement along a predefined patrol route.

Example:

```text
(2,2) → (2,3) → (2,4) → (2,5) → (2,6)
```

The worker changes direction when reaching the end of the patrol route.

**Status: PASS**

---

### 2.2 Random Patrol Route

The worker supports random patrol behaviour.

The worker selects a valid neighbouring grid position and moves within the configured warehouse boundaries.

Random movement uses the four cardinal directions:

```text
N
S
E
W
```

**Status: PASS**

---

### 2.3 Configurable Speed

The `Worker` class provides a configurable `speed` parameter.

The parameter is available for controlling worker movement behaviour and is retained as part of the dynamic obstacle interface for later simulation integration.

**Status: PASS**

---

### 2.4 Direction Changes

For predefined patrol routes, the worker automatically changes direction when reaching either end of the route.

Example:

```text
(2,2) → (2,3) → (2,4) → (2,5)
                         ↓
(2,2) ← (2,3) ← (2,4) ← (2,5)
```

**Status: PASS**

---

### 2.5 Boundary Handling

Worker movement is checked against the configured warehouse grid boundaries.

The worker remains within the valid grid coordinates.

For the project environment, the standard grid is:

```text
12 × 12
```

**Status: PASS**

---

### 2.6 Optional Stopping/Pausing

The worker supports configurable pause steps after movement.

During a pause:

* The worker remains at its current position.
* The remaining pause count is decreased at each update.
* Movement resumes after the pause period.

**Status: PASS**

---

## 3. Dynamic Robot

The `DynamicRobot` class extends `DynamicObstacle` and represents a moving robot that acts as a dynamic obstacle to the main navigation robot.

### 3.1 Linear Movement

The dynamic robot supports linear movement according to a configured movement direction.

Example:

```text
(8,2) → (8,3) → (8,4) → (8,5) → (8,6)
```

**Status: PASS**

---

### 3.2 Forward/Backward Movement

The dynamic robot supports movement along a predefined path in both forward and backward directions.

The robot can reverse traversal direction when the end of its movement path is reached.

**Status: PASS**

---

### 3.3 Path Reversal

When the dynamic robot reaches the end of a predefined movement path, it can reverse its traversal direction.

Example:

```text
(9,2) → (9,3) → (9,4) → (9,5) → (9,6)
                                      ↓
(9,6) → (9,5) → (9,4) → (9,3) → (9,2)
```

**Status: PASS**

---

### 3.4 Configurable Speed

The `DynamicRobot` class provides a configurable `speed` parameter as part of the common dynamic obstacle interface.

The speed parameter is retained for integration with the later simulation and environment update mechanisms.

**Status: PASS**

---

### 3.5 Boundary Handling

The dynamic robot checks warehouse grid boundaries before movement.

When the next position would leave the configured grid and reversal is enabled, the robot reverses its movement direction.

The robot remains inside the valid grid.

**Status: PASS**

---

### 3.6 Collision Detection

The `DynamicRobot` supports collision detection against a configurable collection of occupied grid positions.

Before entering a candidate position, the robot can check whether that position is occupied.

When a collision is detected:

* The collision state is recorded.
* `last_collision` is set to `True`.
* The robot remains at its current position.
* The robot does not enter the occupied position.

When the candidate position is not occupied:

* `last_collision` remains `False`.
* The robot is allowed to move normally.

**Status: PASS**

> **Note:** This is the dynamic robot's generic collision-detection capability. Integration of the collision model with `WarehouseEnv`, including Robot ↔ Shelf, Robot ↔ Worker, Robot ↔ Dynamic Robot, and Robot ↔ Boundary collisions, is part of Day 3.

---

## 4. Multi-Step and Episode Verification

Dynamic obstacle behaviour was verified over multiple movement steps and independent simulation episodes.

### Worker Verification

The verification confirmed:

* Worker movement
* Position updates
* Predefined patrol behaviour
* Random patrol behaviour
* Direction changes
* Pause behaviour
* Boundary safety

**Status: PASS**

---

### Dynamic Robot Verification

The verification confirmed:

* Linear movement
* Forward movement
* Backward movement
* Path reversal
* Position updates
* Boundary safety
* Collision detection

**Status: PASS**

---

### Several-Episode Verification

Multiple independent verification episodes were executed to ensure that dynamic obstacle behaviour remains stable across repeated simulations.

Each episode verified:

* Worker movement
* Dynamic robot movement
* Position updates
* Boundary constraints
* Collision safety

**Status: PASS**

---

## 5. Unit Test Verification

The dynamic obstacle implementation was tested using the project unit-test suite.

### Dynamic Obstacle Tests

Tests cover:

```text
Initialization
Position update
Activation
Deactivation
```

**Status: PASS**

### Worker Tests

Tests cover:

```text
Predefined patrol
Patrol reversal
Random patrol
Pause behaviour
Boundary safety
```

**Status: PASS**

### Dynamic Robot Tests

Tests cover:

```text
Linear movement
Boundary reversal
Path reversal
Boundary safety
Collision detection
Collision blocking
Non-collision movement
Path collision
```

**Status: PASS**

---

## 6. Regression Verification

The existing project functionality was rechecked after the dynamic obstacle implementation.

### Warehouse Environment

The existing environment tests continued to pass for:

* Open layout
* Aisle layout
* Dense layout
* 12×12 grid
* Random start
* Random goal
* Eight-direction movement
* Goal detection
* Shelf collision
* Gymnasium API

**Status: PASS**

### A* Planner

The existing A* tests continued to pass for:

* Path existence
* Start equals goal
* No-path scenario

**Status: PASS**

---

## 7. Simulation Verification

The dynamic obstacle demonstration script:

```text
simulation/dynamic_obstacle_demo.py
```

was used to verify multi-step dynamic behaviour.

The demonstration verifies:

```text
Worker
 ├── predefined patrol
 ├── random patrol
 └── pause behaviour

Dynamic Robot
 ├── linear movement
 ├── boundary reversal
 ├── path reversal
 └── collision detection
```

**Status: PASS**

---

## 8. Files Implemented

### Dynamic Obstacle Module

```text
src/environment/obstacles.py
```

Contains:

* `DynamicObstacle`
* `Worker`
* `DynamicRobot`

### Unit Tests

```text
tests/unit/test_obstacles.py
```

Contains tests for the dynamic obstacle classes and their movement behaviour.

### Simulation Verification

```text
simulation/dynamic_obstacle_demo.py
```

Contains deterministic demonstrations and multi-step verification of dynamic obstacle behaviour.

### Documentation

```text
docs/design/day2_verification.md
```

Contains the Day 2 implementation and verification record.

---

## 9. Integration Boundary

Day 2 establishes the dynamic obstacle entities and their movement/collision-detection capabilities without redesigning the existing warehouse environment.

The existing environment foundation remains unchanged.

Full dynamic collision integration is intentionally deferred to Day 3, where `WarehouseEnv` will be extended to support:

```text
Robot ↔ Shelf
Robot ↔ Worker
Robot ↔ Dynamic Robot
Robot ↔ Boundary
```

This separation keeps the Day 2 implementation modular and preserves the established environment interface.

---

## 10. Final Day 2 Status

| Requirement                     | Status |
| ------------------------------- | ------ |
| DynamicObstacle base class      | PASS   |
| Worker                          | PASS   |
| Predefined patrol route         | PASS   |
| Random patrol route             | PASS   |
| Configurable speed              | PASS   |
| Direction changes               | PASS   |
| Boundary handling               | PASS   |
| Optional stopping/pausing       | PASS   |
| Dynamic Robot                   | PASS   |
| Linear movement                 | PASS   |
| Forward/backward movement       | PASS   |
| Path reversal                   | PASS   |
| Dynamic Robot speed             | PASS   |
| Collision detection             | PASS   |
| Multi-step verification         | PASS   |
| Several-episode verification    | PASS   |
| Position updates                | PASS   |
| Boundary safety                 | PASS   |
| Existing environment regression | PASS   |
| Existing A* regression          | PASS   |

## Conclusion

The Member 1 Day 2 dynamic obstacle implementation has been completed and verified.

The Worker and DynamicRobot entities provide the required movement behaviours, boundary handling, configurable parameters, and collision-detection capability.

The implementation is ready for the **Day 3 collision-model and context-classification work**, where the dynamic obstacle entities will be integrated with `WarehouseEnv`.
