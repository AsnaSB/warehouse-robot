"""
hybrid_agent.py

The central brain of the hybrid architecture.
Combines global A* pathfinding (WaypointManager & DynamicReplanner)
with local DDQN obstacle avoidance.
"""

from typing import Tuple, Set, Optional

Position = Tuple[int, int]


class HybridAgent:
    """
    Coordinates global A* routing with local DDQN action selection.
    """
    def __init__(
        self, 
        ddqn_agent, 
        waypoint_manager, 
        replanner, 
        astar_planner
    ):
        self.ddqn = ddqn_agent
        self.waypoint_manager = waypoint_manager
        self.replanner = replanner
        self.planner = astar_planner

    def update_route_and_waypoint(
        self, 
        current_pos: Position, 
        goal_pos: Position, 
        blocked_cells: Optional[Set[Position]] = None
    ) -> Position:
        """
        Ensures global path validity, triggers replanning if blocked,
        advances waypoints, and returns the active waypoint.
        """
        if blocked_cells is None:
            blocked_cells = set()

        # 1. Initialize path if not set
        if not self.waypoint_manager.path:
            initial_path = self.planner.find_path(current_pos, goal_pos, blocked_cells)
            self.waypoint_manager.set_path(initial_path)

        # 2. Check dynamic obstacles / blocked route
        is_blocked = self.replanner.is_path_blocked(self.waypoint_manager.path, blocked_cells)
        is_invalid = self.waypoint_manager.is_waypoint_invalid(blocked_cells)

        if is_blocked or is_invalid:
            new_path = self.replanner.replan(
                current_position=current_pos,
                goal=goal_pos,
                current_path=self.waypoint_manager.path,
                blocked_cells=blocked_cells
            )
            self.waypoint_manager.set_path(new_path)

        # 3. Advance waypoint if current one reached
        if self.waypoint_manager.is_waypoint_reached(current_pos):
            self.waypoint_manager.advance_waypoint()

        return self.waypoint_manager.get_current_waypoint()

    def get_action(self, state, epsilon: float = 0.0) -> int:
        """Queries the DDQN for local action with epsilon-greedy exploration."""
        return self.ddqn.select_action(state, epsilon=epsilon)