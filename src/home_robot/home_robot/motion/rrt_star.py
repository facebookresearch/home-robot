from random import random
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

from home_robot.motion.base import Planner, PlanResult
from home_robot.motion.space import ConfigurationSpace, Node

class OptimalNode:
    """Placeholder class"""
    pass

class OptimalNode(Node):
    """Stores an individual spot in the tree"""

    def __init__(self, state: np.ndarray, parent: Optional[OptimalNode] = None, d=0):
        """A treenode is just a pointer back to its parent and an associated state."""
        super(OptimalNode, self).__init__(state)
        self.state = state
        self.parent = parent
        self.children = set()
        self.d = d
        if parent is not None:
            self.cost = parent.cost + d
            self.parent.children.add(self)
        else:
            self.cost = d
        self.solution = False
        
    def set_solution(self, solution):
        if self.solution is solution:
            return
        self.solution = solution
        if self.parent is not None:
            self.parent.set_solution(solution)

    def backup(self) -> List[OptimalNode]:
        """Get the full plan by looking back from this point. Returns a list of TreeNodes which contain state."""
        sequence = []
        node = self
        # Look backwards to get a tree
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]
    
    def rewire(self, parent, d):
        if self.solution:
            self.parent.set_solution(False)
        self.parent.children.remove(self)
        self.parent = parent
        self.parent.children.add(self)
        if self.solution:
            self.parent.set_solution(True)
        self.d = d
        self.update()
        
    def update(self):
        self.cost = self.parent.cost + self.d
        for n in self.children:
            n.update()

class OptimalNodeManager():
    def __init__(self, array_len: int):
        self.nodes = []
        self.points = np.zeros((array_len, 3))
        self.points_size = 0
        self.tree = None    # cached value

    def add_node(self, node: OptimalNode):
        self.nodes.append(node)
        self.points[self.points_size] = node.state
        self.points_size += 1
        
        # invalidate tree cache
        self.tree = None

    def get_nearest_neighbours(self, new_node: OptimalNode, k: int) -> List[Tuple[OptimalNode, float]]:
        if self.tree is None:
            self.tree = KDTree(self.points[:self.points_size])
        
        distances, indices = self.tree.query(new_node.state, k if k <= self.points_size else self.points_size)
        return [(self.nodes[idx], distances[i]) for i, idx in enumerate(indices)]

@dataclass
class IntermediateResult:
    success: bool
    cost: Optional[float] = None
    trajectory: Optional[List] = None

class RRTStar(Planner):
    """
    RRT* algorithm for planning optimal navigation trajectories.
    
    NOTE: this function is very inefficient and takes a long time to run. As such, it may not be suitable for simulations.
    If using RTT*, I recommend using it for a single plan only, and falling back to RTTConnect if planning failed.
    I also only recommend using RRT* for actual motion planning where optimal plans are required, and not for example checking if an object is reachable.
    """

    def __init__(
        self,
        space: ConfigurationSpace,
        validate_fn: Callable,
        p_sample_goal: float = 0.1,
        goal_tolerance: float = 2*1e-5,
        max_iter: int = 100,
    ):
        """Create RRT* planner with configuration"""
        super(RRTStar, self).__init__(space, validate_fn)
        self.p_sample_goal = p_sample_goal
        self.goal_tolerance = goal_tolerance
        self.max_iter = max_iter
        self.reset()

    def reset(self):
        self.goal_state = None
        self.node_manager = OptimalNodeManager(self.max_iter * 1000)

    def get_nodes(self) -> List[Node]:
        return self.node_manager.nodes

    def plan(self, start, goal, verbose: bool = False) -> PlanResult:
        """plan from start to goal. creates a new tree.

        Based on Caelan Garrett's code (MIT licensed):
        https://github.com/caelan/motion-planners/blob/master/motion_planners/rrt_star.py
        """
        assert len(start[:2]) == self.space.dof, "invalid start dimensions"
        assert len(goal[:2]) == self.space.dof, "invalid goal dimensions"
        self.reset()
        
        if not self.validate(start):
            if verbose:
                print("[Planner] invalid start")
            return PlanResult(False)
        if not self.validate(goal):
            if verbose:
                print("[Planner] invalid goal")
            return PlanResult(False)
        # Add start to the tree
        self.node_manager.add_node(OptimalNode(start))

        self.goal_state = goal
        best_plan = None
        
        # Always try goal first
        temp_plan = self.step_planner(self.node_manager, force_sample_goal=True)
        if temp_plan.success:
            best_plan = temp_plan
        # Iterate a bunch of times
        for i in range(self.max_iter - 1):
            temp_plan = self.step_planner(self.node_manager)
            if temp_plan.success:
                if best_plan is None or temp_plan.cost < best_plan.cost:
                    best_plan = temp_plan
        
        if best_plan is None:
            return PlanResult(False)
        return PlanResult(True, trajectory=best_plan.trajectory)

    def step_planner(
        self,
        node_manager: OptimalNodeManager,
        force_sample_goal=False,
        next_state: Optional[np.ndarray] = None,
    ) -> IntermediateResult:
        """Continue planning for a while. In case you want to try for anytime planning."""
        assert (
            self.goal_state is not None
        ), "no goal provided with a call to plan(start, goal)"
        if force_sample_goal or next_state is not None:
            should_sample_goal = True
        else:
            should_sample_goal = random() < self.p_sample_goal
            
        res = IntermediateResult(False)

        # Get a new state
        if next_state is not None:
            goal_state = next_state
        else:
            goal_state = self.goal_state
        # Set the state we will try to move to
        if next_state is None:
            next_state = goal_state if should_sample_goal else self.space.sample()
                    
        closest = self.space.closest_node_to_state(next_state, node_manager.nodes)
        path = safe_path(self.space.extend(closest.state, next_state), self.validate)
        if len(path) == 0:
            # no safe path
            return res
        
        for path_node in path:
            closest = OptimalNode(path_node, parent=closest, d=self.space.distance(closest.state, path_node))
            node_manager.add_node(closest)
            
        if self.space.distance(closest.state, goal_state) < self.goal_tolerance:
            res = IntermediateResult(True, closest.cost, closest.backup())
            
        neighbors = node_manager.get_nearest_neighbours(closest, 10)#, self.space.distance)
        for n, d in neighbors:
            if (n.cost + d) < closest.cost:
                path = safe_path(self.space.extend(n.state, closest.state), self.validate)
                path_closest = n
                for path_node in path:
                    path_closest = OptimalNode(path_node, parent=path_closest, d=self.space.distance(path_closest.state, path_node))
                    node_manager.add_node(path_closest)
                if (len(path) > 0) and (self.space.distance(closest.state, path_closest.state) < self.goal_tolerance):
                    closest.rewire(path_closest, self.space.distance(closest.state, path_closest.state))
        return res

def safe_path(sequence, validate):
    path = []
    for q in sequence:
        if not validate(q):
            break
        path.append(q)
    return path

    