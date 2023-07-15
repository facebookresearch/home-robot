# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, List, Optional, Tuple

import numpy as np
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import AgentState
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations import fog_of_war, maps
from omegaconf import DictConfig

try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    pass

cv2 = try_cv2_import()


MAP_THICKNESS_SCALAR: int = 128


@registry.register_measure
class MyTopDownMap(Measure):
    """Top Down Map measure. The same as habitat's TopDownMap but it
    populates the metric with an initial map at step zero.
    """

    def __init__(
        self, sim: "HabitatSim", config: DictConfig, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.map_padding
        self._step_count: Optional[int] = None
        self._map_resolution = config.map_resolution
        self._ind_x_min: Optional[int] = None
        self._ind_x_max: Optional[int] = None
        self._ind_y_min: Optional[int] = None
        self._ind_y_max: Optional[int] = None
        self._previous_xy_location: Optional[Tuple[int, int]] = None
        self._top_down_map: Optional[np.ndarray] = None
        self._shortest_path_points: Optional[List[Tuple[int, int]]] = None
        self.line_thickness = int(
            np.round(self._map_resolution * 2 / MAP_THICKNESS_SCALAR)
        )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution / MAP_THICKNESS_SCALAR)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "top_down_map"

    def get_original_map(self):
        top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=self._map_resolution,
            draw_border=self._config.draw_border,
        )

        if self._config.fog_of_war.draw:
            self._fog_of_war_mask = np.zeros_like(top_down_map)
        else:
            self._fog_of_war_mask = None

        return top_down_map

    def _draw_point(self, position, point_type):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        self._top_down_map[
            t_x - self.point_padding : t_x + self.point_padding + 1,
            t_y - self.point_padding : t_y + self.point_padding + 1,
        ] = point_type

    def _draw_goals_view_points(self, episode):
        if self._config.draw_view_points:
            for goal in episode.goals:
                if self._is_on_same_floor(goal.position[1]):
                    try:
                        if goal.view_points is not None:
                            for view_point in goal.view_points:
                                self._draw_point(
                                    view_point.agent_state.position,
                                    maps.MAP_VIEW_POINT_INDICATOR,
                                )
                    except AttributeError:
                        pass

    def _draw_goals_positions(self, episode):
        if self._config.draw_goal_positions:

            for goal in episode.goals:
                if self._is_on_same_floor(goal.position[1]):
                    try:
                        self._draw_point(goal.position, maps.MAP_TARGET_POINT_INDICATOR)
                    except AttributeError:
                        pass

    def _draw_goals_aabb(self, episode):
        if self._config.draw_goal_aabbs:
            for goal in episode.goals:
                try:
                    sem_scene = self._sim.semantic_annotations()
                    object_id = goal.object_id
                    assert int(sem_scene.objects[object_id].id.split("_")[-1]) == int(
                        goal.object_id
                    ), f"Object_id doesn't correspond to id in semantic scene objects dictionary for episode: {episode}"

                    center = sem_scene.objects[object_id].aabb.center
                    x_len, _, z_len = sem_scene.objects[object_id].aabb.sizes / 2.0
                    # Nodes to draw rectangle
                    corners = [
                        center + np.array([x, 0, z])
                        for x, z in [
                            (-x_len, -z_len),
                            (-x_len, z_len),
                            (x_len, z_len),
                            (x_len, -z_len),
                            (-x_len, -z_len),
                        ]
                        if self._is_on_same_floor(center[1])
                    ]

                    map_corners = [
                        maps.to_grid(
                            p[2],
                            p[0],
                            (
                                self._top_down_map.shape[0],
                                self._top_down_map.shape[1],
                            ),
                            sim=self._sim,
                        )
                        for p in corners
                    ]

                    maps.draw_path(
                        self._top_down_map,
                        map_corners,
                        maps.MAP_TARGET_BOUNDING_BOX,
                        self.line_thickness,
                    )
                except AttributeError:
                    pass

    def _draw_shortest_path(
        self, episode: NavigationEpisode, agent_position: AgentState
    ):
        if self._config.draw_shortest_path:
            _shortest_path_points = self._sim.get_straight_shortest_path_points(
                agent_position, episode.goals[0].position
            )
            self._shortest_path_points = [
                maps.to_grid(
                    p[2],
                    p[0],
                    (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                    sim=self._sim,
                )
                for p in _shortest_path_points
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )

    def _is_on_same_floor(self, height, ref_floor_height=None, ceiling_height=2.0):
        if ref_floor_height is None:
            ref_floor_height = self._sim.get_agent(0).state.position[1]
        return ref_floor_height < height < ref_floor_height + ceiling_height

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        if hasattr(episode, "goals"):
            # draw source and target parts last to avoid overlap
            self._draw_goals_view_points(episode)
            self._draw_goals_aabb(episode)
            self._draw_goals_positions(episode)
            self._draw_shortest_path(episode, agent_position)

        if self._config.draw_source:
            self._draw_point(episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR)

        # want initial map
        self.update_metric(None, None)
        self._step_count -= 1

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        self._metric = {
            "map": house_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": (map_agent_x, map_agent_y),
            "agent_angle": self.get_polar_angle(),
        }

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.max_episode_steps, 245
            )

            thickness = self.line_thickness
            cv2.line(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                color,
                thickness=thickness,
            )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_fog_of_war_mask(self, agent_position):
        if self._config.fog_of_war.draw:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                self.get_polar_angle(),
                fov=self._config.fog_of_war.fov,
                max_line_len=self._config.fog_of_war.visibility_dist
                / maps.calculate_meters_per_pixel(self._map_resolution, sim=self._sim),
            )
