from typing import Any, Dict, List, Tuple, Union

import brax
from brax import jumpy as jp
from brax.envs import env


class PointMaze(env.Env):
    """Jax/Brax implementation of the PointMaze.
    Highly inspired from the old python implementation of
    the PointMaze.

    In order to stay in the Brax API, I will use a fake QP
    at several moment of the implementation. This enable to
    use the brax.envs.env.State from Brax. To avoid this,
    it would be good to ask Brax to enlarge a bit their API
    for environments that are not physically simulated.
    """

    def __init__(
        self,
        scale_action_space: float = 10,
        x_min: float = -1,
        x_max: float = 1,
        y_min: float = -1,
        y_max: float = 1,
        zone_width: float = 0.1,
        zone_width_offset_from_x_min: float = 0.5,
        zone_height_offset_from_y_max: float = -0.2,
        wall_width_ratio: float = 0.75,
        upper_wall_height_offset: float = 0.2,
        lower_wall_height_offset: float = -0.5,
        **kwargs: Any,
    ) -> None:

        super().__init__(None, **kwargs)

        self._scale_action_space = scale_action_space
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max

        self._low = jp.array([self._x_min, self._y_min], dtype=jp.float32)
        self._high = jp.array([self._x_max, self._y_max], dtype=jp.float32)

        self.n_zones = 1

        self.zone_width = zone_width

        self.zone_width_offset = self._x_min + zone_width_offset_from_x_min
        self.zone_height_offset = self._y_max + zone_height_offset_from_y_max

        self.viewer = None

        # Walls
        self.wallheight = 0.01
        self.wallwidth = (self._x_max - self._x_min) * wall_width_ratio

        self.upper_wall_width_offset = self._x_min + self.wallwidth / 2
        self.upper_wall_height_offset = upper_wall_height_offset

        self.lower_wall_width_offset = self._x_max - self.wallwidth / 2
        self.lower_wall_height_offset = lower_wall_height_offset

    @property
    def descriptors_min_values(self) -> List[float]:
        """Minimum values for descriptors."""
        return [self._x_min, self._y_min]

    @property
    def descriptors_max_values(self) -> List[float]:
        """Maximum values for descriptors."""
        return [self._x_max, self._y_max]

    @property
    def descriptors_names(self) -> List[str]:
        """Descriptors names."""
        return ["x_pos", "y_pos"]

    @property
    def state_descriptor_length(self) -> int:
        return 2

    @property
    def state_descriptor_name(self) -> str:
        return "xy_position"

    @property
    def state_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return [self._x_min, self._y_min], [self._x_max, self._y_max]

    @property
    def behavior_descriptor_length(self) -> int:
        return self.state_descriptor_length

    @property
    def behavior_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return self.state_descriptor_limits

    @property
    def action_size(self) -> int:
        """The size of the observation vector returned in step and reset."""
        return 2

    def reset(self, rng: jp.ndarray) -> env.State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jp.random_split(rng, 3)
        # get initial position - reproduce the old implementation
        x_init = jp.random_uniform(rng1, (), low=self._x_min, high=self._x_max) / 10
        y_init = jp.random_uniform(rng2, (), low=self._y_min, high=-0.7)
        obs_init = jp.array([x_init, y_init])
        # create fake qp (to re-use brax.State)
        fake_qp = brax.QP.zero()
        # init reward, metrics and infos
        reward, done = jp.zeros(2)
        metrics: Dict = {}
        # managing state descriptor by our own
        info_init = {"state_descriptor": obs_init}
        return env.State(fake_qp, obs_init, reward, done, metrics, info_init)

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""

        # clip action taken
        min_action = self._low
        max_action = self._high
        action = jp.clip(action, min_action, max_action) / self._scale_action_space

        # get the current position
        x_pos_old, y_pos_old = state.obs

        # compute the new position
        x_pos = x_pos_old + action[0]
        y_pos = y_pos_old + action[1]

        # take into account a potential wall collision
        y_pos = self._collision_lower_wall(y_pos, y_pos_old, x_pos, x_pos_old)
        y_pos = self._collision_upper_wall(y_pos, y_pos_old, x_pos, x_pos_old)

        # take into account border walls
        x_pos = jp.clip(x_pos, jp.array(self._x_min), jp.array(self._x_max))
        y_pos = jp.clip(y_pos, jp.array(self._y_min), jp.array(self._y_max))

        reward = -jp.norm(
            jp.array([x_pos - self.zone_width_offset, y_pos - self.zone_height_offset])
        )
        # determine if zone was reached
        in_zone = self._in_zone(x_pos, y_pos)

        done = jp.where(
            jp.array(in_zone),
            x=jp.array(1.0),
            y=jp.array(0.0),
        )

        new_obs = jp.array([x_pos, y_pos])

        # update state descriptor
        state.info["state_descriptor"] = new_obs
        # update the state
        return state.replace(obs=new_obs, reward=reward, done=done)  # type: ignore

    def _in_zone(self, x_pos: jp.ndarray, y_pos: jp.ndarray) -> Union[bool, jp.ndarray]:
        """Determine if the point reached the goal area."""
        zone_center_width, zone_center_height = (
            self.zone_width_offset,
            self.zone_height_offset,
        )

        condition_1 = zone_center_width - self.zone_width / 2 <= x_pos
        condition_2 = x_pos <= zone_center_width + self.zone_width / 2

        condition_3 = zone_center_height - self.zone_width / 2 <= y_pos
        condition_4 = y_pos <= zone_center_height + self.zone_width / 2

        return condition_1 & condition_2 & condition_3 & condition_4

    def _collision_lower_wall(
        self,
        y_pos: jp.ndarray,
        y_pos_old: jp.ndarray,
        x_pos: jp.ndarray,
        x_pos_old: jp.ndarray,
    ) -> jp.ndarray:
        """Manage potential collisions with the walls."""

        # global conditions on the x axis contacts
        x_hitting_wall = (self.lower_wall_height_offset - y_pos_old) / (
            y_pos - y_pos_old
        ) * (x_pos - x_pos_old) + x_pos_old
        x_axis_contact_condition = x_hitting_wall >= self._x_max - self.wallwidth

        # From down - boolean style
        y_axis_down_contact_condition_1 = y_pos_old <= self.lower_wall_height_offset
        y_axis_down_contact_condition_2 = self.lower_wall_height_offset < y_pos
        # y_pos update
        new_y_pos = jp.where(
            y_axis_down_contact_condition_1
            & y_axis_down_contact_condition_2
            & x_axis_contact_condition,
            x=jp.array(self.lower_wall_height_offset),
            y=y_pos,
        )

        # From up - boolean style
        y_axis_up_contact_condition_1 = (
            y_pos < self.lower_wall_height_offset + self.wallheight
        )
        y_axis_up_contact_condition_2 = (
            self.lower_wall_height_offset + self.wallheight <= y_pos_old
        )
        y_axis_up_contact_condition_3 = y_pos_old < self.upper_wall_height_offset
        # y_pos update
        new_y_pos = jp.where(
            y_axis_up_contact_condition_1
            & y_axis_up_contact_condition_2
            & y_axis_up_contact_condition_3
            & x_axis_contact_condition,
            x=jp.array(self.lower_wall_height_offset + self.wallheight),
            y=new_y_pos,
        )

        return new_y_pos

    def _collision_upper_wall(
        self,
        y_pos: jp.ndarray,
        y_pos_old: jp.ndarray,
        x_pos: jp.ndarray,
        x_pos_old: jp.ndarray,
    ) -> jp.ndarray:
        """Manage potential collisions with the walls."""

        # global conditions on the x axis contacts
        x_hitting_wall = (self.upper_wall_height_offset - y_pos_old) / (
            y_pos - y_pos_old
        ) * (x_pos - x_pos_old) + x_pos_old
        x_axis_contact_condition = x_hitting_wall <= self._x_min + self.wallwidth

        # From up - boolean style
        y_axis_up_contact_condition_1 = (
            y_pos_old >= self.upper_wall_height_offset + self.wallheight
        )
        y_axis_up_contact_condition_2 = (
            self.upper_wall_height_offset + self.wallheight > y_pos
        )
        # y_pos update
        new_y_pos = jp.where(
            y_axis_up_contact_condition_1
            & y_axis_up_contact_condition_2
            & x_axis_contact_condition,
            x=jp.array(self.upper_wall_height_offset + self.wallheight),
            y=y_pos,
        )

        # From down - boolean style
        y_axis_down_contact_condition_1 = y_pos > self.upper_wall_height_offset
        y_axis_down_contact_condition_2 = self.upper_wall_height_offset >= y_pos_old
        y_axis_down_contact_condition_3 = y_pos_old > self.lower_wall_height_offset
        # y_pos update
        new_y_pos = jp.where(
            y_axis_down_contact_condition_1
            & y_axis_down_contact_condition_2
            & y_axis_down_contact_condition_3
            & x_axis_contact_condition,
            x=jp.array(self.upper_wall_height_offset),
            y=new_y_pos,
        )

        return new_y_pos
