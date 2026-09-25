import gymnasium as gym
import numpy as np
from evogym.envs.base import EvoGymBase
from evogym import EvoViewer


class HopperJumpEnv(EvoGymBase):
    metadata = {"render_modes": ["human", "img"]}

    def __init__(self, world, robot_name, total_timesteps, render_mode=None):
        super().__init__(world=world, render_mode=render_mode)

        self.robot_name = robot_name

        self.sim_steps = 3
        self.max_steps = 1000
        self.step_count = 0

        self.total_timesteps = total_timesteps
        self.global_step = 0

        self.n_act = len(self.get_actuator_indices(robot_name))

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_act,),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32
        )

        self.prev_x = self.prev_y = 0.0
        self.prev_theta = 0.0
        self.ground_y = None

        self.stagnant_steps = 0
        self.max_height = 0.0

    # -------------------------------------------------
    def _get_obs(self):
        vel = self.get_vel_com_obs(self.robot_name)
        pos = self.get_pos_com_obs(self.robot_name)

        theta = self.get_ort_obs(self.robot_name)[0]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        ang_vel = theta - self.prev_theta

        return np.array(
            [vel[0], vel[1], pos[1] - self.ground_y, theta, ang_vel],
            dtype=np.float32
        )

    # -------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        pos = self.get_pos_com_obs(self.robot_name)

        self.prev_x, self.prev_y = pos
        self.ground_y = pos[1]
        self.max_height = pos[1]

        theta = self.get_ort_obs(self.robot_name)[0]
        self.prev_theta = (theta + np.pi) % (2 * np.pi) - np.pi

        self.step_count = 0
        self.stagnant_steps = 0

        return self._get_obs(), {}

    # -------------------------------------------------
    def step(self, action):
        self.step_count += 1
        self.global_step += 1

        # ACTION
        raw = np.clip(action, -1.0, 1.0)
        target = np.clip(1.0 + raw, 0.0, 2.0)

        for _ in range(self.sim_steps):
            super().step({self.robot_name: target})

        # STATE
        pos = self.get_pos_com_obs(self.robot_name)
        x, y = pos

        dx = x - self.prev_x
        dy = y - self.prev_y

        self.prev_x, self.prev_y = x, y
        self.ground_y = min(self.ground_y, y)

        theta = self.get_ort_obs(self.robot_name)[0]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        ang_vel = theta - self.prev_theta
        self.prev_theta = theta

        airborne = y > self.ground_y + 0.25
        self.stagnant_steps = self.stagnant_steps + 1 if abs(dx) < 0.02 else 0

        height_gain = max(0.0, y - self.max_height)
        self.max_height = max(self.max_height, y)

        # =======================
        # REWARD
        # =======================
        reward = 0.0

        # forward
        reward += 4.0 * max(dx, 0.0)
        reward -= 3.0 * max(-dx, 0.0)

        # jumping
        if airborne:
            reward += 2.0 * max(dy, 0.0)
            reward += 1.5 * max(dx, 0.0)

        # obstacle crossing
        reward += 5.0 * height_gain

        # stability
        reward += 0.4 * np.cos(theta)
        reward -= 0.25 * (ang_vel ** 2)

        # penalize useless bouncing
        if not airborne:
            reward -= 0.4 * abs(dy)

        # stagnation
        if self.stagnant_steps > 25:
            reward -= 1.0

        # energy
        reward -= 0.01 * np.sum(np.abs(action))

        reward = np.clip(reward, -15.0, 15.0)

        # termination (fall)
        if abs(theta) > 1.6:
            reward -= 5.0
            return self._get_obs(), reward, True, False, {}

        truncated = self.step_count >= self.max_steps
        return self._get_obs(), reward, False, truncated, {}

    # -------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None

        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = EvoViewer(
                self.sim,
                target_rps=None,
                view_size=(160, 60),
                resolution=(1024, 512),
            )
            self.viewer.track_objects(self.robot_name)

        if self.render_mode == "human":
            self.viewer.render('screen', hide_grid=True)
            return None

        elif self.render_mode == "img":
            return self.viewer.render('img', hide_grid=True)