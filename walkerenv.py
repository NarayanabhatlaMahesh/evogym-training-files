import gymnasium as gym
import numpy as np
from evogym.envs.base import EvoGymBase
from evogym import EvoViewer


class JsonWorldEnv(EvoGymBase):
    metadata = {"render_modes": ["human", "img"]}

    def __init__(
        self,
        world,
        robot_name,
        total_timesteps,
        render_mode=None,
    ):
        super().__init__(world=world, render_mode=render_mode)

        self.robot_name = robot_name

        # =======================
        # TIMING
        # =======================
        self.sim_steps = 8
        self.max_steps = 1000
        self.step_count = 0

        self.total_timesteps = total_timesteps
        self.global_step = 0

        # =======================
        # SPACES
        # =======================
        self.n_act = len(self.get_actuator_indices(robot_name))

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_act + 1,),
            dtype=np.float32
        )

        # vel_x, vel_y, height, theta, angular velocity
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32
        )

        # =======================
        # STATE
        # =======================
        self.prev_x = self.prev_y = 0.0
        self.prev_theta = 0.0
        self.ground_y = None

        self.filtered_action = None
        self.noise_state = None

        self.ground_steps = 0
        self.stagnant_steps = 0

    # -------------------------------------------------
    def _get_obs(self):
        vel = self.get_vel_com_obs(self.robot_name)
        pos = self.get_pos_com_obs(self.robot_name)

        theta = self.get_ort_obs(self.robot_name)[0]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        ang_vel = theta - self.prev_theta

        return np.array(
            [
                vel[0],
                vel[1],
                pos[1] - self.ground_y,
                theta,
                ang_vel
            ],
            dtype=np.float32
        )

    # -------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        pos = self.get_pos_com_obs(self.robot_name)
        vel = self.get_vel_com_obs(self.robot_name)

        self.prev_x, self.prev_y = pos
        self.ground_y = pos[1]

        theta = self.get_ort_obs(self.robot_name)[0]
        self.prev_theta = (theta + np.pi) % (2 * np.pi) - np.pi

        self.filtered_action = None
        self.noise_state = None

        self.step_count = 0
        self.ground_steps = 0
        self.stagnant_steps = 0

        return self._get_obs(), {}

    # -------------------------------------------------
    def step(self, action):
        self.step_count += 1
        self.global_step += 1

        # =======================
        # GLOBAL PROGRESS
        # =======================
        p = min(1.0, self.global_step / self.total_timesteps)
        p2 = p * p

        # =======================
        # ACTION PROCESSING
        # =======================
        raw = np.clip(action[:-1], -1.0, 1.0)
        speed = np.clip(action[-1], -1.0, 1.0)

        speed_bias = 0.6 * (1.0 - p)
        speed = np.clip(speed + speed_bias, -1.0, 1.0)

        amp = 1.0 - 0.25 * p2
        target = 1.0 + amp * raw

        if self.noise_state is None:
            self.noise_state = np.random.randn(self.n_act)

        self.noise_state = 0.93 * self.noise_state + 0.03 * np.random.randn(self.n_act)
        target += (0.9 * (1.0 - p2) + 0.2) * self.noise_state

        target = np.clip(
            target,
            0.2 + p * 0.5,
            1.8 - p * 0.6
        )

        alpha = (0.8 * (1 - p) + 0.5 * p)
        alpha *= 0.3 + 0.7 * (speed + 1.0) / 2.0

        self.filtered_action = (
            target if self.filtered_action is None
            else (1 - alpha) * self.filtered_action + alpha * target
        )

        # =======================
        # PHYSICS
        # =======================
        for _ in range(self.sim_steps):
            super().step({self.robot_name: self.filtered_action})

        # =======================
        # STATE UPDATE
        # =======================
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
        self.ground_steps = 0 if airborne else self.ground_steps + 1
        self.stagnant_steps = self.stagnant_steps + 1 if abs(dx) < 0.03 else 0

        # =======================
        # REWARD
        # =======================
        reward = 0.0
        # Forward / backward
        forward_reward = 2.5 * dx
        backward_penalty = -4.0 * max(-dx, 0.0)

        # Stability
        tilt_penalty = -0.6 * (theta ** 2)
        ang_vel_penalty = -0.2 * (ang_vel ** 2)

        # Smooth stagnation penalty
        stagnation_penalty = -0.15 * self.stagnant_steps

        # Small alive penalty
        alive_penalty = -0.1

        # -----------------------
        # PROGRESSIVE BONUSES
        # -----------------------
        p = self.global_step / self.total_timesteps

        # 1. Early exploration (encourage movement)
        move_bonus = (1.0 - p) * 1.0 * max(dx, 0.0)

        # 2. Mid training (smooth motion)
        smooth_bonus = 0.0
        if p > 0.3:
            smooth_bonus = 0.5 * (1.0 - abs(dy))  # discourage vertical bouncing

        # 3. Late training (efficient + stable forward motion)
        efficiency_bonus = 0.0
        if p > 0.6 and abs(theta) < 0.3:
            efficiency_bonus = 0.5 * max(dx, 0.0)

        # 4. Optional controlled jumping (small, non-dominant)
        jump_bonus = 0.0
        if p > 0.6 and airborne and dy > 0.05 and dx > 0.02:
            jump_bonus = 1.5 * dy

        # -----------------------
        # COMBINE
        # -----------------------
        reward = (
            forward_reward +
            backward_penalty +
            tilt_penalty +
            ang_vel_penalty +
            stagnation_penalty +
            alive_penalty +
            move_bonus +
            smooth_bonus +
            efficiency_bonus +
            jump_bonus
        )

        # -----------------------
        # FALL TERMINATION
        # -----------------------
        if theta < -1.2:
            reward -= 10.0
            return self._get_obs(), np.clip(reward, -10.0, 8.0), True, False, {}

        # -----------------------
        # FINAL CLIP
        # -----------------------
        reward = np.clip(reward, -10.0, 8.0)

        truncated = self.step_count >= self.max_steps
        return self._get_obs(), reward, False, truncated, {}

    def render(self):
        if self.render_mode is None:
            return None

        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = EvoViewer(
                self._sim,
                target_rps=None,
                view_size=(160, 60),
                resolution=(2048, 512),
            )
            self.viewer.track_objects(self.robot_name)

        if self.render_mode == "human":
            self.viewer.render('screen', hide_grid=True)
            return None

        elif self.render_mode == "img":
            return self.viewer.render('img', hide_grid=True)
    
    def get_sim(self):
        return self.sim