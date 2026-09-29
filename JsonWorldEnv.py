import os
import time
import gymnasium as gym
import numpy as np
from evogym.envs.base import Any, BenchmarkBase, Dict, EvoGymBase, EvoWorld, Optional, EvoViewer, Tuple
import math
from gymnasium import spaces



class StairsBase(BenchmarkBase):
    
    def __init__(
        self,
        world: EvoWorld,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        super().__init__(world=world, render_mode=render_mode, render_options=render_options)

    def get_reward(self, robot_pos_init, robot_pos_final):
        
        robot_com_pos_init = np.mean(robot_pos_init, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)

        reward = (robot_com_pos_final[0] - robot_com_pos_init[0])
        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        super().reset(seed=seed, options=options)

        # observation
        robot_ort = self.object_orientation_at_time(self.get_time(), "robot")
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            np.array([robot_ort]),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["ground"], self.sight_dist),
            ))

        return obs, {}






class SimpleTraverseEnv(StairsBase):

    def __init__(
        self,
        body: np.ndarray,
        connections: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None
    ):
        self.robot_name='robot'

        self.total_steps = 0
        self.prev_x, self.prev_y = 0,0

        self.window=0
        self.prev_grounded=False

        # make world
        self.world = EvoWorld.from_json(path)
        x,y=1,1
        Executed=False
        while(not Executed):
            try:
                self.world.add_from_array('robot', body, x, y, connections=connections)
                Executed=not Executed
            except Exception:
                y+=1
        self.step_count =0

        self.min_reward = float("inf")
        self.max_reward = float("-inf")
        
        self.dx = 0
        self.dy = 0
        
        self.stage = 0
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.hop_count = 0
        self.hop_start_x = 0.0
        self.max_dx,self.max_dy=0,0

        # init sim
        super().__init__(world=self.world, render_mode=render_mode, render_options=render_options)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 4

        self.action_space = spaces.Box(low= 0.01, high=1.99, shape=(num_actuators,), dtype=float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=float)

    def check_ground_contact(self, dy, robot_pos):
        lowest_y = np.min(robot_pos[1, :])
        tolerance = 0.1 * self.VOXEL_SIZE
        is_near_ground = lowest_y <= tolerance
        is_grounded = is_near_ground and dy <= 0
        return is_grounded
 
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        super().reset(seed=seed, options=options)
        self.window=0
        x, y = self.get_pos_com_obs(self.robot_name)
        self.prev_x, self.prev_y = x, y
        self.hop_start_x = x
        self.prev_grounded = self.check_ground_contact(
            0, self.object_pos_at_time(self.get_time(), self.robot_name)
        )

        robot_ort = self.object_orientation_at_time(
            self.get_time(), "robot")
        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            np.array([robot_ort]),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["ground"], self.sight_dist),
        ))

        return obs, {}

    def render(self, mode):
            if mode is None:
                return None

            if not hasattr(self, "viewer") or self.viewer is None:
                self.viewer = EvoViewer(
                    self.sim,
                    target_rps=None,
                    view_size=(80, 20),
                    resolution=(1680, 400),
                    pos = (20,5)
                )
                # self.viewer.track_objects(self.robot_name)

            if mode == "human":
                self.viewer.render('screen', hide_grid=True)
                return None

            elif mode == "img":
                return self.viewer.render('img', hide_grid=True)

    def close(self):
        try:
            if hasattr(self, "_viewer"):
                self._viewer = None
        except Exception:
            pass
        try:
            super().close()
        except Exception:
            pass

    def set_curriculum_stage(self, stage):
        self.stage = min(3, max(0, int(stage)))


    def compute_reward(self, dx, dy, grounded, goal, fallen, action):
        takeoff = self.prev_grounded and not grounded
        landing = not self.prev_grounded and grounded

        reward = 10.0 * dx - 0.01

        reward += [0.5, 1.0, 1.5, 2.0][self.stage] * max(dy, 0)

        if takeoff:
            self.hop_start_x = self.prev_x

        if landing:
            hop_progress = max(self.prev_x - self.hop_start_x, 0)
            reward += min(2.0 * hop_progress, 5.0)

        if goal:
            reward += 100.0

        if fallen:
            reward -= 50.0

        self.prev_grounded = grounded
        return reward


    def step(self, action):
        pos_init = self.object_pos_at_time(self.get_time(), self.robot_name)
        com_init = np.mean(pos_init, axis=1)

        super().step({self.robot_name: action})

        pos_final = self.object_pos_at_time(self.get_time(), self.robot_name)
        com_final = np.mean(pos_final, axis=1)

        dx = com_final[0] - com_init[0]
        dy = com_final[1] - com_init[1]

        self.dx, self.dy = dx, dy
        self.max_dx = max(self.max_dx, dx)
        self.max_dy = max(self.max_dy, dy)

        grounded = self.check_ground_contact(dy, pos_final)
        orientation = self.object_orientation_at_time(self.get_time(), self.robot_name)

        fallen = com_final[1] < -self.VOXEL_SIZE
        goal = com_final[0] > 69 * self.VOXEL_SIZE

        reward = self.compute_reward(dx, dy, grounded, goal, fallen, action)

        self.prev_x, self.prev_y = com_final
        self.total_steps += 1

        obs = np.concatenate((
            self.get_vel_com_obs(self.robot_name),
            np.array([orientation]),
            self.get_relative_pos_obs(self.robot_name),
            self.get_floor_obs(self.robot_name, ["ground"], self.sight_dist),
        ))

        return obs, reward, fallen or goal, False, {
            "x": com_final[0],
            "y": com_final[1],
            "stage": self.stage,
            "grounded": grounded,
            "goal": goal,
        }



