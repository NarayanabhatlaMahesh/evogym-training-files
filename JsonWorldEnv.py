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
        self.max_dx,self.max_dy=0,0

        # init sim
        super().__init__(world=self.world, render_mode=render_mode, render_options=render_options)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 4

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=float)

    def step(self, action):
        # collect pre step information
        self.total_steps += 1
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")
        for _ in range(1):
            done = super().step({'robot': action})
        
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_ort_final = self.object_orientation_at_time(self.get_time(), "robot")
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            np.array([robot_ort_final]),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["ground"], self.sight_dist),
            ))
        
        # compute reward
        x,y=self.get_pos_com_obs(self.robot_name)
        dx = x - self.prev_x
        dy = y - self.prev_y
        reward = 0.0
        reward = dx
             
        self.prev_x, self.prev_y = x,y
    
        #error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        #check termination conditions
        com_pos = np.mean(robot_pos_final, axis=1)
        if com_pos[0] > 69 * self.VOXEL_SIZE:
            reward += 3.0
            done = True
    
        if robot_ort_final > (math.pi/2 - math.pi/12) and robot_ort_final < (3*math.pi/2 + math.pi/12):
            done = True
            reward -= 3.0
    

        # observation, reward, has simulation met termination conditions, truncated, debugging info
        return obs, reward, done, False, {}


    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        super().reset(seed=seed, options=options)
        self.window=0
        x, y = self.get_pos_com_obs(self.robot_name)
        self.prev_x, self.prev_y = x, y
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


