from evogym.envs.base import Any, BenchmarkBase, Dict, EvoGymBase, EvoViewer, EvoWorld, Optional, Tuple

from stable_baselines3.common.env_util import make_vec_env

import logging
import os
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import torch
import torch as th
from evogym import EvoWorld
from JsonWorldEnv import SimpleTraverseEnv
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

logger = logging.getLogger(__name__)


class RewardDebugCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:

        infos = self.locals.get("infos", [])

        for info in infos:
            if "max_dx" in info:
                self.logger.record("debug/max_dx", info["max_dx"])

            if "max_dy" in info:
                self.logger.record("debug/max_dy", info["max_dy"])

        return True

# ================= SYSTEM STABILITY =================
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

# ================= CONFIG =================
SEED = 84
DEVICE = "cpu"
TOTAL_TIMESTEPS = 1000000
ROBOT_NAME = "robot"
N_ENVS = 3

ROBOT_CSV = r"upsteppers.parquet"
BASE_PATH = r"saved_modelsNEWW"
TRAINING_ENV_LIST = [
    r"env_files\UpStepper-v0.json",
    r"env_files\exampleenvnew.json",
    r"env_files\exampleenv.json",
    # r"env_files\exampleenvold.json",
]

# ================= UTILS =================
def parse_array_blocks(text):
    body = []
    for i in range(len(text)):
        body.append(np.array(text[i]).tolist())
    return np.array(body, dtype=int)

# ================= ROBOT PLACEMENT =================
def safe_add_robot(world, body, connections):
    x, y = 1, 1
    for _ in range(50):
        try:
            world.add_from_array(
                name=ROBOT_NAME,
                structure=body,
                connections=connections,
                x=x,
                y=y,
            )
            return True
        except Exception:
            y += 1
    return False

# ================= ENV FACTORY =================

def make_env_eval(body, connections, json_path, env_class, r_mode):
    def _init():
        env = env_class(
            body=body,
            connections=connections,
            path=json_path,
            render_mode=r_mode,
        )
        env = TimeLimit(env, max_episode_steps=500)
        return Monitor(env)
    return _init

# ================= GIF =================
def save_gif(body, connections, env_class, json_path, model_path, gif_path):

    env = env_class(
        body=body,
        connections=connections,
        path=str(json_path),
        render_mode="img",
    )


    model = PPO.load(model_path, device="cpu")
    obs, info = env.reset()
    frames = []
    try:
        for step in range(800):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render(mode="img")
            frames.append(np.asarray(frame).copy())
            if terminated or truncated:
                break
    finally:
        print("Skipping viewer.close()...", flush=True)
        print("Closing environment...", flush=True)
        try:
            env.close()
        except Exception as e:
            print(f"env.close() failed: {e}", flush=True)
    print(f"Writing {len(frames)} frames", flush=True)
    if not frames:
        raise RuntimeError("No frames generated")
    imageio.mimsave(
        str(gif_path),
        frames,
        fps=24,
    )
    print("GIF WRITTEN", flush=True)
class GIFEvalCallback(EvalCallback):
    def __init__(self, eval_env, body, connections, env_class, json_path, save_path, **kwargs):
        save_path = Path(save_path)
        super().__init__(eval_env, best_model_save_path=save_path, log_path=save_path, **kwargs)
        self.body = body
        self.connections = connections
        self.env_class = env_class
        self.json_path = Path(json_path)
        self.save_path = save_path

    def _on_step(self) -> bool:
        previous_best = self.best_mean_reward
        continue_training = super()._on_step()
        if self.best_mean_reward > previous_best:
            print(f"🏆 New best mean reward: {self.best_mean_reward:.4f}")
            best_model_path = self.save_path / "best_model.zip"
            gif_path = self.save_path / f"best_{self.num_timesteps}.mp4"
            print('before savegif')
            # Execute GIF generation when a new best evaluation model is saved
            try:
                save_gif(
                    self.body,
                    self.connections,
                    self.env_class,
                    self.json_path,
                    best_model_path,
                    gif_path,
                )
            except Exception as e:
                print(e)

        return continue_training

# ================= TRAIN =================
def train_one(body, connections, env_name, env_class, json_paths, idx):
    save_path = os.path.join(BASE_PATH, env_name, f"robot_{idx}")
    os.makedirs(save_path, exist_ok=True)

    try:
        env = SubprocVecEnv([
            make_env_eval(body, connections, json_paths[i % len(json_paths)], env_class, r_mode=None)
            for i in range(N_ENVS)
        ])
    except Exception as e:
        print(f"❌ Skipping robot {idx} (env failed): {e}")
        return

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2.0e-4,
        verbose=1,
        n_steps=256,
        batch_size=4,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        ent_coef=0.01,
        clip_range=0.1,
        tensorboard_log=os.path.join(save_path, "tensorboard"),
    )

    print(f"\n🚀 Training robot {idx} in {env_name}...\n")
    eval_env = SubprocVecEnv(
       [ make_env_eval(body, connections, json_paths[0], env_class, r_mode=None)])
    
    gifcallback = GIFEvalCallback(
        eval_env=eval_env,
        body=body,
        connections=connections,
        env_class=env_class,
        json_path=json_paths[0],
        save_path=save_path,
        eval_freq=10000,
        n_eval_episodes=1,
        deterministic=False,
        verbose=1,
    )
    eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path=save_path,
    log_path=save_path,
    eval_freq=10000,
    n_eval_episodes=1,
    deterministic=False,
    verbose=1,
)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_callback,gifcallback])
    print(f"\n✅ Training finished for robot {idx}\n")
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, "final_model")
    model.save(model_path)
    env.close()
    eval_env.close()

    print(f"✅ Model saved: {model_path}")
    gif_path = os.path.join(save_path, "result.gif")
    save_gif(body, connections, env_class, json_paths[0], model_path, gif_path)
from concurrent.futures import ProcessPoolExecutor


# def run_robot(args):
#     i, env_name, body, connections = args

#     print(f"Starting Row {i}: {env_name}", flush=True)

#     train_one(
#         body,
#         connections,
#         env_name,
#         SimpleTraverseEnv,
#         TRAINING_ENV_LIST,
#         i
#     )

#     save_path = os.path.join(
#         BASE_PATH,
#         env_name,
#         f"robot_{i}"
#     )

#     os.makedirs(save_path, exist_ok=True)

#     # Don't call save_gif here if train_one already does it.
#     # train_one() already saves result.gif after training.

#     print(f"Finished Row {i}: {env_name}", flush=True)


# def main():
#     df = pd.read_parquet(ROBOT_CSV)
#     print("Loaded DF", flush=True)

#     jobs = []

#     for i in range(3):
#         env_name = df.iloc[i]["env_name"]

#         body = parse_array_blocks(df.iloc[i]["body"])
#         connections = parse_array_blocks(df.iloc[i]["connections"])

#         jobs.append(
#             (i, env_name, body, connections)
#         )

#     # 3 robots -> 3 parallel training processes
#     with ProcessPoolExecutor(max_workers=3) as executor:
#         futures = [
#             executor.submit(run_robot, job)
#             for job in jobs
#         ]

#         # Raises exceptions from worker processes
#         for future in futures:
#             future.result()

#     print("All robots finished.", flush=True)



# if __name__ == "__main__":
#     main()



def run_robot(args):
    i, env_name, body, connections = args

    print(f"\nStarting Row {i}: {env_name}", flush=True)

    train_one(
        body,
        connections,
        env_name,
        SimpleTraverseEnv,
        TRAINING_ENV_LIST,
        i
    )

    print(f"Finished Row {i}: {env_name}", flush=True)


def main():
    df = pd.read_parquet(ROBOT_CSV)
    filtered_df = (
        df[df["env_name"] == "UpStepper-v0"]
        .dropna(subset=["body", "connections"])
        .reset_index(drop=True)
    )
    df = filtered_df
    print("Loaded DF", flush=True)

    for i in range(91,92):
        env_name = df.iloc[i]["env_name"]

        print(f"\n{'='*60}", flush=True)
        print(f"Starting robot {i}: {env_name}", flush=True)
        print(f"{'='*60}", flush=True)

        body = parse_array_blocks(df.iloc[i]["body"])
        connections = parse_array_blocks(df.iloc[i]["connections"])

        run_robot(
            (i, env_name, body, connections)
        )

        print(f"\nFinished robot {i}: {env_name}", flush=True)

    print("\nAll robots finished.", flush=True)


if __name__ == "__main__":
    main()