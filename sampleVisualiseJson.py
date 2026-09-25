import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import re
import numpy as np
import pandas as pd
import imageio

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from evogym import EvoWorld
from crawlertraining.JsonWorldEnv import JsonWorldEnv


# ================= CONFIG =================

BASE_PATH = r"C:\Users\numam\EALLMs\EnvGenPrmtOpt\saved_modelsNEWW"
ROBOT_CSV = r"C:\Users\numam\EALLMs\EnvGenPrmtOpt\robot_body_structures.csv"
JSON_WORLD_PATH = r"C:\Users\numam\Desktop\exampleenv.json"

ENV_NAME = "ObstacleTraverser-v0"

ROBOT_NAME = "robot"
MAX_STEPS = 500


# ================= ROBOT =================

def parse_array_blocks(text):
    arrays = re.findall(r'array\(\[([^\]]+)\]\)', text)
    return np.array([np.fromstring(a, sep=',', dtype=int) for a in arrays])


def load_ranked_robots(csv_path, env_name):
    df = pd.read_csv(csv_path)
    df = df[df['env_name'] == env_name]
    df = df.sort_values(by='reward', ascending=False).reset_index(drop=True)
    return df


# ================= ENV =================

def make_env(body, connections):
    def _init():
        world = EvoWorld.from_json(JSON_WORLD_PATH)

        world.add_from_array(
            name=ROBOT_NAME,
            structure=body,
            connections=connections,
            x=3,
            y=3
        )

        return JsonWorldEnv(
            world=world,
            robot_name=ROBOT_NAME,
            total_timesteps=MAX_STEPS,
            render_mode="img"
        )

    return _init


# ================= RENDER =================

def render_robot(model_path, vec_path, body, connections, output_path):

    env = DummyVecEnv([make_env(body, connections)])
    env.envs[0].render_mode = "img"

    # VecNormalize
    if os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False
        print("✅ Loaded VecNormalize")
    else:
        print("⚠️ VecNormalize not found")

    model = PPO.load(model_path, device="cpu")

    obs = env.reset()
    frames = []

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        frame = env.envs[0].render()
        if frame is not None:
            frames.append(frame)

        if dones[0]:
            break

    env.close()

    if frames:
        imageio.mimsave(output_path, frames, fps=24)
        print(f"🎬 Saved → {output_path}")
    else:
        print("❌ No frames")


# ================= MAIN =================

def main():

    df = load_ranked_robots(ROBOT_CSV, ENV_NAME)

    env_path = os.path.join(BASE_PATH, ENV_NAME)

    robot_folders = sorted([
        d for d in os.listdir(env_path)
        if d.startswith("robot_")
    ])

    print(f"\n🔥 Found {len(robot_folders)} trained robots\n")

    for folder in robot_folders:

        idx = int(folder.split("_")[-1])
        robot_dir = os.path.join(env_path, folder)

        model_path = os.path.join(robot_dir, "final_model.zip")
        vec_path = os.path.join(robot_dir, "vecnormalize.pkl")
        gif_path = os.path.join(robot_dir, "render.gif")

        if not os.path.exists(model_path):
            print(f"❌ Missing model: {model_path}")
            continue

        if idx >= len(df):
            print(f"⚠️ CSV missing row for robot_{idx}")
            continue

        print(f"\n🚀 Rendering {folder}")

        body = parse_array_blocks(df.iloc[idx]["body"])
        connections = parse_array_blocks(df.iloc[idx]["connections"])

        render_robot(model_path, vec_path, body, connections, gif_path)


if __name__ == "__main__":
    main()