import os
import pandas as pd
import numpy as np

from evogym import EvoWorld
from JsonWorldEnv import SimpleTraverseEnv

# Import save_gif from your existing training file
from TrainNewRobot import save_gif


# ================= CONFIG =================

ROBOT_NAME = "robot"


BASE_PATH = r"C:\Users\numam\EALLMs\EnvGenPrmtOpt\saved_modelsNEWW"

ROBOT_INDEX = 1


ROBOT_CSV = r"C:\Users\numam\EALLMs\EnvGenPrmtOpt\upsteppers.parquet"

MODEL_PATH = rf"C:\Users\numam\EALLMs\EnvGenPrmtOpt\saved_modelsNEWW\UpStepper-v0\robot_{ROBOT_INDEX}\best_model.zip"

JSON_PATH = r"C:\Users\numam\Desktop\exampleenvnew.json"

GIF_PATH = rf"C:\Users\numam\EALLMs\EnvGenPrmtOpt\saved_modelsNEWW\UpStepper-v0\robot_{ROBOT_INDEX}\best_model.gif"


TRAINING_ENV_LIST = [
    r"C:\Users\numam\Desktop\exampleenvnew.json",
    r"C:\Users\numam\Desktop\exampleenvnew.json",
    r"C:\Users\numam\Desktop\exampleenvnew.json",
]

# Robot row used during training

# Environment used during training
ENV_CLASS = SimpleTraverseEnv


# ================= UTILS =================

def parse_array_blocks(text):
    body = []

    for i in range(len(text)):
        body.append(np.array(text[i]).tolist())

    return np.array(body, dtype=int)


# ================= GENERATE GIF =================

def generate_gif():

    # Load the exact same training file
    df = pd.read_parquet(ROBOT_CSV)


    # Get the exact robot body
    body = parse_array_blocks(
        df.iloc[ROBOT_INDEX]["body"]
    )

    # Get the exact connections
    connections = parse_array_blocks(
        df.iloc[ROBOT_INDEX]["connections"]
    )

    # Get the environment name used during training
    env_name = df.iloc[ROBOT_INDEX]["env_name"]

    # This must match the directory created by train_one()
    save_path = os.path.join(
        BASE_PATH,
        env_name,
        f"robot_{ROBOT_INDEX}"
    )

    # Model produced by your existing training code
    model_path = os.path.join(
        save_path,
        "best_model.zip"
    )

    # Output GIF
    gif_path = os.path.join(
        save_path,
        "result_new.gif"
    )

    # Check model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    # Check environment exists
    if not os.path.exists(TRAINING_ENV_LIST[0]):
        raise FileNotFoundError(
            f"Environment JSON not found:\n"
            f"{TRAINING_ENV_LIST[0]}"
        )

    # Call your EXISTING save_gif()
    save_gif(
        body=body,
        connections=connections,
        env_class=ENV_CLASS,
        json_path=TRAINING_ENV_LIST[2],
        model_path=model_path,
        gif_path=gif_path,
    )

    print("\n✅ GIF generation completed.")

if __name__ == "__main__":
    generate_gif()