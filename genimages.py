import os
import numpy as np
import pandas as pd
from PIL import Image

# Import your environment class
from crawlertraining.JsonWorldEnv import SimpleTraverseEnv


DATASET = r"C:\Users\numam\EALLMs\EnvGenPrmtOpt\upsteppers.parquet"

# Your actual UpStepper world JSON
WORLD_PATH = r"C:\Users\numam\Desktop\UpStepper-v0.json"

OUTPUT_DIR = "robot_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_parquet(DATASET)

filtered_df = (
    df[df["env_name"] == "UpStepper-v0"]
    .dropna(subset=["body", "connections"])
    .reset_index(drop=True)
)

print("Number of robots:", len(filtered_df))


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def convert_array(x):
    """
    Convert HuggingFace/PyArrow nested sequences
    into a proper numpy int64 array.
    """
    return np.array(x.tolist() if hasattr(x, "tolist") else x, dtype=np.int64)


# ---------------------------------------------------------
# Render
# ---------------------------------------------------------

for i in range(10):
    row = df.iloc[i]

    print(
        f"i={i}, "
        f"env={row['env_name']}, "
        f"reward={row['reward']}"
    )


    try:

        # Convert body
        body = convert_array(row["body"])

        # Convert connections
        connections = convert_array(row["connections"])

        print("  body:", body.shape, body.dtype)
        print("  connections:", connections.shape, connections.dtype)

        # -------------------------------------------------
        # Create environment
        # -------------------------------------------------

        env = SimpleTraverseEnv(
            body=body,
            connections=connections,
            render_mode=None,
            path=WORLD_PATH
        )

        # -------------------------------------------------
        # Render
        # -------------------------------------------------

        img = env.render("img")

        if img is None:
            raise RuntimeError("env.render('img') returned None")

        # Make sure image is uint8
        img = np.asarray(img)

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # -------------------------------------------------
        # Save
        # -------------------------------------------------

        filename = os.path.join(
            OUTPUT_DIR,
            f"{i}.png"
        )

        Image.fromarray(img).save(filename)

        print(f"  Saved: {filename}")

        env.close()

    except Exception as e:

        print(f"\nERROR on robot {i}")
        print("Exception:", repr(e))

        # VERY IMPORTANT:
        # print the actual traceback
        import traceback
        traceback.print_exc()

        print("-" * 60)

print("\nDone!")