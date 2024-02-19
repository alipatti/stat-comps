import torch
from pathlib import Path

DATA_PATH = Path("../data")
MODEL_CHECKPOINT_PATH = Path("../models")
IN_PATH = DATA_PATH / "nba-raw"
OUT_PATH = DATA_PATH / "nba-tensors"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)

if __name__ == "__main__":
    print(f"Using {DEVICE} device")
