import torch
from pathlib import Path

DATA_ROOT = Path("../data")
CHECKPOINT_ROOT = Path("../models")

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)

if __name__ == "__main__":
    print(f"Using {DEVICE} device")
