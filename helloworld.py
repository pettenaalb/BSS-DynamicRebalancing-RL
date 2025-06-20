print("Hello\nRight... python works...\n Now for the hard stuff:")

import torch
import matplotlib.pyplot as plt
import numpy as np
from gymnasium_env.simulator.utils import generate_poisson_events 

print("\nI guess you know how this works\n")
print("Device available:   ")
print("gpu") if torch.cuda.is_available() else print("cpu")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(torch.cuda.current_device())}")


