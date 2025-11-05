print("Hello ...   Right, python works...\nNow for the hard stuff:")

print("\n Loading numpy, matplotlib and torch:")
import torch
import matplotlib.pyplot as plt
import numpy as np
print(" OK ")


print("\nI guess you know how this works\n")
print("Device available:   ")
print("gpu") if torch.cuda.is_available() else print("cpu")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(" OK ")
else :
    print("You missed some CUDA requiremets. The simulation will run painfully slow.")

print("\nAnd finally the enviroment: ")
from gymnasium_env.simulator.utils import generate_poisson_events 
print(" OK ")

print("\nWell done")
