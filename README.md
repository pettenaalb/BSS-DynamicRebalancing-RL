# Dynamic Rebalancing for Bike Sharing System (BSS) with Reinforcement Learning

This repository contains code and data for implementing a dynamic rebalancing model in a Bike Sharing System (BSS) using Reinforcement Learning (RL). The goal is to optimize bicycle distribution across stations based on demand patterns.

To correctly install the project follow these steps:
0- make shure to have Anaconda (or Miniconda) and python installed
1- create a new enviroment with python=3.11
2- copy the repository from github with the link -> a new folder BSS-DYNAMICREBALANCING-RL will be created with the project inside
3- activate your envirment and navigate on the folder before the one of the project.
4- run $ pip install -e BSS-DYNAMICREBALANCNG-RL -> the project and all required packets will be installed

Note: if you are on windows and torch doesn't detect your Cuda device, install torch+cu118 with
        $ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    reference: https://pytorch.org/get-started/locally/



# ''' cancel this on final version '''
# sudo ssh -N -f -L localhost:2020:localhost:22 -o ProxyCommand="ssh pettenaalb@login.dei.unipd.it nc %h %p" -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" albertopettena@147.162.14.81 -p 22
# sudo ssh -N -f -L localhost:8893:localhost:8895 -o ProxyCommand="ssh pettenaalb@login.dei.unipd.it nc %h %p" -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" albertopettena@147.162.14.81 -p 22