#!/bin/bash

apt-get install swig
pip install -r requirements_pip.txt 
python -m pip install git+https://github.com/martius-lab/laser-hockey-env.git
python main.py --algo ddpg --env pendulum --wandb