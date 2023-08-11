from run_model import parse_arguments_and_get_agent
from run_model import opponent_action
from run_model import add_derivative
from run_model import parse_arguments_and_get_agent
from run_model import run
from argparse import Namespace

run_args = {
    "validate":True,
    "local_config":"validation/agents/TD3.json",
    "bootstrap_local":True,
    "bootstrap":"validation/agents/TD3.pth",
    "vir":False,
    "max_episodes": 100,
    "max_timesteps": 250,
    "sleep": 0.,
    "legacy":False,
    "action_n":4,
    "validate":True,
    "weak_opponent":False,
    "project":False,
    "run_name":"latest",
    "run_id":"latest"
}
run_args = Namespace(**run_args)
agent,env,args,string = parse_arguments_and_get_agent(run_args)
run(run_args,[agent],env,[args],[string])