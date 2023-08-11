from run_model import parse_arguments_and_get_agent
from run_model import opponent_action
from run_model import add_derivative
from run_model import parse_arguments_and_get_agent
from run_model import run
from argparse import Namespace

MAX_EPISODES = 100

run_args_list_strong = [
Namespace(**{
    "validate":True,
    "local_config":"validation/agents/TD3.json",
    "bootstrap_local":True,
    "bootstrap":"validation/agents/TD3.pth",
    "vir":False,
    "max_episodes": MAX_EPISODES,
    "max_timesteps": 250,
    "sleep": 0.,
    "legacy":False,
    "action_n":4,
    "validate":True,
    "weak_opponent":False,
    "project":False,
    "run_name":"latest",
    "run_id":"latest"
}),
Namespace(**{
    "validate":True,
    "local_config":"validation/agents/TD3_tournament.json",
    "bootstrap_local":True,
    "bootstrap":"validation/agents/TD3_tournament.pth",
    "vir":False,
    "max_episodes": MAX_EPISODES,
    "max_timesteps": 250,
    "sleep": 0.,
    "legacy":False,
    "action_n":4,
    "validate":True,
    "weak_opponent":False,
    "project":False,
    "run_name":"latest",
    "run_id":"latest"
}),
Namespace(**{
    "validate":True,
    "local_config":"validation/agents/DDQN_acceleration.json",
    "bootstrap_local":True,
    "bootstrap":"validation/agents/DDQN_acceleration.pth",
    "vir":False,
    "max_episodes": MAX_EPISODES,
    "max_timesteps": 250,
    "sleep": 0.,
    "legacy":False,
    "action_n":4,
    "validate":True,
    "weak_opponent":False,
    "project":False,
    "run_name":"latest",
    "run_id":"latest"
}),
Namespace(**{
    "validate":True,
    "local_config":"validation/agents/DDQN_tournament.json",
    "bootstrap_local":True,
    "bootstrap":"validation/agents/DDQN_tournament.pth",
    "vir":False,
    "max_episodes": MAX_EPISODES,
    "max_timesteps": 250,
    "sleep": 0.,
    "legacy":False,
    "action_n":4,
    "validate":True,
    "weak_opponent":False,
    "project":False,
    "run_name":"latest",
    "run_id":"latest"
}),
Namespace(**{
    "validate":True,
    "local_config":"validation/agents/DDPG.json",
    "bootstrap_local":True,
    "bootstrap":"validation/agents/DDPG.pth",
    "vir":False,
    "max_episodes": MAX_EPISODES,
    "max_timesteps": 250,
    "sleep": 0.,
    "legacy":True,
    "action_n":8,
    "validate":True,
    "weak_opponent":False,
    "project":False,
    "run_name":"latest",
    "run_id":"latest"
}),
Namespace(**{
    "validate":True,
    "local_config":"validation/agents/DDPG_BC.json",
    "bootstrap_local":True,
    "bootstrap":"validation/agents/DDPG_BC.pth",
    "vir":False,
    "max_episodes": MAX_EPISODES,
    "max_timesteps": 250,
    "sleep": 0.,
    "legacy":True,
    "action_n":8,
    "validate":True,
    "weak_opponent":False,
    "project":False,
    "run_name":"latest",
    "run_id":"latest"
}),
Namespace(**{
    "validate":True,
    "local_config":"validation/agents/DDPG_dense.json",
    "bootstrap_local":True,
    "bootstrap":"validation/agents/DDPG_dense.pth",
    "vir":False,
    "max_episodes": MAX_EPISODES,
    "max_timesteps": 250,
    "sleep": 0.,
    "legacy":True,
    "action_n":8,
    "validate":True,
    "weak_opponent":False,
    "project":False,
    "run_name":"latest",
    "run_id":"latest"
})
]

agents = []
args_list = []
strings = []
for run_args in run_args_list_strong:
    agent,env,args,string = parse_arguments_and_get_agent(run_args)
    agents.append(agent)
    args_list.append(args)
    strings.append(string)

print("Validate for "+str(MAX_EPISODES)+" episodes against the Strong Opponent")
run(run_args,agents,env,args_list,strings)


agents = []
args_list = []
strings = []
for run_args in run_args_list_weak:
    agent,env,args,string = parse_arguments_and_get_agent(run_args)
    agents.append(agent)
    args_list.append(args)
    strings.append(string)

print("Validate for "+str(MAX_EPISODES)+" episodes against the Weak Opponent")
run(run_args,agents,env,args_list,strings)