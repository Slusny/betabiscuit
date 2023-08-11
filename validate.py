from run_model import parse_arguments_and_get_agent
from run_model import opponent_action
from run_model import add_derivative
from run_model import parse_arguments_and_get_agent
from run_model import run
from argparse import Namespace

MAX_EPISODES = 500

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

run_args_list_weak = [
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
    "weak_opponent":True,
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
    "weak_opponent":True,
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
    "weak_opponent":True,
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
    "weak_opponent":True,
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
    "weak_opponent":True,
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
    "weak_opponent":True,
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
    "weak_opponent":True,
    "project":False,
    "run_name":"latest",
    "run_id":"latest"
})
]

agents = []
args_list = []
strings = []
run_args_list=[]
for run_args in run_args_list_strong:
    agent,env,args,string = parse_arguments_and_get_agent(run_args)
    agents.append(agent)
    args_list.append(args)
    strings.append(string)
    run_args_list.append(run_args)


agents_w = []
args_list_w = []
strings_w = []
run_args_list_w=[]
for run_args in run_args_list_weak:
    agent,env,args,string = parse_arguments_and_get_agent(run_args)
    agents_w.append(agent)
    args_list_w.append(args)
    strings_w.append(string)
    run_args_list_w.append(run_args)

print("Validate for "+str(MAX_EPISODES)+" episodes against the Strong Opponent")
run(run_args_list,agents,env,args_list,strings)

print("\n -------------------------------- \n")

print("Validate for "+str(MAX_EPISODES)+" episodes against the Weak Opponent")
run(run_args_list,agents_w,env,args_list_w,strings_w)