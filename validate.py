from run_model import parse_arguments_and_get_agent
from run_model import opponent_action
from run_model import add_derivative
from run_model import parse_arguments_and_get_agent
from run_model import run

run_args = {
    "validate":True,
    "local_config":"./wandb/latest-run/config.yaml",
    "boostrap_local":True,
    "bootstrap":"validation/agents/TD3.pth"
}
agent,env,args,string = parse_arguments_and_get_agent(run_args)
run(run_args,[agent],env,[args],[string])