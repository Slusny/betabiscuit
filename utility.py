from datetime import datetime
import os
import pickle
import torch
import wandb
import numpy as np

def save_statistics(savepath,algo,env_name,i_episode,rewards=None,lengths=None,train_iter=None, losses=None, eps="Nan",lr="Nan",seed="Nan"):
    date_str = datetime.today().strftime('%Y-%m-%dT%H.%M')
    savepath_stats = os.path.join(savepath,f'{algo}_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{seed}-{date_str}-stat.pkl')
    with open(savepath_stats, 'wb') as f:
        pickle.dump({"rewards" : rewards, "lengths": lengths, "losses": losses}, f)
                
def wandb_save_model(wandb_run,savepath,notes=""):
    #print("----------- Writing Model to W&B -----------")
    artifact = wandb.Artifact(wandb_run.name + notes + '_model', type='model')
    artifact.add_file(savepath)
    wandb_run.log_artifact(artifact)

def save_checkpoint(torch_state,savepath,algo,env_name,i_episode,wandb_run=None,eps="Nan",train_iter="Nan",lr="Nan",seed="Nan",rewards=None,lengths=None, losses=None,notes=""):
    print("########## Saving a checkpoint... ##########")
    date_str = datetime.today().strftime('%Y-%m-%dT%H.%M')
    savepath_checkpoint = os.path.join(savepath,f'{algo}_{env_name}_{i_episode}-{date_str}.pth')
    torch.save(torch_state, savepath_checkpoint)
    if wandb_run : wandb_save_model(wandb_run,savepath_checkpoint,notes)
    #save_statistics(savepath,algo,env_name,i_episode,rewards,lengths,train_iter, losses, eps,lr,seed)

def restore_from_wandb(str):
    run = wandb.init()
    artifact = run.use_artifact(str, type='model')
    artifact_dir = artifact.download()
    return artifact_dir

def transform_obs(obs,help=False):
    names = ["x pos player one",
            'y pos player one',
            'angle player one',
            'x vel player one',
            'y vel player one',
            'angular vel player one',
            'x player two',
            'y player two',
            'angle player two',
            'y vel player two',
            'y vel player two',
            'angular vel player two',
            'x pos puck',
            'y pos puck',
            'x vel puck',
            'y vel puck',
            'time left player has puck',
            'time left other player has puck'] # 18 with acceleration 24
    limits = np.array([[-3,0], #[4,0] can go behind barrier
                      [-2,2],  #[3,3]
                      [-1,1],  #[-1.25,1.25] can overshoot with momentum
                      [-10,10],
                      [-10,10],
                      [-20,20],
                      [0,3],
                      [-2,2],
                      [-1,1],
                      [-10,10],
                      [-10,10],
                      [-20,20],
                      [-3,3],
                      [-2,2],
                      [-60,60],
                      [-60,60],
                      [0,15],
                      [0,15]])
    limit_range = (limits[:,1] - limits[:,0]).astype(float)
    if help :
        for i in range(18):
            print(names[i]," : ",limits[i])
    else : 
        return ((obs-limits[:,0]) / limit_range)-0.5

# transform_obs("",True)