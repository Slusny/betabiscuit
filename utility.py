from datetime import datetime
import os
import pickle
import torch
import wandb


def save_statistics(savepath,algo,env_name,i_episode,rewards=None,lengths=None,train_iter=None, losses=None, update_target_every=None, eps="Nan",lr="Nan",seed="Nan",):
    date_str = datetime.today().strftime('%Y-%m-%dT%H.%M')
    savepath_stats = os.path.join(savepath,f'{algo}_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{seed}-{date_str}-stat.pkl')
    with open(savepath_stats, 'wb') as f:
        pickle.dump({"rewards" : rewards, "lengths": lengths, "eps": self._eps, "train": train_iter,
                    "lr": lr, "update_every":update_target_every , "losses": losses}, f)
                
def wandb_save_model(wandb_run,savepath):
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(savepath)
    wandb_run.log_artifact(artifact)

def save_checkpoint(torch_state,savepath,algo,env_name,i_episode,wandb=True,wandb_run=None,eps="Nan",train_iter="Nan",lr="Nan",seed="Nan"):
    print("########## Saving a checkpoint... ##########")
    date_str = datetime.today().strftime('%Y-%m-%dT%H.%M')
    savepath_checkpoint = os.path.join(savepath,f'{algo}_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{seed}-{date_str}.pth')
    torch.save(torch_state, savepath_checkpoint)
    if wandb : wandb_save_model(wandb_run,savepath_checkpoint)
    save_statistics()