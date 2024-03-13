import argparse
from collections import defaultdict
import os
import pickle as pkl
from pyexpat import model
import shutil
import json

import pandas as pd
from detectron2.layers.mask_ops import paste_mask_in_image_old
import torch
import numpy as np


from data_loader import get_dataloaders
from diffusion.resample import create_named_schedule_sampler
from tqdm import tqdm

from models import FaceDiff, FaceDiffBeat, FaceDiffDamm
from utils import *
# add tensorboard support
import wandb

def trainer_diff(config, train_loader, dev_loader, model, diffusion, optimizer, epoch=100, device="cuda"):
    
    train_losses = []
    val_losses = []
    full_save_path = model_dir_init(config)
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
    train_subjects_list = [i for i in config["train_subjects"].split(" ")]

    iteration = 0

    for e in range(epoch + 1):
        loss_log = []
        model.train()
        # use tqdm for progress bar, iterate through all training samples
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()
        
        # iterate through all training samples
        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            # batch size of one, I guess
            iteration += 1
            if config["dataset"] == 'meads': # the flame parameters comes in a pickle of dict, so it need to be unpacked 
                vertice = str(vertice[0]) # vertex path
                all_data = pkl.load(open(vertice, "rb")) # load from pickle
                exp = all_data["exp"][0, :, :50]
                jaw = all_data["jaw"][0]
                vertice = np.concatenate([exp, jaw], axis=1)
                vertice = vertice.astype(np.float32)
                vertice = torch.from_numpy(vertice)
                # template file is just all zero for this
                template = torch.zeros(vertice.shape).float()[0:1]
            else:
                vertice = str(vertice[0]) # vertex path
                vertice = np.load(vertice, allow_pickle=True) # np and load from pickle 
                vertice = vertice.astype(np.float32)
                vertice = torch.from_numpy(vertice)
            # for vocaset reduce the frame rate from 60 to 30
            if config["dataset"] == 'vocaset':
                vertice = vertice[::2, :]
            vertice = torch.unsqueeze(vertice, 0)
            t, weights = schedule_sampler.sample(1, torch.device(device))

            audio, vertice = audio.to(device=device), vertice.to(device=device)
            template, one_hot = template.to(device=device), one_hot.to(device=device)
            loss = diffusion.training_losses(
                model,
                x_start=vertice,
                t=t,
                model_kwargs={
                    "cond_embed": audio,
                    "one_hot": one_hot,
                    "template": template,
                }
            )['loss']
            loss = torch.mean(loss)
            loss.backward()
            loss_log.append(loss.item())

            if i % config["gradient_accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
                del audio, vertice, template, one_hot
                torch.cuda.empty_cache()

            pbar.set_description(
                "(Epoch {}, iteration {}) TRAIN LOSS:{:.8f}".format((e + 1), iteration, np.mean(loss_log)))

        train_losses.append(np.mean(loss_log))
        if config["wandb"]:
            wandb.log({"train_loss": np.mean(loss_log)})

        valid_loss_log = []
        model.eval()
        for audio, vertice, template, one_hot_all, file_name in dev_loader:
            # to gpu
            if config["dataset"] == 'meads': # the flame parameters comes in a pickle of dict, so it need to be unpacked 
                vertice = str(vertice[0]) # vertex path
                all_data = pkl.load(open(vertice, "rb")) # load from pickle
                exp = all_data["exp"][0, :, :50]
                jaw = all_data["jaw"][0]
                vertice = np.concatenate([exp, jaw], axis=1)
                vertice = vertice.astype(np.float32)
                vertice = torch.from_numpy(vertice)
                # template file is just all zero for this
                template = torch.zeros(vertice.shape).float()[0:1]
            else:
                vertice = str(vertice[0]) # vertex path
                vertice = np.load(vertice, allow_pickle=True) # np and load from pickle 
                vertice = vertice.astype(np.float32)
                vertice = torch.from_numpy(vertice)

            # for vocaset reduce the frame rate from 60 to 30
            if config['dataset'] == 'vocaset':
                vertice = vertice[::2, :]
            vertice = torch.unsqueeze(vertice, 0)
            # sample a random time step
            t, weights = schedule_sampler.sample(1, torch.device(device))

            audio, vertice = audio.to(device=device), vertice.to(device=device)
            template, one_hot_all = template.to(device=device), one_hot_all.to(device=device)

            train_subject = file_name[0].split("_")[0]
            if train_subject in train_subjects_list:
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:, iter, :]

                loss = diffusion.training_losses(
                    model,
                    x_start=vertice,
                    t=t,
                    model_kwargs={
                        "cond_embed": audio,
                        "one_hot": one_hot,
                        "template": template,
                    }
                )['loss']

                loss = torch.mean(loss)
                valid_loss_log.append(loss.item())
            else:
                for iter in range(one_hot_all.shape[-1]):
                    one_hot = one_hot_all[:, iter, :]
                    loss = diffusion.training_losses(
                        model,
                        x_start=vertice,
                        t=t,
                        model_kwargs={
                            "cond_embed": audio,
                            "one_hot": one_hot,
                            "template": template,
                        }
                    )['loss']

                    loss = torch.mean(loss)
                    valid_loss_log.append(loss.item())

        current_loss = np.mean(valid_loss_log)
        if config["wandb"]:
            wandb.log({"val_loss": current_loss})
        val_losses.append(current_loss)
        if e == config["max_epoch"] or e % 10 == 0 and e != 0 or e == 1: # save every 10 epoches
            torch.save(model.state_dict(), os.path.join(full_save_path, f'{config["model"]}_{config["dataset"]}_{e}.pth'))
            plot_losses(train_losses, val_losses, os.path.join(full_save_path, f"losses_{config['model']}_{config['dataset']}"))
        print("epcoh: {}, current loss:{:.8f}".format(e + 1, current_loss))

    plot_losses(train_losses, val_losses, os.path.join(full_save_path, f"losses_{config['model']}_{config['dataset']}"))

    return model

@torch.no_grad()
def test_diff(config, model, test_loader, epoch, diffusion, device="cuda"):
    result_path = os.path.join(config["result_path"])
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)
    full_save_path = model_dir_init(config)
    train_subjects_list = [i for i in config["train_subjects"].split(" ")]

    model.load_state_dict(torch.load(os.path.join(full_save_path, f'{config["model"]}_{config["dataset"]}_{epoch}.pth')))
    model = model.to(torch.device(device))
    model.eval()

    sr = 16000
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        if config["dataset"] == 'meads': # the flame parameters comes in a pickle of dict, so it need to be unpacked 
            vertice = str(vertice[0]) # vertex path
            all_data = pkl.load(open(vertice, "rb")) # load from pickle
            exp = all_data["exp"][0, :, :50]
            jaw = all_data["jaw"][0]
            vertice = np.concatenate([exp, jaw], axis=1)
            vertice = vertice.astype(np.float32)
            vertice = torch.from_numpy(vertice)
            # template file is just all zero for this
            template = torch.zeros(vertice.shape).float()[0:1]
        else:
            vertice = str(vertice[0]) # vertex path
            vertice = np.load(vertice, allow_pickle=True) # np and load from pickle 
            vertice = vertice.astype(np.float32)
            vertice = torch.from_numpy(vertice)
        if config["dataset"] == 'vocaset': # resample if it's part of the vocalset
            vertice = vertice[::2, :]
        vertice = torch.unsqueeze(vertice, 0)


        audio, vertice =  audio.to(device=device), vertice.to(device=device)
        template, one_hot_all = template.to(device=device), one_hot_all.to(device=device)

        num_frames = int(audio.shape[-1] / sr * config["output_fps"])
        shape = (1, num_frames - 1, config["vertice_dim"]) if num_frames < vertice.shape[1] else vertice.shape

        train_subject = file_name[0].split("_")[0]
        vertice_path = os.path.split(vertice_path)[-1][:-4]

        if train_subject in train_subjects_list or config["dataset"] == 'beat':
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:, iter, :]
            one_hot = one_hot.to(device=device)

            for sample_idx in range(1, config["num_samples"] + 1):
                sample = diffusion.p_sample_loop(
                    model,
                    shape,
                    clip_denoised=False,
                    model_kwargs={
                        "cond_embed": audio,
                        "one_hot": one_hot,
                        "template": template,
                    },
                    skip_timesteps=config["skip_steps"],  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                    device=device
                )
                sample = sample.squeeze()
                sample = sample.detach().cpu().numpy()

                if config["dataset"] == 'beat':
                    out_path = f"{vertice_path}.npy"
                elif config["dataset"] == "meads":
                    out_path = f"{vertice_path}.pkl"
                else:
                    if config["num_samples"] != 1:
                        out_path = f"{vertice_path}_condition_{condition_subject}_{sample_idx}.npy"
                    else:
                        out_path = f"{vertice_path}_condition_{condition_subject}.npy"
                if 'damm' in config["dataset"]:
                    sample = RIG_SCALER.inverse_transform(sample)
                    np.save(os.path.join(config["result_path"], out_path), sample)
                    df = pd.DataFrame(sample)
                    df.to_csv(os.path.join(config["result_path"], f"{vertice_path}.csv"), header=None, index=None)
                if config["dataset"] == 'meads':
                    out_json = {}
                    out_json["exp"] = sample
                else:
                    np.save(os.path.join(config["result_path"], out_path), sample)

        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:, iter, :]
                one_hot = one_hot.to(device=device)

                # sample conditioned
                sample_cond = diffusion.p_sample_loop(
                    model,
                    shape,
                    clip_denoised=False,
                    model_kwargs={
                        "cond_embed": audio,
                        "one_hot": one_hot,
                        "template": template,
                    },
                    skip_timesteps=config["skip_steps"],  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                    device=device
                )
                prediction_cond = sample_cond.squeeze()
                prediction_cond = prediction_cond.detach().cpu().numpy()

                prediction = prediction_cond
                if 'damm' in config["dataset"]:
                    prediction = RIG_SCALER.inverse_transform(prediction)
                    df = pd.DataFrame(prediction)
                    df.to_csv(os.path.join(config["result_path"], f"{vertice_path}.csv"), header=None, index=None)
                else:
                    np.save(os.path.join(config["result_path"], f"{vertice_path}_condition_{condition_subject}.npy"), prediction)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():


    # use args to modify additional features
    parser = argparse.ArgumentParser()
    # boolean for wandb
    parser.add_argument('--config', type=str, default='configs/configMeads.json')
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--wandb', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    config = json.load(open(args.config))

    # modify config accordingly
    config["wandb"] = args.wandb
    config["max_epoch"] = args.max_epoch

    if config["wandb"]:
        wandb.login()
        run = wandb.init(project="FaceDiffuser", 
                      name=f"{config['model']}_{config['dataset']}", 
                      config=config)

    assert torch.cuda.is_available()
    diffusion = create_gaussian_diffusion(config)

    if 'damm' in config["dataset"]:
        model = FaceDiffDamm(config)
    elif 'beat' in config["dataset"]:
        model = FaceDiffBeat(
                config,
                vertice_dim=config["vertice_dim"],
                latent_dim=config["feature_dim"],
                diffusion_steps=config["diff_steps"],
                gru_latent_dim=config["gru_dim"],
                num_layers=config["gru_layers"],
            )
    elif config["dataset"] == 'meads' :
        model = FaceDiffBeat(
                config,
                vertice_dim=config["vertice_dim"],
                latent_dim=config["feature_dim"],
                # cond_feature_dim=768, # this is weird, is it because of the audio encoder i'm using?
                diffusion_steps=config["diff_steps"],
                gru_latent_dim=config["gru_dim"],
                num_layers=config["gru_layers"],
            )
    else:
        model = FaceDiff(
            config,
            vertice_dim=config["vertice_dim"],
            latent_dim=config["feature_dim"],
            diffusion_steps=config["diff_steps"],
            gru_latent_dim=config["gru_dim"],
            num_layers=config["gru_layers"],
        )
    print("model parameters: ", count_parameters(model))
    cuda = torch.device(config["device"])

    model = model.to(cuda)
    dataset = get_dataloaders(config)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])

    model = trainer_diff(config, dataset["train"], dataset["valid"], model, diffusion, optimizer,
                         epoch=config["max_epoch"], device=config["device"])
    # test_diff(config, model, dataset["test"], config["max_epoch"], diffusion, device=config["device"])


if __name__ == "__main__":
    main()