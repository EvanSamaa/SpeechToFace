import argparse
from ast import Dict
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

@torch.no_grad()
def test_diff(config, model, test_loader, epoch, diffusion, device="cuda"):
    result_path = os.path.join(config["result_path"])
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)
    train_subjects_list = [i for i in config["train_subjects"].split(" ")]
    model = model.to(torch.device(device))
    model.eval()

    sr = 16000
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        all_data = {}
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
        if config["dataset"] == 'vocaset':
            vertice = vertice[::2, :]
        vertice = torch.unsqueeze(vertice, 0)


        audio, vertice =  audio.to(device=device), vertice.to(device=device)
        template, one_hot_all = template.to(device=device), one_hot_all.to(device=device)
        # print(audio.shape, vertice.shape, template.shape, one_hot_all.shape)
        num_frames = int(audio.shape[-1] / sr * config["output_fps"])
        shape = (1, num_frames - 1, config["vertice_dim"]) if num_frames < vertice.shape[1] else vertice.shape
        train_subject = file_name[0].split("_")[0]
        vertice_path = file_name[0][:-4]
        if train_subject in train_subjects_list or config["dataset"] == 'beat' or config["dataset"] == 'meads':
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
                elif config["dataset"] == 'meads':
                    expression_arr = sample[:, :50]
                    jaw_arr = sample[:, 50:]
                    all_data["exp"] = np.expand_dims(expression_arr, axis=0)
                    all_data["jaw"] = np.expand_dims(jaw_arr, axis=0)
                    # save the updated_all_data
                    with open(os.path.join(config["result_path"], out_path), "wb") as f:
                        pkl.dump(all_data, f)
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
                elif config["dataset"] == 'meads':
                    expression_arr = sample[:, :50]
                    jaw_arr = sample[:, 50:]
                    all_data["exp"] = np.expand_dims(expression_arr, axis=0)
                    all_data["jaw"] = np.expand_dims(jaw_arr, axis=0)
                    # save the updated_all_data
                    with open(os.path.join(config["result_path"], f"{vertice_path}_condition_{condition_subject}.pkl"), "wb") as f:
                        pkl.dump(all_data, f)
                else:
                    np.save(os.path.join(config["result_path"], f"{vertice_path}_condition_{condition_subject}.npy"), prediction)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():


    # use args to modify additional features
    parser = argparse.ArgumentParser()
    # boolean for wandb
    parser.add_argument('--model_path', type=str, default='/scratch/ondemand29/evanpan/FaceDiffuser/save/face_diffuser_meads_overfit/face_diffuser_meads_50.pth')
    args = parser.parse_args()

    # get paths
    model_path = args.model_path    
    folder_path = args.model_path.split("/")[:-1]
    folder_path = "/".join(folder_path)
    config_path = os.path.join(folder_path, "config.json")
    # log config
    config = json.load(open(config_path, "r"))

    # load diffusion and model accordingly
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
    cuda = torch.device(config["device"])
    
    model.load_state_dict(torch.load(model_path))
    model = model.to(cuda)
    dataset = get_dataloaders(config)

    test_diff(config, model, dataset["test"], config["max_epoch"], diffusion, device=config["device"])


if __name__ == "__main__":
    main()