import matplotlib.pyplot as plt
import pickle
import os
import json
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

with open('data/damm_rig_equal/scaler_192.pkl', 'rb') as f:
    RIG_SCALER = pickle.load(f)


def model_dir_init(config):
    """
        Create a folder to save the model and config file
        :param config: config file
        :return: model folder, path to the save file
    """
    model_type = config["model"]
    save_path = config["save_path"]
    dataset = config["dataset"]

    model_folder = os.path.join(save_path, f"{model_type}_{dataset}")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        # save config as json
        with open(os.path.join(model_folder, "config.json"), "w") as f:
            json.dump(config, f)
    else:
        # try and create a folder with a different name by appending a number behind
        i = 1
        while os.path.exists(model_folder):
            model_folder = os.path.join(save_path, f"{model_type}_{dataset}_{i}")
            i += 1
        os.makedirs(model_folder)
        # save config as json
        with open(os.path.join(model_folder, "config.json"), "w") as f:
            json.dump(config, f)
    return model_folder

def plot_losses(train_losses, val_losses, save_name="losses"):
    print(train_losses)
    print(val_losses)
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    plt.savefig(f"{save_name}.png")
    plt.close()


def create_gaussian_diffusion(config):
    # default params
    sigma_small = True
    predict_xstart = False  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = config["diff_steps"] 
    scale_beta = 1.  # no scaling 
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False 
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule("cosine", steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )