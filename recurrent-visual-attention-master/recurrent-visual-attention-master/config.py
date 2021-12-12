import argparse
from arguments import *
arg_lists = []
parser = argparse.ArgumentParser(description="RAM")


def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Architecture Params
arch_param = add_argument_group("Architecture Params")
arch_param.add_argument("--mode",type=str, default=mode, help="size of extracted patch at highest res"
)

# glimpse network params
glimpse_arg = add_argument_group("Glimpse Network Params")
glimpse_arg.add_argument(
    "--patch_size", type=int, default=patch_size, help="size of extracted patch at highest res"
)
glimpse_arg.add_argument(
    "--glimpse_scale", type=int, default=glimpse_scale, help="scale of successive patches"
)
glimpse_arg.add_argument(
    "--num_patches", type=int, default=num_patches, help="# of downscaled patches per glimpse"
)
glimpse_arg.add_argument(
    "--loc_hidden", type=int, default=loc_hidden, help="hidden size of loc fc"
)
glimpse_arg.add_argument(
    "--glimpse_hidden", type=int, default=glimpse_hidden, help="hidden size of glimpse fc"
)


# core network params
core_arg = add_argument_group("Core Network Params")
core_arg.add_argument(
    "--num_glimpses", type=int, default=num_glimpses, help="# of glimpses, i.e. BPTT iterations"
)
core_arg.add_argument("--hidden_size", type=int, default=hidden_size, help="hidden size of rnn")

core_arg.add_argument(
    "--core_net_type", type=str, default=core_net_type, help="# of glimpses, i.e. BPTT iterations"
)
# reinforce params
reinforce_arg = add_argument_group("Reinforce Params")
reinforce_arg.add_argument(
    "--std", type=float, default=std, help="gaussian policy standard deviation"
)
reinforce_arg.add_argument(
    "--M", type=int, default=M, help="Monte Carlo sampling for valid and test sets"
)
reinforce_arg.add_argument(
    "--std_decay", type=int, default=std_decay, help="Monte Carlo sampling for valid and test sets"
)

# data params
data_arg = add_argument_group("Data Params")
data_arg.add_argument(
    "--valid_size",
    type=float,
    default=valid_size,
    help="Proportion of training set used for validation",
)
data_arg.add_argument(
    "--batch_size", type=int, default=batch_size, help="# of images in each batch of data"
)
data_arg.add_argument(
    "--num_workers",
    type=int,
    default=num_workers,
    help="# of subprocesses to use for data loading",
)
data_arg.add_argument(
    "--shuffle",
    type=str2bool,
    default=shuffle,
    help="Whether to shuffle the train and valid indices",
)
data_arg.add_argument(
    "--show_sample",
    type=str2bool,
    default=show_sample,
    help="Whether to visualize a sample grid of the data",
)


# training params
train_arg = add_argument_group("Training Params")
train_arg.add_argument(
    "--is_train", type=str2bool, default=is_train, help="Whether to train or test the model"
)
train_arg.add_argument(
    "--momentum", type=float, default=momentum, help="Nesterov momentum value"
)
train_arg.add_argument(
    "--epochs", type=int, default=10, help="# of epochs to train for"
)
train_arg.add_argument(
    "--init_lr", type=float, default=init_lr, help="Initial learning rate value"
)
train_arg.add_argument(
    "--lr_patience",
    type=int,
    default=lr_patience,
    help="Number of epochs to wait before reducing lr",
)
train_arg.add_argument(
    "--train_patience",
    type=int,
    default=train_patience,
    help="Number of epochs to wait before stopping train",
)

train_arg.add_argument(
    "--training_mode",
    type=str,
    default=training_mode,
    help="Number of epochs to wait before stopping train",
)

train_arg.add_argument(
    "--reward",
    type=str,
    default=reward,
    help="Number of epochs to wait before stopping train",
)
train_arg.add_argument(
    "--actor_weight",
    type=str,
    default=actor_weight,
    help="Number of epochs to wait before stopping train",
)
train_arg.add_argument(
    "--critic_weight",
    type=str,
    default=critic_weight,
    help="Number of epochs to wait before stopping train",
)
train_arg.add_argument(
"--vae_patience",
    type=str,
    default=vae_patience,
    help="Number of epochs to wait before starting training VAE"

)

train_arg.add_argument(
    "--partial_vae",
    type=str,
    default=partial_vae,
    help="Number of epochs to wait before starting training VAE"
)
# other params
misc_arg = add_argument_group("Misc.")
misc_arg.add_argument(
    "--use_gpu", type=str2bool, default=use_gpu, help="Whether to run on the GPU"
)
misc_arg.add_argument(
    "--best",
    type=str2bool,
    default=best,
    help="Load best model or most recent for testing",
)
misc_arg.add_argument(
    "--random_seed", type=int, default=random_seed, help="Seed to ensure reproducibility"
)
misc_arg.add_argument(
    "--data_dir", type=str, default=data_dir, help="Directory in which data is stored"
)
misc_arg.add_argument(
    "--ckpt_dir",
    type=str,
    default=ckpt_dir,
    help="Directory in which to save model checkpoints",
)
misc_arg.add_argument(
    "--logs_dir",
    type=str,
    default=logs_dir,
    help="Directory in which Tensorboard logs wil be stored",
)
misc_arg.add_argument(
    "--use_tensorboard",
    type=str2bool,
    default=use_tensorboard,
    help="Whether to use tensorboard for visualization",
)
misc_arg.add_argument(
    "--resume",
    type=str2bool,
    default=resume,
    help="Whether to resume training from checkpoint",
)
misc_arg.add_argument(
    "--print_freq",
    type=int,
    default=print_freq,
    help="How frequently to print training details",
)
misc_arg.add_argument(
    "--plot_freq", type=int, default=plot_freq, help="How frequently to plot glimpses"
)

misc_arg.add_argument(
    "--model_name", type=str, default=model_name, help="storing models"
)

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
