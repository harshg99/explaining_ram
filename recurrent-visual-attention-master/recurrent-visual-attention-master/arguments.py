# Architecture level Params: Soft attention Model
mode = "HardAttRewardShaping" # HardAtt, HardAttLSTM,HardAttwRewardShaping, HardAttAC2 HardAttAC2RewardShaping HardAttAC2LSTMRewardShaping


if mode=="HardAtt":
    core_net_type="Linear"
    training_mode="default"
    reward = "default"
    critic_weight = 1.0
    actor_weight = 0.02
elif mode=="HardAttLSTM":
    core_net_type="LSTM"
    training_mode="default"
    reward= "default"
    critic_weight = 1.0
    actor_weight = 0.02
elif mode=="HardAttRewardShaping":
    core_net_type = "Linear"
    training_mode = "default"
    reward = "logprob"
    critic_weight = 1.0
    actor_weight = 0.02
elif mode=="HardAttAC2":
    core_net_type = "Linear"
    training_mode = "AC2"
    reward = "default"
    critic_weight = 0.4
    actor_weight = 0.05
elif mode=="HardAttAC2RewardShaping":
    core_net_type = "Linear"
    training_mode = "AC2"
    reward = "logprob"
    critic_weight = 0.4
    actor_weight = 0.05
elif mode=="HardAttAC2LSTMRewardShaping":
    core_net_type = "LSTM"
    training_mode = "AC2"
    reward = "logprob"
    critic_weight = 0.4
    actor_weight = 0.05


#Location Network Params
patch_size = 8 # Patch size (retina)
glimpse_scale = 1 # Scale of the glimpse
num_patches = 1 # Number of patches
loc_hidden = 128 #Length of the location encoding
glimpse_hidden = 128 # Length of glimpse encoding


#Core Network Params
num_glimpses = 4 # Length of the recurrent network
hidden_size = 256 # Hidden size of the RNN

# reinforce params
std = 0.05 # Standard deviation for location network
M = 1 #Monte carlo sampling
std_decay = 0.90 #decay in standard deviation across epochs

# data params
valid_size =0.1 #Proportion of training set used for validation"
batch_size = 128 #  images in each batch of data"
num_workers = 4 # of subprocesses to use for data loading",
shuffle = True # Whether to shuffle the train and valid indices",
show_sample = False # Whether to visualize a sample grid of the data",

# training params
is_train = True#Whether to train or test the model
momentum = 0.5 #Nesterov momentum value
epochs = 200 # of epochs to train for"
init_lr = 3e-4 #Initial learning rate value
lr_patience = 20 #Number of epochs to wait before reducing lr"
train_patience = 50 #Number of epochs to wait before stopping train"


# other params
use_gpu = True #Whether to run on the GPU
best = True#Load best model or most recent for testing
random_seed = 1 #Seed to ensure reproducibility
data_dir = "./data" #Directory in which data is stored
ckpt_dir ="./ckpt" #Directory in which to save model checkpoints
logs_dir = "./logs/" #"Directory in which Tensorboard logs wil be stored
use_tensorboard = True #Whether to use tensorboard for visualization"
resume = False #Whether to resume training from checkpoint
print_freq = 10 #How frequently to print training details",
plot_freq = 1 #How frequently to plot glimpses

# Name of the model
version = "vr2"
model_name = "ram_{}_{}x{}_{}_{}_m{}v_{}".format(
    num_glimpses, patch_size, patch_size, glimpse_scale,num_patches,mode,version
)
