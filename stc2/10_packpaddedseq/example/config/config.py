import torch
# Define hyper parameter
use_cuda = True if torch.cuda.is_available() else False

# for dataset
dataset_path_sw = 'dataset/sw-rl.txt'
dataset_path_ww = 'dataset/ww-rl.txt'

# for training
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

# for model
encoder_embedding_size = 256
encoder_output_size = 256
decoder_hidden_size = encoder_output_size
teacher_forcing_ratio = .5
max_length = 20

# for logging
checkpoint_name = 'auto_encoder.pt'