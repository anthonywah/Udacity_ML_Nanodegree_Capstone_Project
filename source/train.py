import argparse
import sys
import os
import json
import timeit
import pandas as pd

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

# import model
from model import DynamicNet


# Most codes are reusable from previous modules


def model_fn(model_dir):
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("Got model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicNet(model_info['input_dim'], 
                       model_info['hidden_dim_list'], 
                       model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    return model.to(device)


# Load the training data from a csv file
def _get_train_loader(batch_size, data_dir):
    print("Get data loader.")

    # read in csv file
    train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None, names=None)

    # labels are first column
    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    # features are the rest
    train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()

    # create dataset
    train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


# Provided train function
def train(model, train_loader, epochs, optimizer, criterion, device):
    """ This is the training method that is called by the PyTorch training script. The parameters passed are as follows:
    
    param model:
        - The PyTorch model that we wish to train.
    param train_loader
        - The PyTorch DataLoader that should be used during training.
    param epochs
        - The total number of epochs to train for.
    param optimizer
        - The optimizer to use during training.
    param criterion
        - The loss function used for training. 
    param device
        - Where the model and data should be loaded (gpu or cpu).
            
    """
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = timeit.default_timer()
        
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() # zero accumulated gradients
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Epoch: {:<3}, Loss: {:<15}, used {}s".format(epoch, 
                                                           round(total_loss / len(train_loader), 10), 
                                                           round(timeit.default_timer() - epoch_start_time, 3)))
        
    return


def save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)
    return
    
def save_model_params(model_info, model_dir):
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        torch.save(model_info, f)
    return


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    parser.add_argument('--hosts',         type=list,  default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host',  type=str,   default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir',     type=str,   default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir',      type=str,   default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters
    parser.add_argument('--batch-size',       type=int,   default=64,           metavar='N',   
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs',           type=int,   default=10,           metavar='N',   
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr',               type=float, default=0.001,        metavar='LR',  
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed',             type=int,   default=1,            metavar='S',   
                        help='random seed (default: 1)')
    parser.add_argument('--input_dim',        type=int,   default=13,           metavar='IN',  
                        help='# input features (default: 13)')
    parser.add_argument('--hidden_dim_list',  type=str,   default="12_24_,8",   metavar='H',   
                        help='# hidden layer features in "x_y_z" format (default: 10)')
    parser.add_argument('--output_dim', type=int, default=1, metavar='OUT', help='# output features (default: 1)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    # get train loader
    train_loader = _get_train_loader(args.batch_size, args.data_dir) # data_dir from above..
    
    # Build Model
    hidden_dim_list_parsed = [int(i) for i in args.hidden_dim_list.split('_')]
    model = DynamicNet(args.input_dim, hidden_dim_list_parsed, args.output_dim).to(device)
    model_info = {'input_dim': args.input_dim, 
                  'hidden_dim_list': hidden_dim_list_parsed, 
                  'output_dim': args.output_dim}
    print(model_info)
    
    save_model_params(model_info, args.model_dir)
    
    print(model.parameters())

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Switching to Mean-Square Loss here since we are not doing a binary classification
    criterion = nn.MSELoss()
    
    # Trains the model, but unlike previous modules, this function DOES NOT save the model state dictionary
    train(model, train_loader, args.epochs, optimizer, criterion, device)
    
    # Save after all training
    save_model(model, args.model_dir)

