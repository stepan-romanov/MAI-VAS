# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 19:25:10 2025

@author: Stepan

Code to train MAI-VAS with.
"""
import argparse
import os
import time
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.utils.data as data_utils
from torch.nn import MSELoss
import torch.optim as optim

from mammo_dataset import MammoDataset
from model import MAI_VAS_Model
from mai_vas_utils import EarlyStopper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='Training pVAS sr')
    parser.add_argument('--no-cuda', action='store_true', default = False, help='disables CUDA training')
    parser.add_argument('--epochs', type=int, default=2, metavar='N', help = 'Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help = 'Learning rate value')
    parser.add_argument('--bs', type=int, default=1, metavar='B', help = 'Batch size value')
    parser.add_argument('--view', type=str, default='MLO', help = 'Which mammographic view to use, CC or MLO options')
    parser.add_argument('--pretrained', action='store_true', default=True, help = 'Use pretrained network or not?')
    parser.add_argument('--name', type=str, default = 'model1', help = 'What to call the model test? Creates a folder with with this name.')
    parser.add_argument('--format', type=str, default = 'PRO', help = 'Image format')
    parser.add_argument('--data_path', type=str, default ='./data', help = 'Path to the images')
    parser.add_argument('--priors', action='store_true', help = 'Test on priors dataset')
    parser.add_argument('--train_split', type=float, default = 0.8, help = 'Portion of data to use for training vs validaiton.')
    parser.add_argument('--seed', type=int, default = 409, help = 'Random seed for reproducibility')

    return parser.parse_args()

    
def train(epoch, model, loader, optimizer, criterion, device):
    """
    Main training function. Trains the model for 1 epoch and returns the loss.

    RETURNS:
        train_loss (float): The average training loss for the epoch.
    """

    # Init
    model.train()
    train_loss = 0

    # Turn transforms back on
    loader.dataset.dataset.transform_flag = True
    logger.info(f'Begin training phase:')
    
    with tqdm(total=len(loader), unit=" batch") as t:
        for batch_idx, item in enumerate(loader):
            optimizer.zero_grad()

            R, loss = forward_pass(model, item['image'], item['label'], item['name'], criterion, device) 

            if R is None:
                t.update(1)
                continue
            
            train_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()
                    
            t.update(1)
            
        train_loss /= len(loader)
        logger.info(f'Epoch: {epoch}, Train Loss: {train_loss}')
        return train_loss


def validate(model, loader, criterion, device):
    """
    Validation function. Validates the model and returns loss only.

    RETURNS:
        val_loss (float): The average validation loss for the epoch.
    """

    # Init
    model.eval()
    val_loss = 0

    # Turns transforms off
    loader.dataset.dataset.transform_flag = False

    logger.info(f'Begin validation phase')
    with torch.no_grad():
        with tqdm(total=len(loader), unit=" batch") as t:
            for batch_idx, item in enumerate(loader):
                R, loss = forward_pass(model, item['image'], item['label'], item['name'], criterion, device)

                if R is None:
                    t.update(1)
                    continue

                loss = loss.float()
                val_loss += loss.item()

                t.update(1)
        
    val_loss /= len(loader)
    
    logger.info(f'Validation Loss: {val_loss}')
    return val_loss



def test(model, loader, criterion, device):
    """
    Test function. Tests the model.

    RETURNS:
        test_loss (float): The average test loss for the epoch.
        name_list : List of image names.
        label_list : List of labels. 
        side_list : List of sides.
        output_list : List of model predictions.
    """
        
    # Init
    model.eval()
    test_loss = 0  

    name_list = []
    label_list = []
    output_list = []
    side_list= []

    logger.info('Begin Testing Phase')
    with torch.no_grad():
        with tqdm(total=len(loader), unit=" batch") as t:
            for batch_idx, item in enumerate(loader):
                image = item['image']
                label = item['label']
                name = item['name']
                
                R, loss = forward_pass(model, image, label, name, criterion, device)

                if R is None:
                    t.update(1)
                    continue

                loss = loss.float()
                test_loss += loss.item()

                name_list.append(name[0])
                label_list.append(label.item())
                output_list.append(R.item())
                side_list.append(item['side'])

                t.update(1)
        
    test_loss /= len(loader)
    
    logger.info(f'Test set, Loss: {test_loss}')
    return test_loss, name_list, label_list, side_list, output_list


def forward_pass(model, image, label, name, criterion, device):
    """Shared forward pass."""
    if torch.isnan(image).any():
        logger.warning(f'Dicom {name[0]} has nan values')
        return None, None
    
    if torch.isinf(image).any():
        logger.warning(f'Dicom {name[0]} has inf values')
        return None, None
    
    image, label = image.to(device), label.to(device)
    R = model(image)
    loss = criterion(R, label.unsqueeze(1))
    
    return R, loss

def _construct_loader(args, device):
    """Constructs 3 dataloaders needed for training, validation and testing. Use separate dataset for priors, the same for val and train."""
    dataset = MammoDataset(data_path = args.data_path, view_form = args.view, image_format = args.format, labels = True)
    if args.priors:
        priors  = MammoDataset(data_path = './data_file_priors.csv', view_form = args.view, image_format = args.format, labels = True)
    else:
        priors = None

    # Retreive all the unique women ID 
    women = list(set(dataset.data['IDS']))
    women_count = len(women)
    
    # Uses 80/20 data splits but these can be adjusted.
    n_train = int(women_count*args.train_split)
    n_val   = women_count - n_train
    logger.info(f'Training on {n_train} women, Validation on {n_val} women')
    logger.info('Testing on the priors dataset.')
    
    # Setup kwargs for loader
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == 'cuda' else {}
    generator = torch.Generator().manual_seed(args.seed)
    
    # We split the women, not the images. This is to ensure both the L and R images are in the same set.
    train_set, val_set = data_utils.random_split(range(women_count), [n_train, n_val], generator=generator)
   
    # Converts the variables to just be a list of patient IDS rather than T/F
    train_set = [women[idx] for idx in train_set.indices]
    val_set = [women[idx] for idx in val_set.indices]
    
    # Maps the IDS to the dataset ID. Creates a mask of indices over the original dataset
    train_ids = np.array([x in train_set for x in dataset.data['IDS']])
    val_ids = np.array([x in val_set for x in dataset.data['IDS']])

    # Splice the sets based on the masks
    train_set = data_utils.Subset(dataset, np.where(train_ids)[0])
    val_set   = data_utils.Subset(dataset, np.where(val_ids)[0])
    
    # Construct the loaders
    train_loader = data_utils.DataLoader(train_set,
                                             batch_size=args.bs,
                                             shuffle=True,
                                             **loader_kwargs)
    
    # Testing sets have batch size 1 to ensure we can track the outputs and names correctly.
    val_loader = data_utils.DataLoader(val_set,
                                             batch_size=1,
                                             shuffle=False,
                                             **loader_kwargs)
    if args.priors:
        prior_loader = data_utils.DataLoader(priors,
                                                batch_size=1,
                                                shuffle=False,
                                                **loader_kwargs)
    else:
        prior_loader = None
    
    return train_loader, val_loader, prior_loader

def main():
    # Init params
    start_time = time.time()
    args = get_args()
    
    # Check if cuda is available and initialise
    cuda = not args.no_cuda and torch.cuda.is_available()
    if cuda:
        device = torch.device('cuda')
        logger.info('GPU is ON!')
    else:
        device = torch.device('cpu')
        logger.warning('GPU is OFF, using CPU instead.')

    # Init saving directory
    os.makedirs('./' + str(args.name), exist_ok=True)
    save_path = f'./{args.name}/{args.view}_{args.format}_model'
    logger.info(f'Initiliasing {args.view} model with {args.format} image format.')

    # Construct loaders
    train_loader, val_loader, prior_loader = _construct_loader(args, device)

    # Init model params
    model = MAI_VAS_Model(pretrain = args.pretrained, dropout = 0.5, model = 'resnet50').to(device).double()
        
    # Init optimiser and stopper
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    stopper = EarlyStopper(patience = 7, min_delta = 4)
    criterion = MSELoss()
    logger.info(f'Using the following parameters:  \n BS: {args.bs} \n LR: {args.lr}, \n pretrained: {args.pretrained}.')
    
    # Init loop & saving parameters
    epoch_losses = []
    val_losses = []
    min_loss = np.inf
    
    # Main training loop. Trains the model for 1 epoch, then validations. If Val loss is minimum saves the model. Tracks the losses.
    for epoch in range(1, args.epochs + 1):
        # Train phase
        train_loss = train(epoch = epoch, model = model, loader = train_loader, optimizer = optimizer, criterion = criterion, device = device)
        epoch_losses.append(round(train_loss,5))
        
        # Val phase
        val_loss = validate(model = model, loader = val_loader, criterion = criterion, device = device)
        val_losses.append(round(val_loss,5))
        
        # Record
        out_data = pd.DataFrame({'Train_loss': epoch_losses, 
                                 'Val_loss':val_losses})
        out_data.to_csv(save_path+'_losses.csv', index = False)
        
        # Stopper
        if val_loss < min_loss:
            logger.info(f'Saving epoch {epoch} for having the lowest loss.')
            min_loss = val_loss
            torch.save(model.state_dict(),save_path+'.pth')
        if stopper.early_stop(val_loss):
            logger.info(f'Training stopped during epoch {epoch} during to satisfying the stopping criterion')
            break
    
    if args.priors:
        # Testing phase. Loads the best model and begins testing phase. Saves output. Plots if not on the csv.
        model.load_state_dict(torch.load(save_path+'.pth', map_location = device))
        
        # Test on the prior set
        test_loss, name_list, label_list, side_list, output_list = test(model = model, loader = prior_loader, criterion = criterion, device = device)
        test_outdata = pd.DataFrame({'Names':name_list, 'Side':side_list, 'Labels':label_list, 'Model_out': output_list})
        test_outdata.to_csv(save_path+'_priors_out.csv', index = False)
        
    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f'Elapsed time: {elapsed_time:.2f} seconds')
    
if __name__ == "__main__":
    main()
    

