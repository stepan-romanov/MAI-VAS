import argparse
import time
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.utils.data as data_utils

from mammo_dataset import MammoDataset
from model import MAI_VAS_Model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_args():
    
    parser = argparse.ArgumentParser(description='Using the MAI-VAS model')
    parser.add_argument('--data_path', type=str, default ='../data', help = 'Path to the images')
    parser.add_argument('--no-cuda', action='store_true', default = False, help='disables CUDA training')
    parser.add_argument('--view', type=str, default='MLO', help = 'Which mammographic view to use')
    parser.add_argument('--format', type=str, default = 'PRO', help = 'RAW or PRO image format')
    parser.add_argument('--model_path', type=str, default = '../model', help = 'Different model path if required')
    parser.add_argument('--bs', type = int, default = 1, help = 'Batch size during inference')
    parser.add_argument('--save_path', type=str, default = '../results', help = 'Save folder location, defaults to ../results')
    
    return parser.parse_args()

def collate_fn(batch):
    """Clears errors out of loader batches."""
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def _construct_loader(args):
    # Load the dataset
    logger.info(f'Initialising {args.view} model with {args.format} image format.')
    dataset = MammoDataset(data_path = args.data_path, view_form = args.view, image_format = args.format)
    logger.info(f'Testing on a dataset with {len(dataset)} images.')
    
    # Initialise the loader
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    loader = data_utils.DataLoader(dataset,
                                   batch_size=args.bs,
                                   shuffle=False,
                                   collate_fn=collate_fn,
                                   **loader_kwargs)
    
    return loader


def inference(args, model, loader, device):
    """
    Main inference function
    
    RETURNS:
        List of image names
        List of model predictions
        List of sides
    """  
    model.eval()
    logger.info('Begin Inferencing Phase')
    
    # Init output lists
    name_list = np.empty((len(loader) * args.bs,), dtype=object)
    output_list = np.empty((len(loader) * args.bs,), dtype=float)
    side_list = np.empty((len(loader) * args.bs,), dtype=object)
    
    # Main loop
    with torch.no_grad():
        with tqdm(total=len(loader), unit=" batch") as t:
            for batch_idx, item in enumerate(loader):
                image = item['image']
                name  = item['name']
                side  = item['side']
                
                image = image.to(device)
            
                # Inference
                R = model(image)
                
                # Save outputs in lists
                start_idx = batch_idx * args.bs
                end_idx = start_idx + len(name)
                output_list[start_idx:end_idx] = R.squeeze().cpu().numpy().round(6)
                name_list[start_idx:end_idx] = np.array(name)
                side_list[start_idx:end_idx] = np.array(side)
                
                t.update(1)

    return name_list[:end_idx], output_list[:end_idx], side_list[:end_idx]

def main():
    # Init params
    start_time = time.time()
    args = get_args()
    
    # Check if cuda is available and initialise TEST ON GPU.
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        device = torch.device('cuda')
        logger.info('GPU is ON!')
    else:
        device = torch.device('cpu')
        logger.warning('GPU is OFF, using CPU instead.')
        
    # Construct the loader
    loader = _construct_loader(args)
    
    # Fetch the model
    model = MAI_VAS_Model(pretrain = True).to(device)
    model.load_state_dict(torch.load(f'{args.model_path}/{args.view}_{args.format}_model.pth', map_location=device, weights_only = True))
    logger.info(f'Loaded the {args.model_path}/{args.view}_{args.format}_model.pth model.')
    
    # Inference
    name_list, output_list, side_list = inference(args = args, model = model, loader = loader, device = device)
    
    # Save the output as a pandas dataframe
    test_outdata = pd.DataFrame({'Names':name_list, 'Side':side_list, 'Model_out': output_list})
    test_outdata.to_csv(f'{args.save_path}/{args.view}_{args.format}_results.csv', index = False)
    logger.info(f'Saved the results in: {args.save_path}/{args.view}_{args.format}_results.csv')
    
    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f'Elapsed time: {elapsed_time:.2f} seconds')
    
if __name__ == "__main__":
    main()
    
    