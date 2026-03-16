import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import pydicom

from torch.utils.data import Dataset
from skimage.transform import resize
from skimage import filters
from scipy.ndimage import generic_filter

logger = logging.getLogger(__name__)

class MammoDataset(Dataset):
    """Mammo Dataset class, compatible with PROCAS."""

    # Size to which the smaller images are padded to.
    PADDED_HEIGHT = 2995
    PADDED_WIDTH = 2394

    # Actual tensor sizes going into model.
    RESIZE_HEIGHT = 640
    RESIZE_WIDTH = 512

    # Average mean and std for PROCAS (training set). 
    PROCAS_PARAMS = {
        'PROCC': {'mean': (0.147, 0.147, 0.147), 'std': (0.261, 0.261, 0.261)},
        'PROMLO': {'mean': (0.185, 0.185, 0.185), 'std': (0.275, 0.275, 0.275)},
        'RAWCC': {'mean': (0.183, 0.183, 0.183), 'std': (0.331, 0.331, 0.331)},
        'RAWMLO': {'mean': (0.216, 0.216, 0.216), 'std': (0.331, 0.331, 0.331)}
    }

    def __init__(self, data_path, view_form, image_format, labels = False, from_csv = True):
        """
        Init the Mammo Dataset

        Args:
            data_path (str): Path to the data folder (must also have the data file in it)
            view_form (str): either CC or MLO depending on parser
            image_format (str): either PRO and RAW dependant on imaging format
            labels (bool, optional): Whether to return labels, by default False
            from_csv (bool, optional): Whether to read the data paths from a csv file, by default True
        """
        self.data_path = data_path
        self.view_form = view_form
        self.image_format = image_format
        self.labels = labels
        self.transform_flag = False
        
        if from_csv:
            if self.data_path[-3:] != 'csv':
                self.paths      = pd.read_csv(Path(self.data_path, 'data_file.csv'))
                self.data       = {}
            else:
                self.paths      = pd.read_csv(Path(self.data_path))
                self.data       = {}
        else:
            self.paths = self.data_path
            self.data  = {}

        # Set random transform values for training.
        self.transform = T.RandomAffine(degrees = 5, translate = (0.05,0.05), scale = (0.95,1.05), shear = (-5,5,-5,5))
        self.norm = T.Normalize(
            self.PROCAS_PARAMS[self.image_format + self.view_form]['mean'],
            self.PROCAS_PARAMS[self.image_format + self.view_form]['std']
            )

        # Construct the dict with data paths.
        self._construct_dict()

    def _construct_dict(self):
        """
        Method to create a dic of datapoints from a given folder. Automatically called during init. 
        Currently returns, the paths and side.
        """
        stripped_data = self.paths[(self.paths['view']==self.view_form) & (self.paths['format']==self.image_format)]
        self.data['Paths'] = list(stripped_data['path'].apply(Path))
        self.data['Side']  = list(stripped_data['side'])
        self.data['IDS']   = list(stripped_data['patient'])
        
        if self.labels:
            self.data['Labels'] = list(stripped_data['label'])

    @staticmethod
    def check_image(image, name = None):
        """Method to visually assess the image. Plots the given numpy array."""
        plt.imshow(image, cmap=plt.cm.gray)
        plt.title(name)
        plt.axis('off')
        plt.show()     
    
    
    def _fix_inf(self, image):
        """
        Convolves across the image in instances where the is an inf after log. Sets the erronous
        values to the value of its neigbours.
        """
        def replace_inf_with_neighbors(values):
            if np.isinf(values[12]):
                neighbors = values[np.isfinite(values)]
                if len(neighbors) > 0:
                    return np.mean(neighbors)
            return values[12]
        
        new_image = generic_filter(image, replace_inf_with_neighbors, size=5, mode='constant', cval=0)
        return new_image
    
    def _pad_image(self, image):
        """Pads image to a consistent size."""
        padded_image = np.zeros((np.amax([self.PADDED_HEIGHT, image.shape[0]]), np.amax([self.PADDED_WIDTH, image.shape[1]])), dtype=image.dtype)
        padded_image[:image.shape[0],:image.shape[1]] = image
        return padded_image[0:self.PADDED_HEIGHT, 0:self.PADDED_WIDTH]
    
    def preprocess_image(self, image_path, side):
        """Load and preprocess the given image. Does not include pixel spacing modifiers."""

        # Read the dicom and fetch the pixel array
        if image_path.suffix != '.dcm' and image_path.suffix != '.npy':
            raise ValueError(f'Image path needs to be a .dcm or .npy file, not {image_path}.')
        if image_path.suffix == '.dcm':
            image = pydicom.read_file(image_path).pixel_array
        else:
            image = np.load(image_path.with_suffix('.npy'))

        # Right side mammograms are flipped
        if side != 'L' and side != 'R':
            raise ValueError(f'The attributed must be either L or R, not {side}.')
        if side == 'R':
            image = np.fliplr(image)

        # Find otsu cutoff threshold
        cut_off = filters.threshold_otsu(image)
        
        # For RAW, apply otsu's, log and invert
        if self.image_format == 'RAW':
            np.clip(image, 0, cut_off, out = image)
            image = np.log(image) # Returns float32.
            np.subtract(np.amax(image), image, out = image) # Flips in place
            
        # For PRO, apply otsu's
        elif self.image_format == 'PRO':
            np.clip(image, cut_off, np.amax(image), out = image)
            image = image.astype(np.float32) 
            image = (image - np.amin(image)) / (np.amax(image)- np.amin(image)) # By default, the normalisation returns float64.
 
        # Resolves issue of invalid pixels for RAW. 
        if np.amax(image) == np.inf:
            logging.warning(f'Fixed inf issue for: {image_path}')
            image = self._fix_inf(image)

        # Pad images to the same size before resizing
        image = self._pad_image(image)

        # Resize to this precise dimension
        image = resize(image, (self.RESIZE_HEIGHT, self.RESIZE_WIDTH))

        # Max min normalise
        image /= np.amax(image)
        
        # Replicate across the channels
        image = np.stack((image, image, image), 0)

        # Send to tensor
        image = torch.as_tensor(image)
        
        # Training phase only:
        if self.transform_flag:
            image = self.transform(image)
            
        # Normalise to 0 mean 1 s.d. These are PROCAS means (training)
        image = self.norm(image)

        return image
    
        
    def __len__(self):     
        return len(self.data['Paths'])
        
        
    def __getitem__(self, index):
        """ Returns the array, name and side as a dict. Label optional"""
        retrieved_sample = {
            'image': self.preprocess_image(self.data['Paths'][index], self.data['Side'][index]),
            'name': self.data['Paths'][index].parent.name+'/'+self.data['Paths'][index].stem, # self.data['Paths'][index].as_posix() for full path
            'side': self.data['Side'][index]
            }

        if self.labels:
            retrieved_sample['label'] = torch.as_tensor(self.data['Labels'][index]).double().contiguous()
            
        return retrieved_sample
        


if __name__ == "__main__":

    dataset = MammoDataset(data_path = 'Z:/Code/Projects/MAI-VAS/data', 
                          view_form = 'MLO', image_format = 'RAW', labels = True)
    output = dataset[0]