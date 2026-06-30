import os
import sys
import traceback
from pathlib import Path

from PySide2.QtCore import QObject, Slot, Signal

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from skimage import filters

import torch
import torch.utils.data as data_utils

from mammo_dataset import MammoDataset
from model import MAI_VAS_Model

class MaiVasModel():
    def __init__(self): 
        self.model_path = self.path_resolver('..\model', '.\model') # Does not need the MEIPASS adjustment as we are adding the model weights manually.
        
        self.model = MAI_VAS_Model(pretrain = True)
        
        self.cuda = torch.cuda.is_available()
        
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Model options
        self.view = 'CC'
        self.format = 'RAW'
        self.save_path = self.path_resolver('..', '.') # In .py saves in parent dir, in exe in current dir.
        self.data_path = self.path_resolver('..\data', '.') # In .py saves in ..\data, otherwise defaults to current dir.
        self.bs = 1
        
        # Model data loading mode: 0 - folder, 1 - single image, 2- metadata
        self.mode = 0
        self.multiple_views = None
        self.data_handler = None

        # Datarelevant
        self.loader = None
        
    def load_model(self):
        try: 
            self.model.load_state_dict(torch.load(f'{self.model_path}/{self.view}_{self.format}_model.pth', map_location=self.device, weights_only = True))
        except Exception:
            traceback.print_exc()
            raise RuntimeError('Failed to load model weights. Check they are in the correct folder.')

    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
    
    @staticmethod
    def path_resolver(path, exe_path = None):
        """Adapts the path whether the GUI was ran through the exe or console. Change variable is for exe instances were you dont use the temp folder."""
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(os.path.dirname(sys.executable), exe_path)
        else:
            return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    
    def load_data(self, progress_signal = None, error_signal = None):
        try:
            self.multiple_views = None
            
            if self.mode == 2:
                dataset = MammoDataset(data_path = self.data_path, view_form = self.view, image_format = self.format, labels = False)
                
            elif self.mode == 0:
                self.data_handler = DataHandler(Path(self.data_path), progress_signal = progress_signal)

                # This check if there are NO valid images in the entire folder and sends signal
                if self.data_handler.meta_df.empty:
                    error_signal(str('Folder does not contain any valid mammograms for inference. See the ERRORS file that was generated for further details.'))
                    self.data_handler.warning.to_csv(f'{self.save_path}/ERRORS.csv', index = False)
                    return

                self.view = self.data_handler.views[0]
                dataset = MammoDataset(data_path = self.data_handler.meta_df, view_form = self.view, image_format = self.format, from_csv = False, labels = False)
                
                if len(self.data_handler.views) > 1:
                    self.multiple_views = self.data_handler.views[1]
                
            elif self.mode == 1:
                self.data_handler = DataHandler(self.data_path)

                # This check if there are NO valid images in the entire folder and sends signal
                if self.data_handler.meta_df.empty:
                    error_signal(str('The Image has associated errors and cannot be processed by MAI-VAS at this stage. Issue: ')) #TODO PRINT STRING
                    return

                self.view = self.data_handler.views[0]
                self.format = self.data_handler.format[0]
                dataset = MammoDataset(data_path = self.data_handler.meta_df, view_form = self.view, image_format = self.format, from_csv = False, labels = False)
            
            loader_kwargs = {'num_workers':4, 'pin_memory':True} if self.cuda else {}
            self.loader = data_utils.DataLoader(dataset,
                                           batch_size = self.bs,
                                           shuffle = False,
                                           collate_fn = self.collate_fn,
                                       **loader_kwargs)
        except Exception:
            traceback.print_exc()
            raise RuntimeError('Failed to load data.')
    
    def test_initialisation(self):
        return
    

class DataHandler():
    def __init__(self, data_path, progress_signal = None):
        
        self.data_path = data_path
        self.progress_signal = progress_signal

        if os.path.isfile(self.data_path) and pydicom.misc.is_dicom(self.data_path):
            self.meta_df, self.warning = self._construct_df([self.data_path])
        elif os.path.isdir(self.data_path):
            path_list = []
            for path in self.data_path.rglob('*'):
                if os.path.isfile(path) and pydicom.misc.is_dicom(path):
                    path_list.append(path)
            self.meta_df, self.warning = self._construct_df(path_list)
            
        else:
            exit(0)
        
        self.views  = self.meta_df['view'].unique().tolist()
        self.format = self.meta_df['format'].unique().tolist()
        
    def _detect_patient(self, dcm):
        try:
            patient_id = dcm[0x00100020].value
            return patient_id
        except Exception:
            return None
         
    def _detect_views(self, dcm):
        try:
            code_meaning = dcm[0x00540220][0][0x00080104].value
            if code_meaning == 'cranio-caudal':
                return 'CC'
            elif code_meaning == 'medio-lateral oblique':
                return 'MLO'
        except Exception:
            return None
        
    def _detect_format(self, dcm):
        try:
            presentation_intent_type = dcm[0x00080068].value
            if presentation_intent_type == 'FOR PRESENTATION':
                return 'PRO'
            elif presentation_intent_type == 'FOR PROCESSING':
                return 'RAW'
        except Exception:
            return None
    
    def _detect_vendor(self, dcm):
        try:
            vendor = dcm[0x00080070].value
            return vendor
        except Exception:
            return None
    
    def _detect_side(self, dcm):
        try:
            image_laterality = dcm[0x00200062].value
            return image_laterality
        except Exception:
            return None

    def _construct_df(self, input_list):
        
        # Init meta data columns
        patients = []
        paths    = []
        sides    = []
        views    = []
        formats  = []

        # For errors
        warnings = []

        total = len(input_list)
        for i, img in enumerate(input_list):
            try:
                dcm = pydicom.read_file(img)
            except Exception:
                warnings.append((img, 'Unable to open dicom file, check image integrity.'))
                continue

            # Detect key metadata
            patient_id = self._detect_patient(dcm)
            vendor = self._detect_vendor(dcm)
            side = self._detect_side(dcm)
            view = self._detect_views(dcm)
            format = self._detect_format(dcm)

            # Check if any of the essential fields have thrown an error
            missing = [column for column, val in {
                'patient_id': patient_id,
                'vendor':     vendor,
                'side':       side,
                'view':       view,
                'format':     format,
            }.items() if val is None]

            # If any are missing, add to warning file.
            if missing:
                warnings.append((img, f'Warning: Missing {", ".join(missing)}. This image will be skipped.'))
                continue 

            # If any are non GE, add to warning file.
            if 'GE' not in vendor:
                warnings.append((img, f'Warning: MAI-VAS has not been developed for {vendor} images.'))
            
            # Append valid entries
            patients.append(patient_id)
            paths.append(img)
            sides.append(side)
            views.append(view)
            formats.append(format)

            # Emit to progress bar
            if self.progress_signal and total > 0:
                self.progress_signal(int((i + 1) / total * 100))
        # Main meta-data file
        data = pd.DataFrame({
            'patient': patients,
            'path':paths,
            'side':sides,
            'view':views,
            'format':formats
                    })
        # Return warning file
        warnings = pd.DataFrame(warnings, columns = ['path', 'warning'])
        
        return data, warnings

class MaiVasWorker(QObject):
    
    finished = Signal()       # Signal that run has ended
    update = Signal(str)    # Signal to widget message box
    progress = Signal(int)    # Progress bar in loop
    image = Signal(Image.Image)
    view_signal = Signal(str) # Signal to connect changes in view attribute to combobox
    format_signal = Signal(str) # Signal to connect changes in format attribute to combobox
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._running = True
        
    def _prevent_execution(self):
        
        # Check save_path is a folder
        if not os.path.isdir(self.model.save_path):
            self.update.emit('WARNING: Save path is not a valid save location or folder.')
            return True
        
        # Check data_path validity
        if not os.path.exists(self.model.data_path):
            self.update.emit('WARNING: Data path is not valid or does not exist.')
            return True
        elif not self.model.data_path[-4:] == ".csv" and not os.path.isfile(os.path.join(self.model.data_path, 'data_file.csv')) and self.model.mode == 2:
            self.update.emit(f'WARNING: Data file needs to be a csv or a folder containing the csv for meta-data mode, not {self.model.data_path}')
            return True
        elif self.model.mode == 1:
            if not os.path.isfile(self.model.data_path) or not pydicom.misc.is_dicom(self.model.data_path):
                self.update.emit(f'WARNING: Data path needs to point to a DICOM, not {self.model.data_path}')
                return True
        elif not os.path.isdir(self.model.data_path) and self.model.mode == 0:
            self.update.emit(f'WARNING: Data path needs to point to a folder, not {self.model.data_path}')
            return True
        
    @Slot()
    def run(self):
        try:
            # Check at the start in case of loading fails.
            if not self._running:
                self.finished.emit()
                return
            
            # Verify parameters
            if self._prevent_execution():
                self.finished.emit()
                return

            # Initial message
            if self.model.mode == 0:
                self.update.emit('Initialising MAI-VAS parameters in folder mode\n')
            elif self.model.mode == 1:
                self.update.emit(f'Initialising MAI-VAS parameters in single-image mode\n')
            else:
                self.update.emit(f'Initialising MAI-VAS parameters in meta-data mode\n')
            self.progress.emit(1)
            
            # Load data
            if self.model.mode == 0:
                self.update.emit('Stand by for processing of images in the folder...')
            else:
                self.update.emit('Stand by for processing of the images...')
            self.model.load_data(progress_signal = self.progress.emit, error_signal = self.update.emit)

            # Check for loading errors:
            if self.model.loader == None:
                self.finished.emit()
                return

            # Connect to Combobox (If self.model.attr is changed internally during init, this ensures the change also happens in the UI.)
            self.view_signal.emit(self.model.view)
            self.format_signal.emit(self.model.format)
            
            # Update UI
            if not self._running:
                self.finished.emit()
                return
            else:
                self.update.emit('Loaded the data matrix!\n')
                self.progress.emit(5)
                if self.model.mode == 1:
                    self.update.emit(f'Single-image mode: autodetected view: {self.model.view} with {self.model.format} image format.')
                elif self.model.mode == 0:
                    self.update.emit(f'Parsing all images in the folder: {self.model.format} image format.')
                    self.update.emit(f'Detected {len(self.model.data_handler.meta_df)} images in folder.')
                else:
                    self.update.emit(f'Parsing images with selected parameters: {self.model.view} with {self.model.format} image format.')

            # Load model
            self.update.emit('\nLoading the AI model parameters matrix...')
            self.model.load_model()

            if not self._running:
                self.finished.emit()
                return
            else:
                self.update.emit('Loaded AI model!\n')
                self.progress.emit(10)
            
            # Init save lists
            name_list = np.empty((len(self.model.loader) * self.model.bs,), dtype=object) # batch size
            output_list = np.empty((len(self.model.loader) * self.model.bs,), dtype=np.float32)
            
            loader_length = len(self.model.loader) # For progres bar only

            # Device message (Not implemented)
            # if self.model.device.type == 'cuda':
            #     self.update.emit('Using GPU for inference.')
            # else:
            #     self.update.emit('GPU not detected, using CPU for inference.')
            self.update.emit('\nBegin mammogram inference...')

            # Main inference loop
            for batch_idx, item in enumerate(self.model.loader):

                # Retrieve batch data
                image = item['image']
                name  = item['name']

                # To device
                image = image.to(self.model.device)

                # Inference
                R = self.model.model(image)
                vas = R.squeeze().detach().cpu().numpy().round(6)

                # Save outputs in lists
                start_idx = batch_idx * self.model.bs
                end_idx = start_idx + len(name)
                output_list[start_idx:end_idx] = vas
                name_list[start_idx:end_idx] = np.array(name)
                
                # Update progress bar
                percent_progress = (1+batch_idx)/loader_length*90 + 10
                if self.model.mode == 0 and self.model.multiple_views:
                    percent_progress = (1+batch_idx)/loader_length*45 + 10
                self.progress.emit(percent_progress)

                # Terminate if stop signal received
                if not self._running:
                    break

            # Do another pass for the other view in folder mode.
            if self.model.mode == 0 and self.model.multiple_views and self._running:
                self.update.emit('\nInitialise model parameters for other mammography view...')

                # Load the other view model
                self.model.model.load_state_dict(torch.load(f'{self.model.model_path}/{self.model.multiple_views}_{self.model.format}_model.pth', map_location=self.model.device, weights_only = True))

                # Load the other view data dataset
                dataset = MammoDataset(data_path = self.model.data_handler.meta_df, view_form = self.model.multiple_views, image_format = self.model.format, from_csv = False)
                loader_kwargs = {'num_workers':4, 'pin_memory':True} if self.model.cuda else {}
                loader = data_utils.DataLoader(dataset,
                                            batch_size = self.model.bs,
                                            shuffle = False,
                                            collate_fn = self.model.collate_fn,
                                        **loader_kwargs)
                
                # Init lists for other view
                name_list_alt = np.empty((len(self.model.loader) * self.model.bs,), dtype=object) #* batch size
                output_list_alt = np.empty((len(self.model.loader) * self.model.bs,), dtype=float)
                
                loader_length = len(loader) # For progres bar only
                self.update.emit('\nContinue mammogram inference of the other view...')

                # Main inference loop for other view
                for batch_idx, item in enumerate(loader):

                    # Retrieve batch data
                    image = item['image']
                    name  = item['name']
                    
                    # Send to device
                    image = image.to(self.model.device)
                
                    # Inference
                    R = self.model.model(image)
                    vas = R.squeeze().detach().cpu().numpy().round(6)
                    
                    # Save outputs in lists
                    start_idx = batch_idx * self.model.bs
                    end_idx = start_idx + len(name)
                    output_list_alt[start_idx:end_idx] = vas
                    name_list_alt[start_idx:end_idx] = np.array(name)

                    # Progress bar update
                    percent_progress = (1+batch_idx)/loader_length*45 + 55
                    self.progress.emit(percent_progress)

                    # Check for termination signal
                    if not self._running:
                        break
                
            # Save new results if completed successfully
            if self._running and self.model.mode == 2:
                test_outdata = pd.DataFrame({'Names':name_list, 'Model_out': output_list})
                test_outdata.to_csv(f'{self.model.save_path}/{self.model.view}_{self.model.format}_results.csv', index = False, float_format='%.6f')

            # Save results if completed successfully
            elif self._running and self.model.mode == 0:
                test_outdata = pd.DataFrame({'Names':np.concatenate([name_list, name_list_alt]), 'Model_out': np.concatenate([output_list, output_list_alt])})
                test_outdata.to_csv(f'{self.model.save_path}/{Path(self.model.data_path).stem}-{self.model.format}_results.csv', index = False, float_format='%.6f')

                # Handle warnings if there are any.
                if not self.model.data_handler.warning.empty:
                    self.model.data_handler.warning.to_csv(f'{self.model.save_path}/WARNINGS.csv', index = False)
                    self.update.emit(f'Some images had issues, see WARNINGS for further details.')
                
            # Display VAS and image if single-image mode
            elif self._running and self.model.mode == 1:
                self.update.emit(f'\n VAS: {vas:.2f}')
                test_outdata = pd.DataFrame({'Names':name_list, 'Model_out': output_list})
                test_outdata.to_csv(f'{self.model.save_path}/{Path(name[0]).stem}_result.csv', index = False, float_format='%.6f')
                self.image.emit(self.to_pil_image(
                    pydicom.read_file(self.model.data_path).pixel_array, 
                    self.model.format
                    ))
                
                # Handle warnings if there are any.
                if not self.model.data_handler.warning.empty:
                    self.model.data_handler.warning.to_csv(f'{self.model.save_path}/WARNINGS.csv', index = False)
                    self.update.emit(f'Some images had issues, see WARNINGS for further details.')

            # Message update
            self.update.emit(f'Inference completed successfully, results have been saved in {self.model.save_path}')
            self.update.emit('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

            # Reset
            self.finished.emit()
        
        except Exception as e:
            self.update.emit(f'An error occurred: {str(e)}') # Print error message to GUI
            self.finished.emit() # Resets
            return

    def to_pil_image(self, image, im_format):
        
        # Otsu's to prevent background
        cut_off = filters.threshold_otsu(image)
        
        # For RAW, apply otsu's, log and invert
        if im_format == 'RAW':
            np.clip(image, 0, cut_off, out = image)
            image = np.log(image)
            np.subtract(np.amax(image), image, out = image)
            
        # For PRO, apply otsu's
        elif im_format == 'PRO':
            np.clip(image, cut_off, np.amax(image), out = image)
            image = (image - np.amin(image)) / (np.amax(image)- np.amin(image))        
        
        # Histogram eq to improve clarity
        image_histogram, bins = np.histogram(image.flatten(), 256, density=True)
        cdf = image_histogram.cumsum() # cumulative distribution function
        cdf = 255 * cdf / cdf[-1] # normalize
        image = np.interp(image.flatten(), bins[:-1], cdf).reshape(image.shape)    
        
        image = Image.fromarray(((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8))
        image.thumbnail((1000, 1000), Image.LANCZOS)
        
        return image
     
    @Slot(str)
    @Slot()
    def stop(self):
        self._running = False
        
    def restart(self):
        self._running = True
        
    @Slot()
    def set_view(self, view):
        self.model.view = view
      
    @Slot()
    def set_format(self, image_format):
        self.model.format = image_format
        
    @Slot()
    def set_data_path(self, data_path):
        self.model.data_path = data_path
        
    @Slot()
    def set_save_path(self, save_path):
        self.model.save_path = save_path
        
    @Slot(int) 
    def set_mode(self, mode):
        self.model.mode = mode