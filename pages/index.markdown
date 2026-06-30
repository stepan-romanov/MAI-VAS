---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---

MAI-VAS is an AI-based model developed at the University of Manchester that can predict expert VAS density assessments from digital mammograms. In support of the model publication, a GUI has been developed to assist in the useability of the model for scientific settings.

# Installation

### 1. Download the executable

Click [here](https://www.dropbox.com/scl/fi/3jlapgjzwu51k7voncqvw/MAI-VAS.exe?rlkey=ztwa3piq1q2f6iji5iti3g5fy&st=ii1he6xt&dl=1) to download the MAI-VAS executable file.

### 2.  Download the model weights

Download the model weights files <a href="https://www.dropbox.com/scl/fo/swolxuige1unyugtmh4sa/AEA4Y9xRD0k7pv32HuGwNmY?rlkey=0unz1vjzq31erw3l63guomgqg&st=66owq1mb&dl=1" onclick="gtag('event', 'download_click', {'event_category': 'engagement', 'event_label': 'weights download'});" rel="noopener">here</a>. There should be 4 files, one for each view and image format combination. Unzip and place the "model" folder with the weights alongside the executable. 

![Model folder location](/assets/images/model_dir_location.png "Model folder location")

### 3. Run

GUI should be ready to go! Note that the first time you launch the app it may take some extra time to download additional files.

# User Guidelines

### 1. Select image input mode

By default there are 3 input modes you can select:

![Input mode image](/assets/images/input_mode_guideline.png "Input mode")

- From folder: MAI-VAS will be run recursievely on all images in a given folder. The programme should auto-detect the view of any image, but you will need to select the image format. The results will be saved in a csv for all of the valid images in the selected folder.
- Single-image mode: MAI-VAS will be run on a one image. Both the view and the image-format will be autodetected. The predicted VAS score will be printed to console, and the image should be automatically displayed. 
- From metadata: MAI-VAS will be run on any files in a csv metadata file. This is useful for when you need to input information by hand due to incomplete dicom headers or complex image paths. The metadata needs to be of a particular format, with a sample found here.

### 2. Select input data and output file locations

Select the input file or folder. The nature of the input will depend on the selected mode. Select where you want for the results csv to be saved. By default it should be placed alongside the executable.

![Path guideline image](/assets/images/data_path_guideline.png "Path guideline")

### 3. Calculate

![Calculate image](/assets/images/calculate_guideline.png "Calculate")

### Warnings

The GUI will generate a WARNINGs fbundile documenting any images with which it had issues. These issues include:

- If the image cannot be opened.
- If there is missing information in the DICOM headers, like view or image format.

Any images that trigger the above warnings will be **skipped**. Additionally, any instances of non **GE Healthcare** images will also be noted in this file, but these instances will **not be skipped**. Do note that MAI-VAS was trained and tested solely using GE mammograms, and may not provide accurate readings for any other providers.

### Metadata file

You can also feed the images into the GUI using metadata file. This is useful if you have images with a lot of missing information, or complex folder structures as you can edit these features in the metadata file. An example file can be found [here](https://www.dropbox.com/scl/fi/ir8a7mc193cvx8i5bjfk2/sample_meta_data_file.csv?rlkey=xfdh2ezyakxz26ya9qnswd9tk&st=l0lm1q2g&dl=1).

![Metadata example](/assets/images/metadata_file.png "Metadata example")

- **patient** - anon id for the image.
- **path** - absolute path to the image.
- **side** - L or R for side.
- **view** - CC or MLO for view.
- **format** - RAW or PRO for image format. 

# Description

### What is VAS?

Visual Analogue Scales (VAS) are a continuous measure of percent breast density. Examiners place a mark on a scale from 0 to 100 corresponding to the percent of fibroglandular tissue. The forms are then electronically scanned and the value recorded. See an example form below.

![VAS form image](/assets/images/VAS_form.png "VAS form")

VAS is less coarse than categoric measures of breast density, and has been shown to have a high correlation with breast cancer risk. More details can be found in this publication that compared VAS against some automated methods [Springer](https://link.springer.com/article/10.1186/s13058-018-0932-z).

### How does MAI-VAS work?

MAI-VAS provides predicted VAS scores, aiming to simulate the opinion of an expert. The model is based on the ResNet-50 architecture and was trained using a large mammographic dataset collected across Greater Manchester. For more architecture details please see our publication [SPIE](https://www.spiedigitallibrary.org/journals/Comparing-percent-breast-density-assessments-of-an-AI-based-method/volume-12/issue-S2/S22011/Comparing-percent-breast-density-assessments-of-an-AI-based-method/10.1117/1.JMI.12.S2.S22011.full). The GUI was developed to simplify the process of using MAI-VAS. Any input images remain strictly on your computer.

### What data did we use for training?

We used images from the Predicting Risk Of Cancer At Screening (PROCAS) study, undertaken in 5 screening areas across Greater Manchester between 2009 and 2015. Consenting women completed a questionnaire about their family history and other breast cancer risk factors at the time of routine mammographic screening. Over the study period, women attening screening in the catchment areas were invited to join the study. Both raw and processed images were retained were possible. All images were recorded using General Electric (GE) Healthcare machines. VAS was independently assessed by two exper readers out of a pool of 19 for each image. Images from a total of 33,408 women were used to develop the model. More information on the study can be found in this paper [link](https://aacrjournals.org/cancerpreventionresearch/article/5/7/943/50026/Assessing-Individual-Breast-Cancer-Risk-within-the).

Evaluation was similarly performed on an independent subset of the PROCAS study. Evaluation details can be found in our paper [SPIE](https://www.spiedigitallibrary.org/journals/Comparing-percent-breast-density-assessments-of-an-AI-based-method/volume-12/issue-S2/S22011/Comparing-percent-breast-density-assessments-of-an-AI-based-method/10.1117/1.JMI.12.S2.S22011.full).

# Limitations & Issues

The MAI-VAS project is a work in-progress which has several issues and limitations that should be acknowledged.

### MAI-VAS limitations

- Mammography vendor: MAI-VAS was trained and tested solely using GE mammograms, and may not work on other providers. Whilst you can get VAS readings for alternative vendors, their accuracy is uncertain. We plan to extend MAI-VAS to other common vendors such as Hologic or Siemens at the end of this year.

### GUI limitations

- DICOM format: The GUI uses the DICOM headers to access important information. Other formats for images such png will not work, unless you use the raw code from git.
- NO GPU version: The GUI uses the CPU only which will be slower for large datasets to keep the install size more manageable. However, the raw code will have functionality for GPUs, and is recommended for large datasets. We plan to add a GPU version in a later release.

### Other

As mentioned previously, MAI-VAS is a work in-progress. If you have any issues, requests or problems, please raise an issue on GitHub or email me directly at [stepan.romanov@manchester.ac.uk](mailto:stepan.romanov@manchester.ac.uk).

# License

Code licensed under the [GPL-3](https://www.gnu.org/licenses/gpl-3.0.html) license.

# Citation

Romanov, S., Howell, S., Harkness, E., Gareth Evans, D., Astley, S., & Fergie, M. (2025). Comparing percent breast density assessments of an AI-based method with expert reader estimates: inter-observer variability. Journal of Medical Imaging, 12(S2), S22011-S22011. [link](https://www.spiedigitallibrary.org/journals/Comparing-percent-breast-density-assessments-of-an-AI-based-method/volume-12/issue-S2/S22011/Comparing-percent-breast-density-assessments-of-an-AI-based-method/10.1117/1.JMI.12.S2.S22011.full)

```
@article{romanov2025comparing,
  title={Comparing percent breast density assessments of an AI-based method with expert reader estimates: inter-observer variability},
  author={Romanov, Stepan and Howell, Sacha and Harkness, Elaine and Gareth Evans, Dafydd and Astley, Sue and Fergie, Martin},
  journal={Journal of Medical Imaging},
  volume={12},
  number={S2},
  pages={S22011--S22011},
  year={2025},
  publisher={Society of Photo-Optical Instrumentation Engineers}
}
```
