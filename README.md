# SAFDN
A Point Cloud Denoising Method in Adverse Weather
## Dataset
Information: Click [here](https://www.uni-ulm.de/index.php?id=101568) for registration and download.
## Requirements
- Install PyTorch
- `pip install -r requirements.txt`
- Download the [DENSE CNN denoising](https://www.uni-ulm.de/index.php?id=101568) dataset, and extract this dataset into the corresponding subfolder of the "cnn_denoising" directory.
## Usage
Training the model: Use Python files starting with "train" for training various models.

Testing the model: Utilize the "getlable.py" script to load the saved model, classify points in the HDF5 file, and then save the generated classification labels.
