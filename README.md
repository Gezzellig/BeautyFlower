# :cherry_blossom: Super Resolution on Images :cherry_blossom:

## Instructions
### Prepare program
1. Clone source to any directory.
2. Install dependencies.

### Prepare dataset
1. Go to: https://data.vision.ee.ethz.ch/cvl/DIV2K/
2. Download here the following:
	a. Train Data Track 1 bicubic downscaling x4 (LR images)
	b. Validation Data Track 1 bicubic downscaling x4 (LR images)
	c. Train Data (HR images)
	d. Validation Data (HR images)
3. Update the config.py to match the paths to the dataset folder.

### Hyperparameters
These can be adjusted in the config.py file.

### Run
1. python3 main.py --numResidualBlocks <number_of_residual_blocks>

The code is extracted from https://github.com/tensorlayer/srgan and modified to answer our research question: What is the effect of using less residual blocks.

### Generate images using trained model
1. Choose the model that you want to use to rescale the images. These models can be found in output/<your_current_run>/checkpoint/g_srgan<num-epoch>.npz
2. Make a folder containing the images that have to be enlarged
3. Make an output folder
4. run: Python3 genImage.py <model_path> <num_residual_blocks> <input_folder> <output_folder>



