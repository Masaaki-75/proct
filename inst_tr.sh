# One may encounter issues when installing the torch-radon toolbox for the first time.
# Here's how I install without triggering many errors.

# You may install torch-radon v2 if you are using higher versions of pytorch/CUDA.
# Torch-radon v2 has more features, and enjoys faster processing speed since it
# supports higher CUDA driver version. 
# The v2 package is maintained at: https://github.com/carterbox/torch-radon 
# Or you can use my fork: https://github.com/Masaaki-75/torch-radon. However,
# I found replacing v1 with v2 leads to slight performance drop, and I am still
# looking into why. Anyway, here's the walkthrough:

# ============================ Torch-Radon V1 ============================ #
# Step 1: build a new environment
conda create -n tr37 python==3.7
conda activate tr37

# Step 2: set up basic pytorch
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
#pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Step 3: download code from https://github.com/matteo-ronchetti/torch-radon 
# (master branch), `cd` to the directory, and run the setup:
python setup.py install

# Step 4: install other related packages
conda install -c astra-toolbox astra-toolbox
conda install matplotlib
pip install einops
pip install opencv-python
conda install pillow
conda install scikit-image
conda install scipy==1.6.0
conda install wandb
conda install tqdm
# ======================================================================== #


# Installing torch-radon v2 is similar, just using a different repository.
# ============================ Torch-Radon V2 ============================ #
# Step 1: build a new environment
conda create -n tr39 python==3.9
conda activate tr39

# Step 2: set up basic pytorch
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Step 3: download code from https://github.com/carterbox/torch-radon
# and run the setup.py with the following:
python setup.py install

# Step 4: install other related packages
conda install -c astra-toolbox astra-toolbox
conda install matplotlib
pip install einops
pip install opencv-python
conda install pillow
conda install scikit-image 
conda install wandb
conda install tqdm
# ======================================================================== #


