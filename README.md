# progrockdiffusion
A command line version of [Disco Diffusion](https://github.com/alembics/disco-diffusion).

# Hardware prerequisites
You will need at least 16gb of RAM if not using a GPU.
An Nvidia GPU is *highly* recommended! The speed improvement is massive.
8gb is probably the minimum amount of GPU memory.

This author has an RTX 3080 with 10gb and it runs fairly well, but some advanced features are not possible with "only" 10gb.

You'll also need between 20 and 40gb of free disk space, depending on which models you enable.

# Software prerequisties
## Linux
Ubuntu 20.04 or similar (A docker environment, VM, or Windows Subsystem for Linux should work provided it can access your GPU).

If using a GPU: CUDA 11.4+ (installation instructions can be found here: https://developer.nvidia.com/cuda-11-4-1-download-archive).

## Windows
Windows 10 or 11 (If using a GPU, NVIDIA drivers installed)
Other versions may work but are untested

## MacOS
Minimal testing has been done with the latest MacOS on an M1 Macbook Air.
PLEASE NOTE: GPU acceleration is not yet supported. It will work, but it will be *slow.*

## If using an NVIDIA GPU, test NVIDIA drivers
You can test that your environment is working properly by running:

```
nvidia-smi
```

The output should indicate a driver version, CUDA version, and so on. If you get an error, stop here and troubleshoot how to get Nvidia drivers, CUDA, and/or a connection to your GPU with the environment you're using.

# First time setup

## Download and install Anaconda
**[Linux]**
```
sudo apt update
sudo apt upgrade -y
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
respond 'yes' to accept license terms and provide install dir when prompted
respond 'yes' to run conda initialization
```
Logout and back in for the changes to take effect


**[Windows]**
```
Download from here and install: https://www.anaconda.com/products/individual
```
From the start menu, open a "Anaconda Powershell Prompt" (*Powershell* is important)

**[MacOS]**

Install Homebrew:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Restart your terminal, then:
```
brew install miniforge
conda init zsh
```
Restart your terminal again.

## Create prog rock diffusion env

**[Linux and Windows]**
```
conda create --name progrockdiffusion python=3.7
```

**[MacOS]**
```
conda create --name progrockdiffusion python=3.8
```

**[All Platforms]**
```
conda activate progrockdiffusion
```

Now change to whatever base directory you want ProgRockDiffusion to go into.
## Clone the prog rock diffusion repo
```
git clone https://github.com/lowfuel/progrockdiffusion.git
cd progrockdiffusion
```
**Note: the "cd" command above is important, as the next steps will add additional libraries and data to ProgRockDiffusion**

**From here on out, this is the directory you'll want to be in when you use the program.**

## Install the required libraries and tools
```
pip install -r requirements.base.txt
```

## Basic or GPU Accelerated PyTorch
You defnitely should install the GPU version if you have an NVIDIA card. It's almost 30x faster.
Otherwise, you can install the CPU version instead (required for MacOS)

### EITHER Install GPU accelerated PyTorch
```
pip install -r requirements.gpu.txt
```

### OR install the basic CPU version of PyTorch (warning - very slow!)
```
pip install -r requirements.cpu.txt
```

## Install remaining libraries and tools

**[MacOS]**

```
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
pip install grpcio
```

**[Linux]** Depending on your Linux platform, you may get an error about libGL.so.1
If you do, try installing these dependencies:
```
sudo apt-get install ffmpeg libsm6 libxext6 -y
```
**[Linux]** Finally:
```
sudo apt install imagemagick
```

# Use

NOTE: On your first run it might appear to hang. Let it go for a good while, though, as it might just be downloading models.
Somtimes there is no feedback during the download process (why? Who knows)

If you've opened a new terminal or powershell prompt, you may need to activate your ProgRockDiffusion session again:
```
conda activate progrockdiffusion
```

CD to the directory where you installed ProgRockDiffusion. Now you're ready!

The simplest way to run it is:

[Linux]
```
python3 prd.py
```
[Windows and MacOS]
```
python prd.py
```
This will generate an image using the settings from "settings.json", which you could edit to adjust the defaults (or, better yet, make a copy of it and tell prd to use an alternative settings file using the command line arguments below).

**Note: On windows you'll type "python" instead of "python3" in the commands below.**

```
usage: python3 prd.py [-h] [-s SETTINGS] [-o OUTPUT] [-p PROMPT]

Generate images from text prompts.
By default, the supplied settings.json file will be used.
You can edit that, and/or use the options below to fine tune:

Optional arguments:
  -h, --help            show this help message and exit
  -s SETTINGS, --settings SETTINGS
                        A settings JSON file to use, best to put in quotes
  -o OUTPUT, --output OUTPUT
                        What output directory to use within images_out
  -p PROMPT, --prompt PROMPT
                        Override the prompt
  -i, --ignoreseed
                        Use a random seed instead of what is in your settings file

  -c, --cpu CORES
                        Force CPU mode, and (optionally) specify how many threads to run.

  -g, --geninit:
                        Will save an image called geninit.png at 20% of overall steps, for use below:

  -u, --useinit:
                        Forces use of geninit.png as an init_image starting at 20% of defined steps.

Usage examples:

To use the Default output directory and settings from settings.json:
 python3 prd.py

To use your own settings.json file (note that putting it in quotes can help parse errors):
 python3 prd.py -s "some_directory/mysettings.json"

To quickly just override the output directory name and the prompt:
 python3 prd.py -p "A cool image of the author of this program" -o Coolguy

Multiple prompts with weight values are supported:
 python3 prd.py -p "A cool image of the author of this program" -p "Pale Blue Sky:.5"

You can ignore the seed coming from a settings file by adding -i, resulting in a new random seed

To force use of the CPU for image generation, add a -c or --cpu (warning: VERY slow):
 {python_example} prd.py -c
```
Simply edit the settings.json file provided, or copy it and make several that include your favorite settings, if you wish to tweak the defaults.

# Tips and Troubleshooting
## If you get an error about pandas needing a different verison of numpy, you can try:
```
pip install --force-reinstall numpy
```
## If you are getting "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized"
This seems to be because of a Pytorch compiling bug for Intel CPUs. 
You can set an environment variable that will fix this, either on your machine (if you know how to do that), or by editing prd.py.
To do it by editing prd.py, find the line that says "import os" and add the following right below it:
```
os.environ['KMP_DUPLICATE_LIB_OK']='True'
```

## Switch between GPU and CPU modes
Let's assume you installed the GPU version. You can adjust these instructions if you did CPU first, of course.
Clone your existing conda environment:
```
conda create --name prdcpu --clone progrockdiffusion
conda activate prdcpu
```
Now install the CPU version of pytorch:
```
pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install torchvision==0.12.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install torchaudio==0.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```
All set! You can now switch between the two by simply doing:
```
conda activate progrockdiffusion
conda activate prdcpu
```

# Notes

- Currently Superres Sampling doesn't work, it will crash.
- Animations are untested but likely not working

# TODO

- Get Animations working
