# SRGAN-PyTorch
Implementation of Super Resolution Algorithm SRGAN in PyTorch. Paper is available [here](https://arxiv.org/pdf/1609.04802.pdf).

## Introduction
Image super-resolution is the process of enhancing the resolution of a low-resolution image to produce a high-resolution counterpart. SRGAN stands out by achieving this enhancement with an unprecedented level of photorealism, making it particularly valuable for various applications, including image upscaling and enhancement.

The key innovation of SRGAN lies in its use of a Generative Adversarial Network (GAN). GANs consist of two neural networks, a generator, and a discriminator, engaged in a competitive training process. In the context of SRGAN, the generator is responsible for taking a low-resolution input image and producing a high-resolution output, while the discriminator assesses the realism of the generated images.


## Installation
### 1. Clone repository:
```
git clone https://github.com/milojkonikolic/SRGAN-PyTorch.git
cd SRGAN-PyTorch
```
### 2. Setup the environment
This project uses a [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#managing-environments) for running training and evaluation.

The following Python packages are required:
```
torch==2.2.0.dev20231020+cu118
torchvision==0.17.0.dev20231020+cu118
tensorboardX==2.6.2.2
tensorboard==2.8.0
opencv-python==4.8.1.78
numpy==1.24.1
PyYAML==5.4
tqdm==4.66.1
```

To create the conda environment with all required packages run the following commands:
```
conda create -n <name> python=3.9
conda activate <name>
pip3 install -r requirements.txt
```

### 3. GPU info
The model is trained on one NVIDIA GeForce RTX 3070 GPU. Driver Version: 510.47.03, CUDA Version: 11.8

## Data
No annotations are required for training SRGAN. The preparation of data is straightforward. The input data is provided in a JSON file, which contains paths to images to be used for training. Here's an example:
```
[
  "/Datasets/CelebA/img_1.jpg",
  "/Datasets/CelebA/img_2.jpg",
  "/Datasets/CelebA/img_3.jpg",
  "/Datasets/CelebA/img_4.jpg"
]
```
Two JSON files are required for training. One JSON file with training images and the second JSON file with validation images. The split to train and random is done randomly by selecting 20% of data for validation dataset. The format for both train and validation JSON files is the same (shown above).

## Training
All the parameters required for training are specified in the `config.yaml` file:
```
Data:
  train_data_path: "/Datasets/SRGAN/CelebA/train.json"
  val_data_path: "/Datasets/SRGAN/CelebA/val.json"
  input_shape: [192, 128]  # Input shape to the network in the format [width, height]
  downsample_factor: 4  # Downsample factor for low resolution images
  gaussian_blur: True  # Apply Gaussian blur on original images to create low resolution images
  num_workers: 30  # Number of dataloader workers
  gpu: 0  # GPU ID
Generator:
  pretrained_weights: ""
  optimizer: "adam"
  lr: 0.0001
Discriminator:
  pretrained_weights: ""
  optimizer: "adam"
  lr: 0.0001
Hyperparameters:
  batch_size: 16
  epochs: 150
Logging:
  tb_images_per_epoch: 5  # How many images to generate on Tensorboard during an epoch
  train_logs: "/train_logs/SRGAN/train_logs/run1"
```

The input to the GAN model consists of two images: the original image from the dataset resized to the specified `input_shape`, and a low-resolution image. The low-resolution image is obtained by reducing the size of the original image `downsample_factor` times and applying Gaussian blur.

Due to the challenging task of generating realistic images, the generator neural network is trained `generator_num` times more frequently than the discriminator neural network.

To start the training run the command:
```
python3 train.py --config config.yaml
```
Make sure that you changed the path to train and val JSON files and path for saving weights: `train_logs`. Also, feel free to go through other parameters.

Monitor training progress using TensorBoard. Generated images will be displayed on TensorBoard. Run TensorBoard from `train_logs` directory specified in the config file:
```
tensorboard --logdir .
```
TensorBoard displays learning rate, loss, and SSIM. More details about SSIM are below.

## Test
To test the trained generator model on sample of images run the script `test.py` with the following arguments:
```
--gen-model-path - Path to trained generator model - PyTorch file
--input-dir - Path to directory with images for test
--out-dir - Path to output directory for saving images
--high-res-images - [Optional] Path to directory with high resolution images. Only for calculating SSIM
```
Ensure that the images in the input directory are low-resolution images downscaled by the specified factor. In this case `downsample_factor` is 4.

This script is used to verify the model after the training on a sample of images. The output of the script are generated images specified as `out-dir` and SSIM metric if `high-res-images` is provided. The SSIM (Structural Similarity Index) measures the similarity between the original high resolution image and generated image. It quantifies the similarity of structures (patterns, textures) in the images rather than simply measuring pixel-wise differences.

## Results
The model is trained on [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

During the training, the metric that is measured is SSIM. The value for this metric is about 0.85.

TODO: Visual examples and tb logs


## License
SRGAN-PyTorch is licensed under the MIT License.

## Acknowledgments
This project builds upon the SRGAN paper by Christian Ledig et al. (2017).
