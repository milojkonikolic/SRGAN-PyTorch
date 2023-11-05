# SRGAN-PyTorch
Implementation of Super Resolution Algorithm SRGAN in PyTorch. Paper is available [here](https://arxiv.org/pdf/1609.04802.pdf).

## Introduction
Image super-resolution is the process of enhancing the resolution of a low-resolution image to produce a high-resolution counterpart. SRGAN stands out by achieving this enhancement with an unprecedented level of photorealism, making it particularly valuable for various applications, including image upscaling and enhancement.

The key innovation of SRGAN lies in its use of a Generative Adversarial Network (GAN). GANs consist of two neural networks, a generator, and a discriminator, engaged in a competitive training process. In the context of SRGAN, the generator is responsible for taking a low-resolution input image and producing a high-resolution output, while the discriminator assesses the realism of the generated images.

## Usage

### Environment
This project uses a [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#managing-environments) for running training and evaluation. The following Python packages are required:
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

### Data
No annotations are required for training SRGAN. The data format for training is straightforward. The input data is provided in a JSON file, which contains paths to images to be used for training. Here's an example:
```
[
  "/Datasets/CelebA/img_1.jpg",
  "/Datasets/CelebA/img_2.jpg",
  "/Datasets/CelebA/img_3.jpg",
  "/Datasets/CelebA/img_4.jpg"
]
```
### Running Training
All the parameters required for training are specified in the `config.yaml` file:
```
Data:
  train_data_path: "/Datasets/SRGAN/CelebA/train.json"
  val_data_path: "/Datasets/SRGAN/CelebA/val.json"
  input_shape: [192, 128]  # Input shape to the network in the format [width, height]
  downsample_factor: 4  # Downsample factor for low resolution images
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
  epochs: 10
  generator_num: 5  # Train generator generator_num times more often than discriminator
Logging:
  tb_images_per_epoch: 5  # How many images to generate on Tensorboard during an epoch
  train_logs: "/train_logs/SRGAN/train_logs/run1"
```

The input to the GAN model consists of two images: the original image from the dataset resized to the specified `input_shape`, and a low-resolution image. The low-resolution image is obtained by reducing the size of the original image `downsample_factor` times and applying Gaussian blur.

Due to the challenging task of generating realistic images, the generator neural network is trained generator_num times more frequently than the discriminator neural network.

### Inference


### Results
