# Universal Incomplete-View CT Reconstruction with Prompted Contextual Transformer
This repository contains the official implementation of the paper: "Universal Incomplete-View CT Reconstruction with Prompted Contextual Transformer".

🚧We are currently cleaning and reformatting the code, please stay tuned!🚧

## Abstract
> Despite the reduced radiation dose, suitability for objects with physical constraints, and accelerated scanning procedure, incomplete-view computed tomography (CT) images suffer from severe artifacts, hampering their value for clinical diagnosis. The incomplete-view CT can be divided into two scenarios depending on the sampling of projection, sparse-view CT and limited-angle CT, each encompassing various settings for different clinical requirements. Existing methods tackle with these settings separately and individually due to their significantly different artifact patterns; this, however, gives rise to high computational and storage costs, hindering its flexible adaptation to new settings. To address this challenge, we present the first-of-its-kind all-in-one incomplete-view CT reconstruction model with PROmpted Contextual Transformer, termed ProCT. More specifically, we first devise the projection view-aware prompting to provide setting-discriminative information, enabling a single model to handle diverse incomplete-view CT settings. Then, we propose artifact-aware contextual learning to provide the contextual guidance of image pairs from either CT phantom or publicly available datasets, making ProCT capable of accurately removing the complex artifacts from the incomplete-view CT images. Extensive experiments demonstrate that ProCT can achieve superior performance on a wide range of incomplete-view CT settings using a single model. Remarkably, our model with only image-domain information surpasses the state-of-the-art dual-domain methods that require the access to raw data.


## Updates
- [ ] code coming soon.
- [x] 2023/12/13. Initial commit.


## Requirements
```
- python==3.7.16
- torch==1.7.1+cu110  # depends on the CUDA version of your machine
- torchaudio==0.7.2
- torchvision==0.8.2+cu110
- torch-radon==1.0.0
- monai==1.0.1
- scipy==1.7.3
- einops==0.6.1
- opencv-python==4.7.0.72
- SimpleITK==2.2.1
- numpy==1.21.6
- pandas==1.3.5  # optional
- tensorboard==2.11.2  # optional
- wandb==0.15.2  # optional
- tqdm==4.65.0  # optional
```
