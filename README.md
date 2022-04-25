# Nonuniform-to-Uniform Quantization

This repository contains the training code of N2UQ introduced in our CVPR 2022 paper: "[Nonuniform-to-Uniform Quantization: Towards Accurate Quantization via Generalized Straight-Through Estimation](https://arxiv.org/abs/2111.14826)"

In this study, we propose a quantization method that can learn the non-uniform input thresholds to maintain the strong representation ability of nonuniform methods, while output uniform quantized levels to be hardware-friendly and efficient as the uniform quantization for model inference.

<div align=center>
<img width=60% src="https://github.com/liuzechun/Nonuniform-to-Uniform-Quantization/blob/main/U2UQ_github.jpg"/>
</div>
    
To train the quantized network with learnable input thresholds, we introduce a generalized straight-through estimator (G-STE) for intractable backward derivative calculation w.r.t. threshold parameters.

The formula for N2UQ is simply as follows,

Forward pass:

<div align=center>
<img width=40% src="https://github.com/liuzechun/Nonuniform-to-Uniform-Quantization/blob/main/Formula01.jpg"/>
</div>


Backward pass:
<div align=center>
<img width=40% src="https://github.com/liuzechun/Nonuniform-to-Uniform-Quantization/blob/main/Formula02.jpg"/>
</div>


Moreover, we proposed L1 norm based entropy preserving weight regularization for weight quantization.



## Citation

If you find our code useful for your research, please consider citing:
    
    @inproceedings{liu2022nonuniform,
      title={Nonuniform-to-Uniform Quantization: Towards Accurate Quantization via Generalized Straight-Through Estimation},
      author={Liu, Zechun and Cheng, Kwang-Ting and Huang, Dong and Xing, Eric and Shen, Zhiqiang},
      journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022}
    }
    
## Run

### 1. Requirements:
* python 3.6, pytorch 1.7.1, torchvision 0.8.2
* gdown
    
### 2. Data:
* Download ImageNet dataset

### 3. Pretrained Models:
* pip install gdown \# gdown will automatically download the models
* If gdown doesn't work, you may need to manually download the [pretrained models](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/Eqrxgh_tbOVHkp9bqhWukaoBA8thLh0eCtOTdSxhEjsWuw?e=cw9ydr) and put them in the correponding `./models/` folder.

### 4. Steps to run:
(1) For ResNet architectures:
* Change directory to `./resnet/`
* Run `bash run.sh architecture n_bits quantize_downsampling`
* E.g., `bash run.sh resnet18 2 0` for quantize resnet18 to 2-bit without quantizing downsampling layers

(2) For MobileNet architectures:
* Change directory to `./mobilenetv2/`
* Run `bash run.sh`
       
## Models

### 1. ResNet

| Network | Methods | W2/A2 | W3/A3 | W4/A4 |
| --- | --- | --- | --- | --- |
| ResNet-18 | | | |
| | PACT | 64.4  | 68.1 | 69.2 |
| | DoReFa-Net | 64.7 | 67.5 | 68.1 | 
| | LSQ | 67.6 | 70.2 | 71.1 |
| | **N2UQ** | **69.4** [Model-Res18-2bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EVP4Ie28TDlHm3mSWdjvAk4BwdksW8fvoNSg11B3FilHEA?e=7PAOAd) | **71.9** [Model-Res18-3bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EdinnTKYXktBmuFD2SksafsBAJcp536gV-R0G46fMWRbiA?e=deqLyP) | **72.9** [Model-Res18-4bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EZKIm_I5AqxJmABKCKFojegB1JleuC__2KIFjIbT3ItD4A?e=8YcGaR) |
| | **N2UQ** \* | **69.7** [Model-Res18-2bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EeEndtQyWLdEowoUcRG-DjIB5aQWGKCkP2U59MfshyX0fA?e=EkJ8GQ) | **72.1** [Model-Res18-3bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/ET-NhKk0ZiFBrcFUGVvoQBMB9z9bRR_JPtA7bqYV4Wqu0A?e=vTuSq3) | **73.1** [Model-Res18-4bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EbkGs6B9FTlKr98PsoLTVfsB3zY0twxasPAUncE8eu5S2A?e=yQAjnp) |
| ResNet-34 | | | |
| | LSQ | 71.6 | 73.4 | 74.1 |
| | **N2UQ** | **73.3** [Model-Res34-2bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EeYYBsUmd8hFi6K8PL1_ungBuj8JD2X4SfCZMR9EWg3KVw?e=igOAwz) | **75.2** [Model-Res34-3bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/Ea4Asqw9rVBNlhfKagWrdB0BW_j0RfHqXUf0q1QdcnGHcw?e=SzsGDM) | **76.0** [Model-Res34-4bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EZLd9m4OxZxKpHOWaB6wdF4BF4oZAg-Xok5JOC1M_Y9jsg?e=KeuA41) |
| | **N2UQ** \* | **73.4** [Model-Res34-2bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/Eaj_R8tZ8k1MuagOxVPvZ8cBEAgbmjPeE0WhNvJM04msoQ?e=IAw64e) | **75.3** [Model-Res34-3bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EcRHSaYTgRZGuG9xJgjlBHMBPwR3s14bbNzc9KuQSQhGhA?e=mtBEEK) | **76.1** [Model-Res34-4bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EQuLjGtwCXNAnzAG5GcTTKoB22F48tLqvYX5nn2XsZ4_4Q?e=P5SlNO) |
| ResNet-50 | | | |
| | PACT | 64.4  | 68.1 | 69.2 |
| | LSQ | 67.6 | 70.2 | 71.1 |
| | **N2UQ** | **75.8** [Model-Res50-2bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EZ4Ya2NykY5BoaR3_zHVJIUBvw8fEQ3NLvO8TcrezInNyg?e=ozHw9F) | **77.5** [Model-Res50-3bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EcOgGqzybEJOuo8pHuSzF2UBbow-FhMUfnIXPfhbMTCF_Q?e=1SSWwx) | **78.0** [Model-Res50-4bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/ESm2mabqyC1EleyaBQL8NpsBzi8XDPalFz88KV0ioJNr2Q?e=aTQJgt) |
| | **N2UQ** \* | **76.4** [Model-Res50-2bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EdEe4J5szAtOosuS9AbBs0kBlLs0QnF4cKqjDpdVl39Zug?e=KpANaF) | **77.6** [Model-Res50-3bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EUMCYazwLU1FioPLpdlVYCEBZZNop4eFKWVBR6nu9M1j7g?e=Wo8deQ) | **78.0** [Model-Res50-4bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EVlBJCvBVFRGmc_eXbmFkjYBoTr3E-_6AjRfHOp5kSCy3w?e=yyxEfd) |

<!-- | LQ-Nets | 64.9 | 68.2 | 69.3 | -->

Note that N2UQ without \* denotes quantizing all the convolutional layers except the first input convolutional layer.

N2UQ with \* denotes quantizing all the convolutional layers except the first input convolutional layer and three downsampling layers.

W2/A2, W3/A3, W4/A4 denote the cases where the weights and activations are both quantized to 2 bits, 3 bits, and 4 bits, respectively.

### 2. MobileNet

| Network | Methods | W4/A4 |
| --- | --- | --- | 
| MobileNet-V2 | **N2UQ** \* | **72.1** [Model-MBV2-4bit](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EYejI78B4TBFvG4d_nSPDfYBCkiv9QJKiBaMl8kJqzg5Xw?e=8UeKFm) |

## Contact

Zechun Liu, HKUST (zliubq at connect.ust.hk)
