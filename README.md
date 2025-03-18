<p align=center><img src="figs/logo_new.png" width="200px"> </p>

# <p align=center> `CORUN 🏃 Colabator 🏃🏻‍♂️` </p>

<b><p align=center> <a href='https://arxiv.org/pdf/2406.07966'><img src='https://img.shields.io/badge/ArXiv-2406.07966-red'></a>
  NeurIPS2024 Spotlight ✨</p></b>

This is the official PyTorch codes for the paper. 
>**Real-world Image Dehazing with Coherence-based Pseudo Labeling and Cooperative Unfolding Network** <br> [Chengyu Fang](https://cnyvfang.github.io/), [Chunming He](https://chunminghe.github.io/), Fengyang Xiao, [Yulun Zhang](https://yulunzhang.com), Longxiang Tang, Yuelin Zhang, [Kai Li](https://kailigo.github.io), and Xiu Li <br>
> Advances in Neural Information Processing Systems 2024<br>

[//]: # (**Abstract:** Real-world Image Dehazing &#40;RID&#41; aims to alleviate haze-induced degradation in real-world settings. This task remains challenging due to the complexities in accurately modeling real haze distributions and the scarcity of paired real-world data. To address these challenges, we first introduce a cooperative unfolding network that jointly models atmospheric scattering and image scenes, effectively integrating physical knowledge into deep networks to restore haze-contaminated details. Additionally, we propose the first RID-oriented iterative mean-teacher framework, termed the Coherence-based Label Generator, to generate high-quality pseudo labels for network training. Specifically, we provide an optimal label pool to store the best pseudo-labels during network training, leveraging both global and local coherence to select high-quality candidates and assign weights to prioritize haze-free regions. We verify the effectiveness of our method, with experiments demonstrating that it achieves state-of-the-art performance on RID tasks.  )


<details>
<summary>🏃 The architecture of the proposed CORUN with the details at k-th stage (CORUN)</summary>
<center>
    <img
    src="figs/Arch.jpg">
</center>
</details>

<details>
<summary>🏃🏻‍♂️ The plug-and-play Coherence-based Pseudo Labeling paradigm (Colabator)</summary>
<center>
    <img
    src="figs/CPL.jpg">
    <br>
</center>
</details>



## 🔥 News
- **2025-01-17:** We upload a simple example to pretrain and fine-tune with our models. 🎉
- **2024-11-06:** We fix some bugs in the code and support the single GPU training now. 🐞
- **2024-10-26:** Our results and pre-trained weights have been released! ❤️
- **2024-10-23:** We are processing the camera-ready version of this paper, the pretrained weights and test results will be released soon.
- **2024-09-26:** This paper has been accepted by **NeurIPS 2024 as a Spotlight Paper**. 🎉 Thanks all the participants, reviewers, chairs and committee. We will release the code soon.

[//]: # (- **2024-07-26:** We have organized and refined the code for the Colabator framework into a separate repository to facilitate more efficient application across different networks and tasks. The repository of Colabator Template will be made public together with the code of this repository once the paper is accepted. 😚)

[//]: # (- **2024-06-13:** We release the preprint and the citation.)

[//]: # (- **2024-06-12:** We release the results and acknowledgements of this work.)

[//]: # (- **2024-05-28:** We release this repository, the preprint of full paper will be release soon.)

## 🎺 Something You Should Know Before Taste

We provide two types of dataset loading functions for model training: **1. loads clean images and corresponding depth maps to generate hazy images** using the RIDCP Online Haze Generation Pipeline, **2. directly loads paired clean and degraded images**. You can choose the appropriate method based on your dataset and task.

```diff
+ 1. For image dehazing task with online haze generation. (Our CORUN used)
```
We support loading the depth map from .npy (used by RICDP500) of .mat files (used by OTS/ITS). You can also use depth estimation methods like Depth Anything or RA-Depth to construct the depth maps for your own dataset and save as .npy files.

To train or fine-tune our CORUN or any other Image Dehazing methods by online haze generation. **Please refer to** [HERE](https://github.com/cnyvfang/CORUN-Colabator?tab=readme-ov-file#-for-image-dehazing-task-corun-as-example)

```diff
+ 2. For any image restoration tasks with offline paired degraded-clean images.
```
To train or fine-tune any Image-to-Image based Image Restoration tasks (also including image dehazing task). **Please refer to** [HERE](https://github.com/cnyvfang/CORUN-Colabator?tab=readme-ov-file#-for-any-image-restoration-tasks-restormer-as-example).


## ⚙️ Dependencies and Installation

### 1. Download Datasets (Optional)

- **RTTS** dataset can be downloaded from [Dropbox](https://utexas.app.box.com/s/2yekra41udg9rgyzi3ysi513cps621qz).
- **URHI** dataset can be downloaded from [Dropbox](https://utexas.app.box.com/s/7hu094vwkw0cwowv5wijwv9pure2fvup).
- **Duplicate Removed URHI** can be downloaded from [Google Drive](https://drive.google.com/file/d/1B29LsNhBWoRHDk2R_cc5nNqcn7c87sg-/view?usp=sharing)
- **RIDCP500** can be downloaded from [RIDCP's Repo](https://github.com/RQ-Wu/RIDCP_dehazing)

### 2. Download Necessary Pretrained Weights

Download the pre-trained da-clip weights and place it in `./pretrained_weights/`. You can download the daclip weights we used from [Google Drive](https://drive.google.com/file/d/1bIlKYouxwizQXbud7SXd5F5oOyoHFH4x/view?usp=sharing). You can also choose other type of clip models and corresponding weights from openclip, if you do this, don't forget to modify your options.


### 3. Initialize Conda Environment and Clone Repo

⚠️ To ensure consistency of the results, we recommend following our package version to install dependencies.

```bash
git clone https://github.com/cnyvfang/CORUN-Colabator.git
conda create -n corun_colabator python=3.9
conda activate corun_colabator
# If necessary, Replace pytorch-cuda=? with the compatible version of your GPU driver.
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

```

### 4. Install Modified BasicSR
```bash
cd basicsr_modified
pip install tb-nightly -i https://mirrors.aliyun.com/pypi/simple # Run this line if in Chinese Mainland
pip install -r requirements.txt
python setup.py develop
cd ..
```

### 5. Install Our CORUN-Colabator
```bash
pip install -r requirements.txt
python setup.py develop
python init_modules.py
```
🚩 If you are in China Mainland, run the script before init_modules.py to speed up the download of the pre-trained models:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```


## 🏃 For Image Dehazing Task (CORUN as Example)
If you want to use other network to replace our CORUN, you only need to add your network to [archs](corun_colabator/archs), replace the network definition in option files and run the script. If you need to define your own loss function and training strategies, you need to modify the corresponding [models](corun_colabator/models) and [losses](corun_colabator/losses) before you invoke it in option files.

⚠️ **Please replace the dataset path in the corresponding option files with your own dataset path.**

### 1. Pretraining
This step can be skipped IF YOU DO NOT USE OUR CORUN, and have already well-trained your model in your framework.

```bash
# Multi-GPU
sh dehazing_options/train_corun_by_depth.sh
# Single-GPU (Not recommended)
sh dehazing_options/train_corun_by_depth_single_gpu.sh

```
### 2. Fine-tuning with Colabator
```bash
# Multi-GPU
sh dehazing_options/train_corun_with_colabator_by_depth.sh
# Single-GPU (Not recommended)
sh dehazing_options/train_corun_with_colabator_by_depth_single_gpu.sh
```


## 🏃 For Any Image Restoration Tasks (Restormer as Example)
If you want to use other network to replace Restormer, you only need to add your network to [archs](corun_colabator/archs), replace the network definition in option files and run the script. If you need to define your own loss function and training strategies, you need to modify the corresponding [models](corun_colabator/models) and [losses](corun_colabator/losses) before you invoke it in option files.

⚠️ **Please replace the dataset path in the corresponding option files with your own dataset path.**

### 1. Pretraining
This step can be skipped if you have already well-trained your model in your framework.
```bash
# Multi-GPU
sh image_restoration_options/train_stage1_restormer.sh
# Single-GPU
sh image_restoration_options/train_stage1_restormer_single_gpu.sh

```
### 2. Fine-tuning with Colabator
```bash
# Multi-GPU
sh image_restoration_options/train_stage2_restormer_with_colabator.sh
# Single-GPU
sh image_restoration_options/train_stage2_restormer_with_colabator_single_gpu.sh
```

## 🏃‍♂️ Testing Our CORUN+
Download the pre-trained CORUN weight and place it in `./pretrained_weights/`. You can download the CORUN+ weight from [Google Drive](https://drive.google.com/file/d/18afbgAOLYYr8Ef4JsUtz8WNl9xzQ3cd9/view?usp=sharing). **To quickly use the results of our experiments without manual inference or retraining, you can download all results dehazed/restored by our model from [Google Drive](https://drive.google.com/file/d/1yhaNHc8eV-88Bn7YKa7PehZsUddRBjA1/view?usp=sharing).** 


### 1. Inference
```bash
CUDA_VISIBLE_DEVICES=0 sh dehazing_options/valid.corun.sh
# OR
CUDA_VISIBLE_DEVICES=0  python3  corun_colabator/simple_test.py \
  --opt dehazing_options/valid_corun.yml \
  --input_dir /path/to/testset/images  \
  --result_dir ./results/CORUN \
  --weights ./pretrained_weights/CORUN.pth \
  --dataset RTTS
```

### 2. Evalutation
Caculate the NIMA and BRISQUE results.
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --input_dir /path/to/results
```

## 🔍 Results

We achieved state-of-the-art performance on *RTTS* and *Fattal's* datasets and corresponding downstream tasks. More results can be found in the paper.

<details>
<summary>Quantitative Comparison (click to expand)</summary>

- Quantitative results on RTTS
  <p align="center">
  <img width="900" src="figs/tab-1.jpeg">
  </p>
- User study scores on RTTS and Fattal’s data
  <p align="center">
  <img width="900" src="figs/tab-2.png">
  </p>
- Object detection results on RTTS
  <p align="center">
  <img width="900" src="figs/tab-3.png">
  </p>  
  </details>

<details> 
<summary>Visual Comparison (click to expand)</summary>

- Visual comparison on RTTS
  <p align="center">
  <img width="900" src="figs/RTTS.jpg">
  </p>
- Visual comparison on Fattal’s data
  <p align="center">
  <img width="900" src="figs/Fattal.jpg">
  </p>
- Visual comparison of object detection on RTTS
  <p align="center">
  <img width="900" src="figs/detection.jpg">
  </p>
  
  </details>


## 📎 Citation

If you find the code helpful in your research or work, please cite the following paper(s).

```
@inproceedings{fang2024real,
  title={Real-world Image Dehazing with Coherence-based Pseudo Labeling and Cooperative Unfolding Network},
  author={Fang, Chengyu and He, Chunming and Xiao, Fengyang and Zhang, Yulun and Tang, Longxiang and Zhang, Yuelin and Li, Kai and Li, Xiu},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```


## 💡 Acknowledgements
The codes are based on [BasicSR](https://github.com/XPixelGroup/BasicSR). Please also follow their licenses. Thanks for their awesome works.
