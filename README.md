# **DAYI - Backend**

This repository contains the implementation for end-to-end custom Temporal Action Localization. With the idea of the code being used along with a UI, the pipeline has been split into the following three sections:

1. Feature Extraction
2. Model Training
3. Inference

The [inference model](https://github.com/dingfengshi/tridetplus/tree/main) we are using (SOTA of October 2023) requires pre-extracted features. The best performing model uses the [InternVideo](https://github.com/OpenGVLab/InternVideo/tree/main) backbone for feature extraction. Because this process is time consuming, we decoupled it from training and inference. In this way, the user can start extraction upon upload of a new video.

## **Installation**

Step 1. Create a conda env in python 3.11
```
conda create -n backend python=3.11
conda activate backend
```

Step 2. Install the required packages
`python -m pip install -r requirements.txt`

Step 3. Install nms_1d_cpu:
```
cd ./src/libs/utils
python setup.py install
cd ../../..
```

## **Data Preparation**

To download the THUMOS subset videos

```
cd ./src/data/thumos/videos/
source download.sh
cd ../../../..
```

The project data should have the following structure:

```
data
└───subset/
│    ...  
│
└───thumos/
│    └───annotations/
│    │	 └───subset_thumos.json
│    │	 └───thumos14.json
│    └───videos/
│    	 └───download.sh
│    	 └───video_test_0000051.mp4
│    	 │    ...
│    	 └───video_validation_0000054.mp4
│    	 │    ...
```

## **Feature Extraction**

The feature extraction model we use is the [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2) model. The VideoMAEv2 repository offers pretrained checkpoints that we downloaded and used as a frozen backbone. Various checkpoints come with different inference speeds, model sizes, and accuracies. The user can pick which model they prefer based on their needs.

**Extraction Parameters**:

| Name | Type | Default | Description |
| -------- | -------- | -------- | -------- |
| data_dir     | *dir*     | None     | Path of the directory containing the videos to extract features from. |
| save_dir     | *dir*     | None     | Path of the directory to save the extracted features in. |
| base_model     | *string*     | None     | Name of the base model that the checkpoint is using. Refer to the checkpoint repo to see which one works. |
| ckpt_path     | *path*     | None     | Path to the checkpoint file. |
| device     | *string*     | None     | Cuda device to run the extraction on. For now, it is only possible to run it on one device at a time. |
| subset_path     | *path*     | None     | Path to the json file containing a list of video ids. Extraction will only be run on this subset. |

Example extraction on a subset of THUMOS14 with a [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md) pretrained distilled model, checkpoint `vit_s_k710_dl_from_giant.pth`.

```
cd ./src
python extract.py \
    --data_dir ./data/thumos/videos \
    --save_dir ./data/example \
    --base_model vit_small_patch16_224 \
    --ckpt_path ./ckpt/videomae/vit_s_k710_dl_from_giant.pth \
    --device cuda:0 \
    --subset_path data/example/example_subset.json
```

## **Training and Evaluation**

The prediction head is the [TriDet](https://github.com/dingfengshi/tridetplus/tree/main) model.

**Training Parameters**

| Name | Type | Default | Description |
| -------- | -------- | -------- | -------- |
| model_config     | *path*     | None     | Path to the yaml file containing the model information. |
| dataset_config     | *path*     | None     | Path to the yaml file containing the dataset information. |
| save_dir     | *dir*     | None     | Path of the directory where the checkpoints will be saved in. |
| data_dir     | *dir*     | None     | Path of the directory containing the training features. |
| validation_step     | *int*     | None     | Number of epochs between validating. |
| ckpt_freq     | *int*     | None     | Number of epochs between checkpointing. |
| device     | *string*     | None     | Cuda device to run the training on. For now, it is only possible to run it on one device at a time. |

```
cd ./src
python train.py \
    --model_config ./configs/model/base_model.yaml \
    --dataset_config ./configs/dataset/base_data.yaml \
    --save_dir ./ckpt/example/ \
    --data_dir ./data/example/ \
    --validation_step 5 \
    --ckpt_freq 10 \
    --device cuda:0
```

**Eval Parameters**

| Name | Type | Default | Description |
| -------- | -------- | -------- | -------- |
| model_config     | *path*     | None     | Path to the yaml file containing the model information. |
| dataset_config     | *path*     | None     | Path to the yaml file containing the dataset information. |
| data_dir     | *dir*     | None     | Path of the directory containing the evaluation features. |
| ckpt_path     | *path*     | None     | Path to the model checkpoint file. |
| device     | *string*     | None     | Cuda device to run the evaluation on. For now, it is only possible to run it on one device at a time. |

```
cd ./src
python eval.py \
    --model_config ./configs/model/base_model.yaml \
    --dataset_config ./configs/dataset/base_data.yaml \
    --data_dir ./data/example/ \
    --ckpt_path ./ckpt/example/epoch_029.pth.tar \
    --device cuda:0
```

## **Inference**

To deploy on new videos, change the directories in the dataset_config (yaml) file for features directory. (To be fixed)

**Inference Parameters**

| Name | Type | Default | Description |
| -------- | -------- | -------- | -------- |
| model_config     | *path*     | None     | Path to the yaml file containing the model information. |
| dataset_config     | *path*     | None     | Path to the yaml file containing the dataset information. |
| ckpt_path     | *path*     | None     | Path to the model checkpoint file. |
| save_dir     | *dir*     | None     | Path of the directory where the annotations will be saved in. |
| data_dir     | *dir*     | None     | Path of the directory containing the inference features. |
| device     | *string*     | None     | Cuda device to run the inference on. For now, it is only possible to run it on one device at a time. |
| subset_path     | *path*     | None     | Path to the json file containing a list of video ids. Extraction will only be run on this subset. |

```
cd ./src
python deploy.py \
    --model_config ./configs/model/base_model.yaml \
    --dataset_config ./configs/dataset/base_data.yaml \
    --ckpt_path ./ckpt/example/epoch_029.pth.tar \
    --data_dir ./data/example \
    --save_dir ./data/example \
    --subset_path data/example/example_subset.json \
    --device cuda:0
```

## **Testing, Formatting and Linting**

The pipeline needs to be tested. I started with testing the tool functions I made, but ideally even the code we imported from other repos needs to be tested. To make the repo safe and clean, I used PyLint and Black.

To run them:
- pytest `python -m pytest tests/` at root
- pylist `pylint path/to/file_or_dir` at root, or with git saved files: eg. `pylint $(git ls-files 'src/processes/*.py')`
- black `black relative/path/to/file` anywhere
