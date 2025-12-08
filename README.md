# XAI605-Midterm-Exam
This repository contains the final assignment for XAI605 (Deep Learning).
Your goal is to train a robot manipulator to complete the following sequential task from demonstration data:

1. Open the drawer
2. Place the target object inside
3. Close the drawer

## Task
You are provided with a set of human demonstrations and a partially implemented training codebase.
Your task is to:

1. Explore the provided demonstrations
2. Understand the objective of this task
3. Fill in the missing implementation blocks from `policies/baseline/modeling.py`. (It's indicated as `TODO`)
4. Train a model that successfully completes the full task

At the end, you must submit your trained model. Compress the files in `ckpt`.

## What is provided?
This repository includes:
1. Demonstration dataset collected in advance
2. Visualization tools to inspect and replay demonstrations
3. Starter training code with missing sections marked as TODO
4. Some implementation blocks are intentionally removed as part of the assignment.

## Install Dependency
Tested environment ```python 3.11.9```

```
pip install -r requrements.txt
```

## Files

### 0.teleop.ipynb
Contains keyboard teleoperation demo.

Use WASD for the xy plane, RF for the z-axis, QE for tilt, and ARROWs for the rest of rthe otations.

SPACEBAR will change your gripper's state, and Z key will reset your environment with discarding the current episode data.

```
This is how the demonstration data was originally collected. 
However, because manual teleoperation is quite time-consuming, we provide a dataset that has already been recorded for you.

Feel free to experiment with the teleoperation environment.
```


### 1.Visualize.ipynb

It contains downloading dataset from huggingface and visualizing it.

First, download the dataset
```
python download_data.py
```


```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
root = './dataset/demo_data'
dataset = LeRobotDataset('Jeongeun/deep_learning_2025',root = root )
```
Running this code will download the dataset independatly.

### 2.transform.ipynb
Define the action and observation space for the environment. 
```python
action_type = 'delta_eef_pose'  # Options: 'joint','delta_joint, 'delta_eef_pose', 'eef_pose'
proprio_type = 'eef_pose' # Options: 'joint', 'eef_pose'
observation_type = 'object_pose' # options: 'image', 'object_pose'
image_aug_num = 2  # Number of augmented images to generate per original image
transformed_dataset_path = './dataset/transformed_data'
```

Based on this configuration, it will transform the actions into the action_type and create new dataset for training. 

- action_type: representation of the actions. Options: 'joint','delta_joint','eef_pose','delta_eef_pose'
- proprio_type: representations of propriocotative informations. Options: eef_pose, joint_pos
- observation_type: whether to use image of a object position informations. Options: 'image','objet_pose'
- image_aug_num: the number of augmented trajectories to make when you are using image features

### 3.train.ipynb
Train simple MLP models with dataset.

### Expectected Results
<img src="./media/baseline.gif" width="480" height="360" controls></img>

Success Rate should be around 50-65%.
If you can acheive higher, it's the better.
