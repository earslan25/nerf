# NeRF
Basic implementation of paper: Representing Scenes as Neural Radiance Fields for View Synthesis. 
Implemented as a final group project for Brown University's Deep Learning course. 
Team member github usernames: Zihan53, sylviebartusek, cintroca

## Installation

```
git clone https://github.com/earslan25/nerf.git
cd nerf
conda create --name nerf python=3.9
conda activate nerf
pip install -r requirements.txt # Change the last line in requirements.txt for gpu training
```

## How to run
Train
```
python train_nerf.py
```

Test
```
python train_nerf.py --test
```

Change config in configs/default.json

## Dataset

Refer to https://github.com/maximeraafat/BlenderNeRF
