# Quick Arthropod Detector
Apply detection model to quickly detect any Arthropod on images

## Installation

##### Clone repository

```bash
git clone https://github.com/edgaremy/quick-detector.git
```
```bash
cd quick-detector
```

##### Option #1: Setup Python venv with Conda

- Make sure you first have Conda installed
- Create a new conda virtual env with the requirements:
```bash
# You can replace "detector" by any name you like
conda create --name detector --file requirements.txt
```
- You can now activate the venv whenever you want to use it, and deactivate it when you're done:
```bash
# Activate the new venv:
conda activate detector

# Deactivate
conda deactivate
```

##### Option #2: Use already existing Python venv

Install requirements with pip:
```bash
pip install -r requirements.txt
```
##### Adding custom model

The model weights used in this code are not available on this repo. We recommend adding them manually in the `/model` directory.

## Usage

TODO