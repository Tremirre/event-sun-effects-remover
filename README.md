# DeLux: Lighting Artifact Removal from RGB Video Using Neuromorphic Data

This repository contains the official implementation of **DeLux**, a deep learning framework for removing lighting artifacts (lens flares, glare, flicker, overexposure) from RGB video sequences by leveraging neuromorphic event data.

Developed as part of the master's thesis:  
**"Lighting Artifact Removal from RGB Video Sequences Using Neuromorphic Data"**  
by Bartosz Stachowiak, 2025

## 📌 Key Features

- Unified detection and removal of lighting artifacts using RGB + event data
- Modular architecture supporting different fusion and loss configurations
- Support for both real-world ([E2VID](https://rpg.ifi.uzh.ch/E2VID.html), [DSEC](https://dsec.ifi.uzh.ch/dsec-datasets/download/)) and synthetic datasets ([CARLA](https://carla.org/), custom augmenters)
- Evaluation tools for detection, image quality, and artifact removal metrics
 
---

## 📁 Project Structure

- `data-prep/` # Scripts for preparing datasets
- `src/` # Core model, training, and evaluation code
- `*.ipynb` # Jupyter notebooks for experiments and analysis
- `*.sh` # Shell scripts for training and testing
- `requirements.txt` # Core dependencies for both data preparation and training
- `requirements.prod.txt` # Core dependencies for model training and inference
- `Dockerfile` # Docker configuration
- `setup.sh` # Setup utility for environment and configs
