# Automated C-Arm Positioning via Conformal Landmark Localization

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository provides the official implementation of our paper "Automated C-Arm Positioning via Conformal Landmark Localization" 🏆 Accepted at **[ICCV Workshop on Advanced Perception for Autonomous Healthcare (APAH), 2025](https://apahws.github.io/)**


## Model 🤖
<p align="center">
  <img src="assets/main_model.svg" width="85%">
</p>

## Interpretation 👨‍🏫
<p align="center">
  <img src="assets/conformal_score.svg" width="75%">
</p>

## Code Workflow 👨‍💻
1. **Train the Model**: Begin by training the model with your dataset using the script located at `src/train_landmark_regression.py`.
2. **Calculate Calibration Intervals**: After training, calculate the necessary calibration intervals using `src/calibration_interval_generator.py`.
3. **Test the Models**: Finally, test the models and obtain the test metrics with the script found at `src/test_landmark_regression.py`.

## Project Status 🚧
This repository is still under active development.
We are working on releasing a version adaptable to any X-ray dataset with defined landmark annotations.
For early collaboration or dataset access inquiries, please contact me at 📧 ahmad.arrabi@uvm.edu

