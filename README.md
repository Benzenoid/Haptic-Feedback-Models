# Haptic Feedback Modeling for Liver Tissue Simulation

This repository contains a Jupyter notebook implementing physical models (Kelvin-Voigt and Hertz-Crossley) and machine learning approaches to predict force feedback in liver tissue simulation.

## Data Set used

Kagglke Link : https://www.kaggle.com/datasets/benzenoid/haptic-feedback-liver

## Introduction

This project focuses on modeling haptic feedback for liver tissue simulation using both physical models and machine learning techniques. The work aims to accurately predict force feedback responses that can be used in medical training simulators and virtual surgery environments.

## Features

- Data transformation of haptic feedback measurements
- Implementation of physical models:
  - Kelvin-Voigt model
  - Hertz-Crossley model
- Machine learning models for force prediction:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
  - Neural Networks
- Comprehensive model evaluation and comparison
- Visualization of results and performance metrics

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow/Keras
- Matplotlib
- Jupyter Notebook

## Installation

To run this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/haptic-feedback-modeling.git
cd haptic-feedback-modeling
pip install -r requirements.txt
```

## Usage

Open the Jupyter notebook to explore the analysis and models:

```bash
jupyter notebook
```

The notebook contains all the code for data processing, model implementation, training, and evaluation.

## Project Structure

```
├── data/                # Data files for haptic measurements
├── notebooks/           # Jupyter notebooks with analysis
│   └── haptic_modeling.ipynb  # Main notebook with all models
├── models/              # Saved trained models
├── README.md            # This file
└── requirements.txt     # Required dependencies
```

## Results

The notebook compares different modeling approaches for haptic feedback prediction, evaluating them using metrics such as R², Mean Squared Error (MSE), and Mean Absolute Error (MAE). Visualizations show the performance of each model and their ability to capture the physical properties of liver tissue.

## Future Work

- Implementation of more complex physical models
- Exploration of deep learning architectures
- Real-time implementation for haptic devices
- Integration with virtual reality environments

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Mail ID - rishabhbhardwaj_me21b15_57@dtu.ac.in

