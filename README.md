# Generating COVID-19 Chest X-Ray Images Using DCGAN and WGAN-GP

This project implements and compares Deep Convolutional Generative Adversarial Networks (DCGAN) and Wasserstein GAN with Gradient Penalty (WGAN-GP) for generating synthetic COVID-19 chest X-ray images. The goal is to address class imbalance in medical image datasets and improve classification model performance.

## Features

- Implementation of DCGAN and WGAN-GP architectures
- Generation of synthetic COVID-19 chest X-ray images
- Hyperparameter tuning and comparison
- Evaluation using Fréchet Inception Distance (FID) score
- Integration with a DenseNet161 classifier for performance assessment

## Installation

1. Clone this repository
2. Install the required dependencies:
```
pip install torch torchvision PIL numpy matplotlib tqdm scipy mlxtend sklearn
```

## Usage

1. Download the COVID-19_Radiography_Dataset folder at https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
2. Run the notebook `gan_project.ipynb`

Note: The original code was developed and tested in Google Colab. For optimal performance and compatibility, it is recommended to run this code in the Google Colab environment.

[Google Colab notebook link](https://colab.research.google.com/drive/1oq08BeHJ3WphL0E5kZ12zK0b-28jIl51?usp=sharing 'Google Colab notebook link')

## Data Sources

The project uses a dataset of 3616 COVID-19 chest X-ray images (299x299 pixels). The images are preprocessed using Histogram Equalization (CLAHE) and resized to 128x128 or 64x64 pixels.

## Technologies and Libraries Used
Python, PyTorch, torchvision, PIL (Python Imaging Library), NumPy, Matplotlib, tqdm, SciPy, mlxtend, scikit-learn, Google Colab

## Visualizations

![image85](https://github.com/user-attachments/assets/505db638-286e-4f7a-84f9-7034b470076b)

![image87](https://github.com/user-attachments/assets/a865c24b-2354-456e-9a2b-6f777902967a)

![image86](https://github.com/user-attachments/assets/f7f67035-87dc-4b5e-8555-30fb66339936)

*Comparison of real and generated COVID-19 chest X-ray images*


![image88](https://github.com/user-attachments/assets/712278d4-9651-43ed-b2ec-edc068f0b821)

*Confusion matrix of classifiers using DCGAN generated images*


![image89](https://github.com/user-attachments/assets/aa0b104d-5eb1-4472-87c2-1c3dac7738a6)

 *Confusion matrix of classifiers using WGAN-GP generated images*

## License

This project is licensed under the MIT License.

## Project Report

The project report provides a comprehensive overview of the research, implementation, and results of this study. Key sections include:

1. Introduction: Background on class imbalance in medical image datasets and the potential of GANs.
2. Theory: Detailed explanations of GAN, DCGAN, and WGAN-GP architectures, including their loss functions and improvements.
3. Experimental Setup: Description of datasets, preprocessing steps, and model architectures.
4. Hyperparameter Testing: Analysis of various hyperparameters' effects on model performance.
5. Results: Comparison of DCGAN and WGAN-GP performance using FID scores and classifier metrics.
6. Conclusion: Summary of findings, highlighting WGAN-GP's superior performance for medical image generation.

The report also includes visualizations of generated images, confusion matrices, and tables comparing model performance across different configurations.
