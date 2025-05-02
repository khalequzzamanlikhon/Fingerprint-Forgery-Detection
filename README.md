# Fingerprint Analysis with YOLOv11 and Minutiae Extraction

This project performs fingerprint analysis using the [SOCOFing dataset](https://www.kaggle.com/datasets/ruizgara/socofing). It combines **minutiae extraction** for detailed fingerprint feature analysis with **YOLOv11-based classification** to distinguish between real and fake fingerprints. The project is implemented in Python and leverages libraries like Ultralytics YOLO, OpenCV, and scikit-image.

![Example Output](https://via.placeholder.com/600x300.png?text=Minutiae+Extraction+and+YOLO+Prediction)  
*Example output showing minutiae extraction (left) and YOLOv11 classification (right).*

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview
The project aims to analyze fingerprints by:
1. **Extracting Minutiae**: Identifies termination and bifurcation points in fingerprint images using image processing techniques.
2. **Classifying Real vs. Fake**: Uses a YOLOv11 nano classification model (`yolo11n-cls`) to detect and classify fingerprints as real or fake.
3. **Visualizing Results**: Displays minutiae points (terminations in blue, bifurcations in red) and YOLO predictions with confidence scores.
4. **Monitoring Training**: Logs training metrics to TensorBoard for visualization of loss, accuracy, and other metrics.

The pipeline is optimized for Kaggle’s T4x2 GPU environment but can be run locally with appropriate hardware.

## Features
- **Minutiae Extraction**: Detects fingerprint features (terminations and bifurcations) using skeletonization and morphological operations.
- **YOLOv11 Classification**: Classifies fingerprints as real or fake with a pre-trained YOLOv11 model fine-tuned on the SOCOFing dataset.
- **Preprocessing**: Enhances fingerprint images with Gaussian blur and Otsu’s thresholding.
- **Evaluation**: Computes classification metrics (precision, recall, F1 score) and visualizes confusion matrices.
- **TensorBoard Integration**: Logs training metrics for real-time monitoring.
- **Modular Code**: Organized into reusable Python modules for data preparation, preprocessing, minutiae extraction, training, and visualization.

## Dataset
The project uses the [SOCOFing dataset](https://www.kaggle.com/datasets/ruizgara/socofing), which contains:
- **Real Fingerprints**: BMP images of genuine fingerprints.
- **Altered Fingerprints**: Modified fingerprints (fake) for classification.

The dataset should be placed in a directory (e.g., `data/socofing/`) with `Real/` and `Altered/` subdirectories.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/fingerprint-analysis.git
   cd fingerprint-analysis

2. **Set Up a Virtual Environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
4. **Install Dependencies**:
   ``` bash
      pip install -r requirements.txt
6. **Prepare the Dataset**:
Download the SOCOFing dataset from Kaggle.
Place it in data/socofing/ or update DATA_DIR in main.py to point to your dataset location.

8. ** Run the full pipeline**: In the terminal run
   ``` bash
   python main.py .
you can monitor the training progress using 
    ``` bash
    
    tensorboard --logdir logs/runs

## Results

## Contributing
Contributions are welcome! To contribute:

- Fork the repository.
- Create a feature branch (git checkout -b feature/your-feature).
- Commit changes (git commit -m "Add your feature").
- Push to the branch (git push origin feature/your-feature).
- Open a pull request.

## License
This project is licensed under the MIT License. 

## Acknowledgements
- SOCOFing Dataset for providing fingerprint images.
- Ultralytics YOLO for the YOLOv11 implementation.
- Kaggle for providing the T4x2 GPU environment for training.
- Open-source libraries: OpenCV, scikit-image, Matplotlib, Seaborn, and TensorBoard.
