🚗 AI-Driven Object Detection for Autonomous Cars

🌟 Overview

This project focuses on developing a state-of-the-art object detection system for autonomous vehicles using advanced computer vision techniques. The primary goal is to ensure accurate detection of objects such as pedestrians, vehicles, and road signs to improve road safety and enhance autonomous driving capabilities.

🎯 Objective

Develop a precise object detection system for autonomous cars.

Target MNCs by making the project innovative, scalable, and production-ready.

📂 Project Structure

data/: 📊 Datasets for training, validation, and testing.

scripts/: 📝 Python scripts for data preprocessing, augmentation, training, testing, and deployment.

models/: 🤖 Pre-trained and custom models.

notebooks/: 📓 Jupyter notebooks for experimentation and visualization.

outputs/: 🗃️ Outputs, logs, and evaluation results.

Dockerfile: 🐋 Docker configuration for containerization.

requirements.txt: 📜 List of required Python dependencies.

README.md: 📘 Project documentation.

⚙️ Installation and Setup

Step 1: 🔧 Setup

Clone the repository from GitHub:

git clone https://github.com/astromanu/AI-Object-Detection-For-Autonomous-Cars.git
cd AI-Object-Detection-For-Autonomous-Cars

Install necessary dependencies:

pip install -r requirements.txt

Step 2: 🗂️ Project Organization

The project follows a structured organization with the following folder and file structure:

data/: Place your datasets here. Ensure the folder has raw, processed, and augmented subdirectories.

scripts/: Contains Python scripts for each stage of the project.

models/: Holds the pre-trained models and trained versions of YOLO.

notebooks/: Interactive Jupyter notebooks for training, evaluation, and experimentation.

outputs/: Stores logs, results, and other output files.

🚀 Execution Steps

Step 3: 📊 Data Collection & Preparation

Collect datasets like COCO, KITTI, or Waymo.

Run the preprocessing script to prepare the data:

python scripts/data_preprocessing.py

Perform data augmentation:

python scripts/data_augmentation.py

Step 4: 🏋️‍♂️ Model Training

Train the YOLOv5 model using the provided notebook:

jupyter notebook notebooks/train_yolov5.ipynb

Fine-tune the model on specific classes by updating parameters in the training notebook.

Step 5: 📈 Model Evaluation

Evaluate model performance on the test dataset:

python scripts/evaluate_model.py

Plot evaluation metrics like precision-recall curves using Jupyter notebooks:

jupyter notebook notebooks/evaluation.ipynb

Step 6: ⚡ Model Optimization

Run quantization and pruning for real-time optimization:

python scripts/model_optimization.py

Convert the model using TensorRT for faster inference.

python scripts/tensorrt_inference.py

Step 7: 🌐 Deployment

Create a REST API for real-time inference with FastAPI:

uvicorn scripts.app:app --host 0.0.0.0 --port 8000

Use Docker to containerize the deployment for easy scalability:

docker build -t ai-object-detection .
docker run -p 8000:8000 ai-object-detection

Step 8: 🎥 Real-World Testing

Test the model in real-world scenarios by running the video testing script:

jupyter notebook notebooks/real_world_testing.ipynb

Step 9: 📊 Visualization & Documentation

Include precision-recall curves and loss graphs to visualize model performance.

Add animations and screenshots of the model's predictions in action.

📈 Example Precision-Recall Graph:

import matplotlib.pyplot as plt

# Plot precision-recall curve
def plot_precision_recall(precision, recall):
    plt.plot(recall, precision, marker='o', linestyle='-', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

precision = [0.9, 0.85, 0.8, 0.75]
recall = [0.1, 0.4, 0.6, 0.9]
plot_precision_recall(precision, recall)

🤝 Contributions

Feel free to contribute by creating issues or pull requests on the GitHub repository. We welcome suggestions and improvements.

📧 Contact Information

Author: Manish Dhatrak

Email: Eleven to one at the rate Gmail.com

🖥️ Deployment Demo for MNC Presentation

Set up a simulated environment using the CARLA simulator.

Showcase the real-time detection capabilities through detailed visual demonstrations.

By following these instructions, you will be able to execute this project seamlessly. The project is well-documented, and each part is designed to ensure accuracy and efficiency, making it suitable for a research internship application.

🔗 GitHub Repository

🚀 Good luck with your internship application! 🍀

