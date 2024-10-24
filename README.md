# 🚗 **AI-Driven Object Detection for Autonomous Cars**

![AI Object Detection Banner](https://via.placeholder.com/1200x400.png?text=AI+Driven+Object+Detection+for+Autonomous+Cars-)

## 🔍 Overview

This project focuses on developing a state-of-the-art object detection system for autonomous vehicles using advanced computer vision techniques. The primary goal is to ensure accurate detection of objects such as pedestrians, vehicles, and road signs to improve road safety and enhance autonomous driving capabilities.

## 🗂️ Project Structure

- **📂 data/**: Datasets for training, validation, and testing.
- **📜 scripts/**: Python scripts for data preprocessing, augmentation, training, testing, and deployment.
- **📁 models/**: Pre-trained and custom models.
- **📓 notebooks/**: Jupyter notebooks for experimentation and visualization.
- **📤 outputs/**: Outputs, logs, and evaluation results.
- **🐳 Dockerfile**: Docker configuration for containerization.
- **📄 requirements.txt**: List of required Python dependencies.
- **📖 README.md**: Project documentation.

---

## 🚀 Installation and Setup

### Step 1: Setup

1. **Clone the repository from GitHub**:

    ```bash
    git clone https://github.com/astromanu/AI-Driven-Object-Detection-for-Autonomous-Cars-.git
    cd AI-Object-Detection-For-Autonomous-Cars
    ```

2. **Install necessary dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Project Organization

The project follows a structured organization with the following folder and file structure:

- **📂 data/**: Place your datasets here. Ensure the folder has `raw`, `processed`, and `augmented` subdirectories.
- **📜 scripts/**: Contains Python scripts for each stage of the project.
- **📁 models/**: Holds the pre-trained models and trained versions of YOLO.
- **📓 notebooks/**: Interactive Jupyter notebooks for training, evaluation, and experimentation.
- **📤 outputs/**: Stores logs, results, and other output files.

---

## 🔧 Execution Steps

### Step 3: Data Collection & Preparation

1. **Collect datasets** like COCO, KITTI, or Waymo.

2. **Run the preprocessing script** to prepare the data:

    ```bash
    python scripts/data_preprocessing.py
    ```

3. **Perform data augmentation**:

    ```bash
    python scripts/data_augmentation.py
    ```

### Step 4: Model Training

1. **Train the YOLOv5 model** using the provided notebook:

    ```bash
    jupyter notebook notebooks/train_yolov5.ipynb
    ```

2. **Fine-tune the model** on specific classes by updating parameters in the training notebook.

### Step 5: Model Evaluation

1. **Evaluate model performance** on the test dataset:

    ```bash
    python scripts/evaluate_model.py
    ```

2. **Plot evaluation metrics** like precision-recall curves using Jupyter notebooks:

    ```bash
    jupyter notebook notebooks/evaluation.ipynb
    ```


### Step 6: Model Optimization

1. **Run quantization and pruning** for real-time optimization:

    ```bash
    python scripts/model_optimization.py
    ```

2. **Convert the model using TensorRT** for faster inference:

    ```bash
    python scripts/tensorrt_inference.py
    ```

### Step 7: Deployment

1. **Create a REST API** for real-time inference with FastAPI:

    ```bash
    uvicorn scripts.app:app --host 0.0.0.0 --port 8000
    ```

2. **Use Docker** to containerize the deployment for easy scalability:

    ```bash
    docker build -t ai-object-detection .
    docker run -p 8000:8000 ai-object-detection
    ```

### Step 8: Real-World Testing

1. **Test the model in real-world scenarios** by running the video testing script:

    ```bash
    jupyter notebook notebooks/real_world_testing.ipynb
    ```

### Step 9: Visualization & Documentation

- **Include precision-recall curves and loss graphs** to visualize model performance.
- **Add animations and screenshots** of the model's predictions in action.

---

## 📊 Evaluation Graphs and Metrics

- **Precision-Recall Curve**:

  ![Precision-Recall Curve](https://via.placeholder.com/800x400.png?text=Precision-Recall+Curve)

- **Loss Convergence Graph**:

  ![Loss Convergence](https://via.placeholder.com/800x400.png?text=Loss+Convergence)

- **mAP Results**:

  ![mAP Results](https://via.placeholder.com/800x400.png?text=mAP+Results)

---

## 🤝 Contributions

Feel free to contribute by creating issues or pull requests on the GitHub repository. We welcome suggestions and improvements.

---

## 📞 Contact Information

- **Author**: Manish Dhatrak
- **Email**: manishdhatrak1121Gmail.com

## 🏁 Deployment Demo for MNC Presentation

- **Set up a simulated environment** using the CARLA simulator.
- Showcase the **real-time detection capabilities** through detailed visual demonstrations.

![CARLA Simulation Demo](https://via.placeholder.com/1200x600.png?text=CARLA+Simulation+Demo)


[🌐 GitHub Repository](https://github.com/astromanu007/AI-Driven-Object-Detection-for-Autonomous-Cars-)



