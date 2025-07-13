# Tomato Leaf Health Classification Using CNNs and Clustering

This project explores and compares two machine learning techniques—K-means clustering and convolutional neural networks (CNNs)—to classify tomato leaves as **healthy** or **unhealthy**, using labeled image data from the [PlantVillage dataset](https://www.plantvillage.org/).

> **You are viewing the CNN portion of the work**. All CNN-related code, visualizations, and training details are included in this repository.

---

## Project Summary

In this study, we implemented two core approaches:

1. **Clustering-Based Classification**  
   Using K-means to cluster pixel colors and representing each image by the distribution of clustered pixel colors.

2. **Deep Learning-Based Classification (CNNs)**  
   Training a convolutional neural network to learn hierarchical features that distinguish healthy from diseased leaves.

The models were evaluated on accuracy, precision, recall, and, most importantly, their ability to minimize misclassification of *unhealthy* leaves. Our goal is to contribute to **early plant disease detection**, a critical need in precision agriculture and food security.

---

## 📁 Repository Structure

    ├── Image_Classification_Write_Up.pdf     # Complete project write-up (clustering + CNNs)
    ├── code/                                 # Scripts for image preprocessing and model training
    │   ├── segment.py
    │   └── train_model.py
    ├── notebooks/                            # Jupyter notebooks for demos and visualization
    │   ├── segment_example.ipynb
    │   └── gradcam_example.ipynb
    └── README.md                             # This file

---

## Model & Data Access

Due to GitHub size limits, the image datasets and the pre-trained CNN model are hosted externally:

🔗 **[Google Drive Folder](https://drive.google.com/drive/folders/1_2LBAiQUiLUfs9wU5Zb0JWFQcEyku59s?usp=sharing)**

- `Train/` and `Test/` image directories  
- `example_model.keras` (trained CNN model)

---

## 🔧 Getting Started

1. **Clone the repo**  
    
    git clone [https://github.com/clarkvd/tomato-leaf-cnn.git  ](https://github.com/clarkvd/Plant-Classification.git)
    cd tomato-leaf-cnn

2. **Install requirements**  
    
    pip install -r requirements.txt

3. **Download data & model**  
    Download the `Train/`, `Test/`, and `example_model.keras` files from the Google Drive link below and place them into the following structure:

        project_root/
        ├── data/
        │   ├── train/
        │   └── test/
        └── models/
            └── example_model.keras

    🔗 **[Google Drive Folder](https://drive.google.com/drive/folders/1_2LBAiQUiLUfs9wU5Zb0JWFQcEyku59s?usp=sharing)**

4. **Run segmentation preprocessing and training**  
    
    python code/segment.py
    python code/train_model.py

6. **Explore notebooks**  
    
    jupyter notebook notebooks/image_segmentation_demo.ipynb
    jupyter notebook notebooks/gradcam_example.ipynb

---

## 📄 Documentation

For full methodology, experiments, and metrics, see the project write-up:

📘 **[`Image_Classification_Write_Up.pdf`](./Image_Classification_Write_Up.pdf)**

---

## 📊 Key Results

- **CNN** significantly outperforms clustering, especially in reducing false negatives on unhealthy leaves.  
- **Grad-CAM** visualizations highlight the regions the CNN focuses on, improving interpretability and trust.

---

## 🤝 Acknowledgments

- **Data:** [PlantVillage](https://www.plantvillage.org/)  
- **CNN Work & Write-Up:** Ryan Clark
