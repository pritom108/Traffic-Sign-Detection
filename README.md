# 🚦 Enhancing Traffic Sign Detection: YOLO + GAN Performance Boost

This project focuses on improving traffic sign detection using **YOLOv4/YOLOv5** combined with **GAN-generated images** and **EDSR upscaling**, boosting detection performance for real-world traffic scenarios.  

---

## 📑 Thesis Overview

The thesis was carried out in **7 main steps**:

1. **Dataset Preparation**  
   - Selected **5 traffic sign classes** for training and testing.  
   - Prepared base dataset for YOLOv4 and YOLOv5 training.

2. **Training and Testing YOLO Models**  
   - Trained **YOLOv4** and **YOLOv5** on the base dataset and base models.  
   - Evaluated base models on selected traffic sign classes.

3. **Data Augmentation using GANs**  
   - Generated new images from the base dataset using:
     - **LSGAN**  
     - **ProGAN**

4. **Upscaling Generated Images**  
   - Applied **EDSR GAN** to upscale LSGAN and ProGAN generated images.

5. **Training with Augmented Data**  
   - Combined datasets to train YOLO models with generated and upscaled images:
     - **YOLOv4 + LSGAN:** Main dataset + LSGAN generated + EDSR upscaled  
     - **YOLOv4 + ProGAN:** Main dataset + ProGAN generated + EDSR upscaled  
     - **YOLOv5 + LSGAN:** Main dataset + LSGAN generated + EDSR upscaled  
     - **YOLOv5 + ProGAN:** Main dataset + ProGAN generated + EDSR upscaled  

6. **Evaluation**  
   - Evaluated all trained models using standard **metrics** (precision, recall, mAP, etc.)  
   - Compared performance of YOLOv4 and YOLOv5 across different augmentation strategies.

7. **Inference App**  
   - Built a **Python Tkinter application** to detect traffic signs in real-life scenarios.  
   - Allows real-time testing of trained models.

---

## 💾 Training Resources

- **YOLOv5 + LSGAN / YOLOv5 + ProGAN Training**  
  [Download Here](https://drive.google.com/drive/folders/164LqX1mP0CHViDIn-Nlxx5E3aw7Km749?usp=sharing)

- **EDSR GAN for Upscaling Images**  
  [Download Here](https://drive.google.com/drive/folders/1EPjuBQsTT47K_3gySrSJlsPupjv5CoWB?usp=sharing)

---

## ⚙️ Model Inference Resources

To run inference, download the following folders and weights:

### MODELS Folder
[MODELS Main Folder](https://drive.google.com/drive/folders/1AeVq5ySUPlbsAsbvDnQ2TAjOXCzZDhEz?usp=sharing)

#### YOLOv4
- [YOLOv4 Folder](https://drive.google.com/drive/folders/11qbBuZS7rOGzvcMlfJi50woQp2QuVWeQ?usp=sharing)  
  _Download the `yolo-v4_custom_best.weights` file._

#### YOLOv4 + LSGAN
- [YOLOv4 + LSGAN Folder](https://drive.google.com/drive/folders/1PaC4GNE2AAN8JqLRLOuBnd5nCscFLw_X?usp=sharing)  
  _Download the `yolo-v4_custom_best.weights` file._

#### YOLOv4 + ProGAN
- [YOLOv4 + ProGAN Folder](https://drive.google.com/drive/folders/1TSL-WK1Z2_88l1haAtKmkbRBUZNQaAxw?usp=sharing)  
  _Download the `yolo-v4_custom_best.weights` file._

---

### Images & Test Data

- **[images Folder](https://drive.google.com/drive/folders/18HVkoes6DxOACFhTNnK8uiolDretbslB?usp=sharing)**  
- **[testImages Folder](https://drive.google.com/drive/folders/1AMEGuug8N0aCaon6zjURBbTfPMGr5MEM?usp=sharing)**

---



<h2>🗂 Folder Structure for Inference App</h2>

<pre>
project_root/
├── MODELS/
│   ├── YOLOv4/
│   │   └── yolo-v4_custom_best.weights
│   ├── YOLOv4+LSGAN/
│   │   └── yolo-v4_custom_best.weights
│   └── YOLOv4+ProGAN/
│       └── yolo-v4_custom_best.weights
├── images/
├── testImages/
└── inference_app.py   (Tkinter-based inference script)
</pre>
