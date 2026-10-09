# Title 
A Federated Learning Framework for Kidney Stone Diagnosis with Explainable AI for Clinical Interpretability

## 1. Description
This project implements FKKSD, a privacy-preserving, explainable, and scalable deep learning framework for automated kidney stone detection from KUB (Kidney-Ureter-Bladder) X-ray images.
The framework integrates three main components:
1. Federated Learning (FL): enables privacy-preserving collaborative training across distributed hospital clients without sharing raw patient images.
2. Deep Learning: uses a ResNet-18** backbone optimized using nine confusion-matrix-based evaluation metrics.
3. Explainable AI (XAI): provides visual explanations using Grad-CAM, LIME, SHAP, Integrated Gradients, Occlusion Sensitivity, and Bounding Box.

The framework is designed to combine diagnostic performance, data privacy, reliability-aware collaborative learning, and model interpretability in a single pipeline for computer-aided kidney stone screening and referral support.

---

## 2. Dataset Information

The project uses KUB X-ray images using the dataset publicaly available  GitHub. The original dataset contains 500 anteroposterior KUB X-ray images, with 250 Kidney Stone images and 250 Normal images.
For this implementation, data augmentation was applied to extend the dataset to 14,000 images, equally distributed between:
- Kidney Stone: 7,000 images
- Normal: 7,000 images
An 80:20 split was applied:
- Training: 11,200 images
- Testing:** 2,800 images
The data were distributed across three simulated hospital clients while preserving the class distribution.

Permission to use the kidney X-ray dataset for academic research was obtained from its original author, Dr. Fahad Ahmed, via email on 8 October 2026. The author granted permission for academic use, provided that the original dataset source and associated research publication are appropriately acknowledged and cited in the manuscript.


### Dataset Source
The KUB X-ray dataset used in this project was obtained from the following GitHub repository: https://github.com/engrsmali219-coder/KUB-Kidney-Stone-or-Normal.
The dataset contains KUB X-ray images categorized into Kidney Stone and Normal classes and was used for training and evaluation of the proposed FKKSD framework. This study describes 500 AP KUB X-ray images, including 250 kidney-stone and 250 normal cases.
> **Note:** The 14,000-image augmented dataset and 80:20 split described above refer to the preprocessing/training setup used in this project.

---
## 3. Project Structure / Code Information

The main scripts used by the project are summarized below.

| File / Script | Purpose |
|---|---|
| `federated_server.py` | Starts the federated-learning server and performs global model aggregation across the three clients. |
| `local_training.py` | Performs local ResNet-18 training for an individual hospital/client. |
| `evaluate.py` | Evaluates the trained/global model and generates classification metrics, confusion matrix, ROC curve, and AUC. |
| `xai_explainer.py` | Generates explanations using Grad-CAM, LIME, SHAP, Integrated Gradients, Occlusion Sensitivity, and Bounding Box. |
| `FKKSD_Workflow.ipynb` | Provides the end-to-end workflow for dataset preparation, training, evaluation, and XAI analysis in Jupyter/Google Colab. |
| `requirements.txt` | Contains the Python dependencies required to run the project. |
| `models/` | Directory for storing trained/global model files. |
| `outputs/` | Directory for generated plots, evaluation results, and XAI visualizations. |
**Important:** If the repository uses different filenames or folder names, replace the names in this table with the exact names present in the repository.

---

## 4. Requirements

### Software Requirements
| Library | Recommended Version |
|---|---:|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.1 |
| torchvision | ≥ 0.16 |
| TensorFlow | ≥ 2.12 |
| NumPy | ≥ 1.25 |
| Pandas | Recommended |
| scikit-learn | ≥ 1.3 |
| Matplotlib | ≥ 3.8 |
| Seaborn | ≥ 0.12 |
| OpenCV | ≥ 4.8 |
| SHAP | Required for SHAP explanations |
| LIME | Required for LIME explanations |
| Captum | Required for Integrated Gradients and related attribution methods |

### Install Dependencies

```bash
open google.colab install Import Libraries (torchvision tensorflow numpy pandas scikit-learn matplotlib seaborn opencv-python shap lime captum)
```
# Set GPU Device and check Check if CUDA is available (Mounting the google drive)

## 5. Dataset Preparation

Organize the dataset for each simulated hospital/client as follows:

```text
/path/to/hospital_1/
├── kidney_stone/
└── normal/

/path/to/hospital_2/
├── kidney_stone/
└── normal/

/path/to/hospital_3/
├── kidney_stone/
└── normal/
```

Update the dataset path in the relevant training/configuration file:

```python
data_dir = "/content/drive/MyDrive/Thesis/1. Kidney"
```
---

## 6. Usage Instructions
### Step 1: Install the Requirements
Install the required Python packages:
```bash
pip install torch torchvision tensorflow numpy pandas scikit-learn matplotlib seaborn opencv-python shap lime captum
```
### Step 2: Prepare the Dataset

Place the Kidney Stone and Normal KUB X-ray images in the corresponding client directories described in the **Dataset Preparation** section.
### Step 3: Run Federated Training
Start federated training across the three simulated hospitals:
```bash
python federated_server.py --rounds 8 --batch_size 32 --clients 3
```
The framework performs **8 federated communication rounds** with a batch size of **32**.

### Step 4: Run Local Training
Local training can be performed for an individual hospital/client using:

```bash
python local_training.py --hospital 1 --epochs 8 --lr 0.0001
```
Change the hospital/client identifier as required.
Alternatively, the complete workflow can be executed using:
```text
FKKSD_Workflow.ipynb
```
in Jupyter Notebook or Google Colab.
### Step 5: Evaluate the Model
Evaluate the trained/global model:
```bash
python evaluate.py --model models/global_model.h5 --test_data /path/to/test
```
The evaluation generates classification metrics, confusion matrix, ROC curve, and AUC.
> **Model-format check:** The model filename and format used in the command above must match the actual model saved by the repository. If the implementation saves a PyTorch `.pth`/`.pt` model instead of a TensorFlow/Keras `.h5` model, update this command accordingly.
### Step 6: Generate XAI Explanations
Generate explanations for an input KUB X-ray:
```bash
python xai_explainer.py --image path/to/kub_xray.png --methods gradcam,lime,shap,ig,occlusion,bbox
```
The framework generates visual explanations using six XAI techniques.
---
## 7. Methodology

The complete FKKSD workflow follows the sequence:

```text
KUB X-ray Dataset
        ↓
Data Preprocessing
        ↓
Image Resizing (224×224)
        ↓
ImageNet Normalization
        ↓
Gaussian Filtering
        ↓
Training Data Augmentation
        ↓
Distribution to 3 Hospital Clients
        ↓
Local ResNet-18 Training
        ↓
Client-Specific Optimizers
        ↓
Reliability-Aware Aggregation
        ↓
Global Federated Model
        ↓
Testing & Performance Evaluation
        ↓
XAI Analysis
```
### 7.1 Preprocessing
Input images are resized to **224×224 pixels** and normalized using ImageNet mean and standard deviation. Gaussian filtering is used for noise reduction.
### 7.2 Data Augmentation
Augmentation is applied to the training data using:
- Flipping
- Rotation
- Zoom
- Blur
- Brightness/contrast adjustment
- HSV jitter
- Grayscale transformation
### 7.3 Federated Learning
Three simulated hospitals act as independent clients. Each client stores and processes its local data without transmitting raw patient images to the central server.
The ResNet-18 model is trained locally at each client using different optimizers:
- **Hospital 1:** AdamW
- **Hospital 2:** SGD with Momentum
- **Hospital 3:** RMSprop
The federated process uses **8 communication rounds** and a batch size of **32**.
### 7.4 Reliability-Aware Aggregation
Instead of relying only on standard FedAvg, the framework uses a custom reliability-scoring mechanism to weight client contributions according to validation performance and training stability.
### 7.5 Performance Evaluation
The framework evaluates:
- Accuracy
- Misclassification Rate
- Specificity
- Recall / Sensitivity
- Precision
- Negative Predictive Value (NPV)
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- F1-score
- ROC-AUC
Convergence is monitored using global loss variation, and McNemar's test is used for statistical analysis.
### 7.6 Explainable AI
Six complementary XAI techniques are integrated:
| XAI Method | Purpose |
|---|---|
| Grad-CAM | Highlights image regions contributing to the model prediction. |
| LIME | Provides local interpretable explanations for individual predictions. |
| Integrated Gradients | Produces pixel-level attribution scores. |
| Occlusion Sensitivity | Verifies important regions by masking parts of the image. |
| SHAP | Estimates feature contributions to the prediction. |
| Bounding Box | Provides explicit localization of the detected stone region. |
---
## 8. Performance Summary
The reported performance of the proposed FKKSD framework is:
| Metric | Value |
|---|---:|
| Accuracy | 98.57% |
| Misclassification Rate | 1.43% |
| Specificity | 99.21% |
| Recall / Sensitivity | 97.93% |
| Precision | 99.20% |
| False Positive Rate | 0.79% |
| False Negative Rate | 2.07% |
| AUC | 1.00 |
McNemar's test was also applied for comparative architecture analysis.
---
## 9. Outputs Generated
The project can generate the following outputs:
- Client-wise testing accuracy and loss plots
- Client-wise accuracy and loss across federated rounds
- Optimizer-specific performance curves
- Global model accuracy across federated rounds
- Confusion matrix
- ROC curve and AUC
- Grad-CAM heatmaps
- LIME explanations
- Integrated Gradients attribution maps
- Occlusion Sensitivity maps
- SHAP contribution maps
- Bounding Box localization
- McNemar's statistical test results

---

## 10. Evaluation Environment
The framework was developed and tested using:
- **Operating Systems:** Windows 10 / Linux
- **Platform:** Google Colab with GPU acceleration
- **Frameworks:** PyTorch and TensorFlow
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, OpenCV, SHAP, LIME, Captum
- **Federated Setup:** 3 simulated hospitals
- **Federated Rounds:** 8
- **Batch Size:** 32
---
## 11. Results and Discussion

**The ResNet-18 model within the proposed federated framework achieved **98.57% accuracy**, **99.21% specificity**, **97.93% recall**, and **99.20% precision**.
Federated Learning enables collaborative model training across three simulated hospitals without sharing raw patient images. Reliability-weighted aggregation prioritizes clients according to their validation performance and stability.
The use of AdamW, SGD with Momentum, and RMSprop across different clients provides a heterogeneous client-training configuration. The framework also integrates six XAI techniques to provide complementary visual explanations of model predictions.
The proposed framework is intended as a **preliminary screening and referral support tool**, rather than a replacement for clinical diagnosis.
**---

## 12. Limitations
The current implementation has the following limitations:
- The study is based on a single dataset and binary classification, which limits generalizability to diverse clinical environments and multi-class stone severity grading.
- Patient-level stratification was not possible because patient identifiers were unavailable.
- XAI outputs are qualitative because radiologist annotations were not available for quantitative validation.
- The individual contribution of preprocessing and augmentation techniques was not evaluated through ablation studies.
- Ensemble and advanced architectures were not explored; ResNet-18 was the primary backbone evaluated within the federated framework.
Future work may investigate multi-center datasets, multi-modal imaging such as CT and ultrasound, clinical metadata integration, advanced architectures such as Vision Transformers and EfficientNet, personalized federated learning, real-time edge/mobile deployment, and expert-validated explainability.
---
### Methods and Frameworks
Please cite the original publications associated with the methods used in this project, including:
- ResNet-18
- Federated Learning / Federated Averaging
- Grad-CAM
- LIME
- SHAP
- Integrated Gradients
- Occlusion Sensitivity
- Captum, where applicable
The complete bibliographic entries should be added according to the citation style used in the associated research paper.
---
## 13. Conclusion
The proposed **FKKSD (Federated Kidney Stone Detection)** framework integrates:
- ResNet-18-based kidney stone classification
- Privacy-preserving Federated Learning
- Reliability-weighted client aggregation
- Nine confusion-matrix-based evaluation metrics
- Convergence and statistical analysis
- Six complementary XAI techniques
The framework provides a unified pipeline emphasizing **accuracy, privacy, reliability, and interpretability** for automated kidney stone detection from KUB X-ray images.anatomical regions

This framework represents a practical, trustworthy AI pipeline for urological diagnostics, emphasizing accuracy, privacy, reliability, and transparency, with strong potential for clinical deployment as a screening and referral support tool in distributed healthcare systems.
