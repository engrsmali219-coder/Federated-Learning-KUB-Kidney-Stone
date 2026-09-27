**TITLE**
A Federated Learning Framework for Kidney Stone Diagnosis with Explainable AI for Clinical Interpretability

**Description**
This project implements FKKSD, a privacy-preserving, explainable, and scalable deep learning framework for automated kidney stone detection from KUB (Kidney-Ureter-Bladder) X-ray images. The framework uniquely integrates three core components. The first component is Federated Learning for privacy-preserving collaborative training across distributed hospitals, where raw patient data never leaves local premises. The second component is a Deep Learning backbone based on ResNet-18, optimized with nine confusion-matrix-based evaluation metrics. The third component is Explainable AI using six complementary techniques, namely Grad-CAM, LIME, SHAP, Integrated Gradients, Occlusion Sensitivity, and Bounding Box, providing clinically interpretable visual explanations. This framework addresses the critical research gap where existing studies focus on accuracy, explainability, or privacy independently, as FKKSD unifies all three into a single clinically deployable pipeline for computer-aided kidney stone diagnosis.

**Dataset Information**
The model was trained on KUB X-ray images collected from Ahmed et al. The dataset originally contains 500 anteroposterior (AP) KUB X-rays, evenly split between Kidney Stone and Normal classes (250 Kidney Stone and 250 Normal). Several data augmentation approaches were applied to extend the dataset to 14,000 images, evenly split between Kidney Stone (7,000) and Normal (7,000). An 80:20 split was applied, with 11,200 images for training and 2,800 for testing, with preserved class distribution across all three simulated hospital clients.

**Code Overview**
The pipeline integrates:
Data Preprocessing & Augmentation
Resizing to 224×224 pixels, normalization using ImageNet mean and standard deviation, Gaussian filtering for noise reduction, and augmentation applied only to the training set (flip, rotation, zoom, blur, brightness/contrast, HSV jitter, grayscale).
Federated Learning Setup
Three simulated hospitals (clients) with locally stored non-IID datasets.
ResNet-18 backbone trained independently at each client.
Client-specific optimizers: AdamW at Hospital 1, SGD with Momentum at Hospital 2, and RMSprop at Hospital 3.
Eight federated rounds with batch size = 32.
Only model parameters are transmitted to the cloud server, not raw data.
Reliability-Aware Aggregation
Custom reliability scoring mechanism evaluates each client's contribution.
Replaces standard FedAvg with weighted aggregation that prioritizes trustworthy clients with higher validation accuracy and stability.
Performance Evaluation
Confusion matrix generation for the testing set.
Computation of accuracy, misclassification rate, specificity, recall, precision, negative predictive value, false positive rate, false negative rate, and F1-score.
ROC curve and AUC computation.
Robustness & Statistical Analysis
Convergence monitoring via global loss variation.
McNemar's test for statistical significance.
Explainable AI (XAI) Integration
Grad-CAM class activation heatmaps.
LIME local interpretable explanations (yellow contours).
Integrated Gradients pixel-wise attribution.
Occlusion Sensitivity region masking verification.
SHAP feature contribution scores.
Bounding Box explicit stone localization (green rectangle).

All analyses are implemented in PyTorch and TensorFlow, with plotting via Matplotlib and Seaborn.

**Usage Instructions**
1. Dataset Preparation
   Organize the dataset directory at each hospital as:
text
/path/to/hospital_k/
   ├── kidney_stone/
   └── normal/
Update the path in the code:
python
data_dir = "/path/to/hospital_k"
2. Run Federated Training
   Execute the federated server across three simulated hospitals:
python federated_server.py --rounds 8 --batch_size 32 --clients 3
3. Run Local Training (per client)
python local_training.py --hospital 1 --epochs 8 --lr 0.0001
Or open and run FKKSD_Workflow.ipynb in Jupyter/Colab.

4. Run Evaluation
python evaluate.py --model models/global_model.h5 --test_data /path/to/test

5. Run XAI Explanations
python xai_explainer.py --image path/to/kub_xray.png --methods gradcam,lime,shap,ig,occlusion,bbox

6. Outputs Generated
Client-wise testing accuracy and loss plots across federated rounds
Client-wise accuracy and loss across FL rounds
Optimizer-specific performance curves for AdamW, SGD, and RMSprop
Global model accuracy across FL rounds
Confusion matrix for the testing set
ROC curve with AUC
XAI heatmaps including Grad-CAM, LIME, Integrated Gradients, Occlusion Sensitivity, SHAP, and Bounding Box
McNemar's test statistical results

**Requirements**
| Library | Version (Recommended) |
|----------|-----------------------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.1 |
| torchvision | ≥ 0.16 |
| TensorFlow | ≥ 2.12 |
| numpy | ≥ 1.25 |
| scikit-learn | ≥ 1.3 |
| matplotlib | ≥ 3.8 |
| seaborn | ≥ 0.12 |
| OpenCV | ≥ 4.8 |

Install dependencies:
bash
pip install torch torchvision tensorflow numpy scikit-learn matplotlib seaborn opencv-python

**Methodology Summary**
Component |	Description
Base Model |	ResNet-18 (federated backbone)|
Training Strategy	| Federated Learning with reliability-weighted aggregation|
Preprocessing	| Resizing (224×224), ImageNet normalization, Gaussian filtering|
Augmentation	| Flip, rotation (0–30°), zoom, blur, brightness/contrast, HSV jitter, grayscale|
Dataset Split	| 80% training / 20% testing with preserved class distribution|
Clients	| 3 simulated hospitals (non-IID data)|
Optimizers |	AdamW (H1), SGD+Momentum (H2), RMSprop (H3)|
Federated Rounds |	8 rounds, batch size = 32|
Aggregation	| Reliability-weighted FedAvg|
Evaluation Metrics |	Accuracy, Misclassification Rate, Specificity, Recall, Precision, NPV, FPR, FNR, F1-Score|
Robustness Analysis |	Convergence monitoring, McNemar's test|
Interpretability	| 6 XAI techniques: Grad-CAM, LIME, SHAP, Integrated Gradients, Occlusion Sensitivity, Bounding Box|

**Performance Summary**
Metric |	Value|
Accuracy |	98.57%|
Misclassification Rate	| 1.43%|
Specificity	| 99.21%|
Recall (Sensitivity)	| 97.93%|
Precision |	99.20%|
False Positive Rate |	0.79%|
False Negative Rate |	2.07%|

McNemar's test applied for comparative architecture analysis.
AUC of 1.00 confirms perfect classification capability on evaluated data.

**Visualization Examples**

- Confusion Matrix – class-wise accuracy for the testing set
- Client-wise Testing Accuracy & Loss – convergence analysis across federated rounds
- Client-wise Accuracy & Loss Across FL Rounds – per-client convergence
- Optimizer Performance Curves – AdamW, SGD+Momentum, RMSprop comparison
- Global Model Accuracy Across FL Rounds – server-side convergence
- ROC Curve – AUC = 1.00 demonstrating perfect classification
- Grad-CAM Heatmaps – highlight stone regions (renal pelvis, ureter) influencing predictions
- LIME Explanations – influential regions via yellow contours
- Integrated Gradients – pixel-wise attribution maps
- Occlusion Sensitivity – masked region verification
- SHAP – positive/negative feature contribution maps
- Bounding Box – detected stone localization (green rectangle)



**Evaluation Environment**
Developed and tested on:
- OS: Windows 10 / Linux
- Platform: Google Colab (GPU acceleration)
- Frameworks: PyTorch and TensorFlow
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, OpenCV, SHAP, LIME, Captum
- Federated Setup: 3 simulated hospitals, 8 rounds, batch size = 32

**Results and Discussion**
- ResNet-18 within the federated learning framework achieved the best performance with 98.57% accuracy, 99.21% specificity, 97.93% recall, and 99.20% precision.
- Federated learning enabled privacy-preserving collaboration across three hospitals without sharing raw patient data.
- Reliability-weighted aggregation outperformed standard FedAvg by prioritizing trustworthy clients.
- All clients converged rapidly, reaching 96–98% accuracy by Round 3 and stabilizing at ~99% by Round 8.
- Optimizer diversity with AdamW, SGD with Momentum, and RMSprop demonstrated the framework's robustness to heterogeneous client configurations.
- The ROC curve with AUC of 1.00 confirms perfect classification capability on evaluated data.
- Six XAI techniques provided multi-perspective visual explanations, highlighting stone regions and clinically relevant anatomy.
- The framework is designed as a preliminary screening and referral support tool for clinical practice, where avoiding false negatives is especially important.

**Limitations**
- Current study is limited to a single dataset and binary classification, restricting generalizability to diverse clinical environments and multi-class stone severity grading.
- Patient-level stratification was not possible due to unavailable patient identifiers.
- XAI outputs remained qualitative only, as radiologist annotations were unavailable for quantitative validation.
- The impact of individual preprocessing steps was not evaluated through ablation studies.
- Ensemble and advanced architectures were not explored; only ResNet-18 was benchmarked within the FL framework.
- Future work should address these gaps through multi-center datasets, multi-modal imaging (CT, ultrasound), clinical metadata integration (blood glucose levels, medical history), advanced architectures (Vision Transformers, EfficientNet), personalized FL, real-time mobile/edge deployment, and expert-validated explainability analysis.

**Conclusion**
The proposed FKKSD (Federated Kidney Stone Detection) framework delivers:
- State-of-the-art accuracy (98.57%) with ResNet-18 for binary kidney stone classification
- Privacy-preserving federated learning across three simulated hospitals without centralized data sharing
- Reliability-weighted aggregation improving robustness over standard FedAvg
- Comprehensive benchmarking with nine confusion-matrix-based evaluation metrics
- Robustness validation through convergence monitoring and statistical analysis
- Six clinically interpretable XAI techniques (Grad-CAM, LIME, SHAP, Integrated Gradients, Occlusion Sensitivity, Bounding Box) highlighting disease-related anatomical regions

This framework represents a practical, trustworthy AI pipeline for urological diagnostics, emphasizing accuracy, privacy, reliability, and transparency, with strong potential for clinical deployment as a screening and referral support tool in distributed healthcare systems.