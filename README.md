# Robustness Analysis: TabPFN-2.5 vs. Gradient Boosting for Student Dropout Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow)
![TabPFN](https://img.shields.io/badge/Model-TabPFN%202.5-brightgreen)

## 📌 Project Overview
This repository contains a rigorous machine learning pipeline designed to predict student dropout and academic success using tabular data. The research explicitly benchmarks **TabPFN-2.5** (a state-of-the-art Tabular Foundation Model) against three robust Gradient Boosting models (**XGBoost, LightGBM, and CatBoost**). 

Instead of merely chasing the highest standard accuracy, this project focuses on **Model Robustness** in handling two major real-world challenges in educational data mining:
1. **Moderate Class Imbalance:** The target variable is heavily skewed, leaving students who are currently *Enrolled* as the minority class.
2. **Temporal Data Availability:** Evaluating how early the prediction can be made accurately (Enrollment data vs. 1st Semester vs. 2nd Semester data).

## 📊 Dataset & Data Transparency
* **Source:** [Predict Students' Dropout and Academic Success (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
* **Total Observations:** 4,424 students
* **Total Features:** Up to 36 features (Demographics, Socio-economic, and Academic performance)
* **Target Class Distribution (Original / I0):** 
  * `Graduate`: 2,209 observations (49.93%)
  * `Dropout`: 1,421 observations (32.12%)
  * `Enrolled`: 794 observations (17.95%) *(Minority Class)*
 <img width="2140" height="1434" alt="1_Distribusi_Target" src="https://github.com/user-attachments/assets/e2234bdc-8e5b-40e3-9067-d9cbce19adca" />


## 🔬 Methodology: Rigorous ML Experimentation
The pipeline strictly adheres to the **CRISP-DM** methodology and implements a strict **Anti-Data Leakage Protocol**:
* **Untouched Test Set:** The dataset is split into an 80% Development Set (3,539 rows) and a 20% Final Test Set (885 rows) *before* any preprocessing or balancing occurs.
* **The 36-Matrix Experiment:** The models are evaluated inside the 80% Development Set using **5-Fold Stratified Cross-Validation** across 36 combinations:
  * **3 Temporal Scenarios (Dimension A):** 
    * `T1`: Enrollment-only features (21 features - Early Detection).
    * `T2`: Enrollment + 1st Semester features (27 features - Early Academic).
    * `T3`: Full Information (33 features - Retrospective Benchmark).
      <img width="2082" height="1138" alt="2_Jumlah_Fitur_Waktu" src="https://github.com/user-attachments/assets/6b8a9405-c300-45b6-9e78-22e0547bd3b2" />

  * **3 Imbalance Strategies (Dimension B):** 
    * `I0`: Original distribution.
    * `I1`: Class Weighting (Cost-sensitive learning).
    * `I2`: Random Oversampling (Applied *only* to the training folds during CV).
      <img width="3547" height="1443" alt="3_Dampak_Oversampling_ROS" src="https://github.com/user-attachments/assets/fa80b75b-1a3e-4195-95bc-cebad78b5235" />

  * **4 Algorithms (Dimension C):** TabPFN-2.5, XGBoost, LightGBM, CatBoost.

## 📈 Evaluation Metrics
To prevent misleading conclusions caused by class imbalance, standard accuracy is sidelined during development. The primary metrics are:
* **Macro F1-Score** (Primary Metric)
* **Balanced Accuracy**
* **Precision-Recall AUC (PR-AUC)** (Crucial for identifying the minority class)
* **Statistical Significance:** Evaluated using the **Friedman Test** (p < 0.05).
  <img width="1579" height="1427" alt="4_Final_Confusion_Matrix" src="https://github.com/user-attachments/assets/38bb9eb6-8dd0-489b-a8ee-fdcf41a55184" />
  <img width="472" height="247" alt="accuracy" src="https://github.com/user-attachments/assets/073c9a44-5530-4b3e-8952-b4af0b9f1797" />
  <img width="2097" height="1658" alt="5_Final_PR_AUC_Curve" src="https://github.com/user-attachments/assets/a08dd450-a899-4036-8e98-58ca2a0493a5" />




## 🏆 Key Findings & Experimental Results
Based on the execution of the 36-Matrix Pipeline, the analysis yielded the following transparent insights:

1. **The Impact of Temporal Data (T1 vs. T2 vs. T3):**
   * **T1 (Enrollment-only):** All models performed poorly (Macro F1 ~0.53 - 0.55). Demographic and socio-economic data alone at enrollment are insufficient predictors.
   * **T2 (Early Academic):** Adding 1st-semester performance data caused a massive performance spike (Macro F1 ~0.65 - 0.69). 
   * **T3 (Full Info):** Adding 2nd-semester data slightly improved performance (Macro F1 ~0.69 - 0.72), proving that 1st-semester data is the absolute critical threshold for early intervention.
2. **Model Robustness & TabPFN's Superiority:**
   * **TabPFN-2.5** demonstrated absolute robustness. Its performance delta between original (I0) and imbalanced strategies (I1/I2) was `0.0000`. It dominated the T2 and T3 scenarios out-of-the-box without any hyperparameter tuning.
   * Tree-based models (XGBoost, LightGBM, CatBoost) were highly sensitive to class imbalance and required Class Weighting (I1) to stabilize.
3. **Final Deployment & Explainability:**
   * In the final untouched test, the tuned **CatBoost** achieved an overall Accuracy of **0.71** and a Macro F1 of **0.67**.
   * **SHAP (Feature Importance)** revealed that `Curricular units 1st sem (approved)`, `Curricular units 1st sem (grade)`, and `Tuition fees up to date` are the top three defining factors for student dropout.
   * The minority class (`Enrolled`) remains the hardest to predict, highlighting the gray area of students who are active but struggling.
     <img width="2351" height="2864" alt="6_SHAP_Feature_Importance" src="https://github.com/user-attachments/assets/007c12ff-69b5-427b-ac0d-6bbc111c8aed" />


## 🚫 Exclusions (Out of Scope)
To maintain academic rigor and focus on the core research questions, the following aspects were intentionally excluded from this project:
* **Software Deployment:** This project does not include the deployment of the machine learning model into a web application, REST API, or mobile app. The "deployment" phase refers to the final evaluation on the 20% unseen test set.
* **Deep Learning Models:** Neural networks (e.g., MLPs, PyTorch/TensorFlow architectures) are excluded, as the research specifically targets Tree-based Gradient Boosting vs. Tabular Foundation Models.
* **SMOTE on Test Data:** Synthetic Minority Over-sampling Technique (SMOTE) or Random Oversampling is strictly forbidden from being applied to the Validation/Test sets to preserve the original real-world distribution and prevent data leakage.

## 🚀 How to Run the Pipeline
The codebase is optimized for execution on **Google Colab (T4 GPU)** due to TabPFN-2.5's hardware requirements.

### Prerequisites (TabPFN API Token)
To use TabPFN-2.5, you must authenticate using an API token:
1. Register and accept the license at [Prior Labs](https://ux.priorlabs.ai).
2. Get your API Key from the Account tab.
3. Insert the key in the first cell of the notebook:
```python
import os
os.environ["TABPFN_TOKEN"] = "your_api_key_here"
