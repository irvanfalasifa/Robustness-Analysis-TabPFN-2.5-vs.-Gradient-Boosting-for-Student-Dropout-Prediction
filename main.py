# Cell 2: Data Acquisition, Audit, and Target Distribution Transparency.

# Install library yang dibutuhkan
!pip install tabpfn catboost lightgbm xgboost imbalanced-learn scikit-learn pandas numpy matplotlib seaborn

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import Library Utama
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
sns.set_theme(style="whitegrid")