
<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-Best%20Model-orange?style=for-the-badge&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/R²%20Score-0.9865-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Complete-success?style=for-the-badge"/>

<br/><br/>

# 🦕 Fossil Age Prediction — Regression

### *Predicting the age of fossils using geological, chemical & physical features*

<br/>

> **Can machine learning unlock the secrets buried in stone?**  
> This project builds regression models to estimate fossil ages from radiometric and stratigraphic data — bridging paleontology and data science.

<br/>

---

</div>

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Feature Engineering](#-feature-engineering)
- [Models & Results](#-models--results)
- [Key Insights](#-key-insights)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)

---

## 🔭 Overview

This project tackles a fascinating **Advanced Regression** problem: estimating how old a fossil is based on measurable physical and chemical properties. The dataset blends real-world paleontological records from **PaleoBioDB** with synthetically generated samples to create a rich, realistic foundation for machine learning.

**Core Goals:**
- Build robust regression models to accurately predict fossil ages
- Identify the most informative geological and chemical features
- Apply end-to-end ML — from raw data to a deployable prediction pipeline

---

## 📊 Dataset

**Source:** [Fossil Age Prediction Regression — Kaggle](https://www.kaggle.com/)  
**Difficulty:** Advanced  
**Split:** ~4,398 training + 1,100 test samples

| Feature | Description |
|---|---|
| `uranium_lead_ratio` | Ratio of uranium to lead isotopes |
| `carbon_14_ratio` | Ratio of carbon-14 isotopes |
| `radioactive_decay_series` | Measurement of parent-to-daughter isotope decay |
| `stratigraphic_layer_depth` | Depth of fossil in the stratigraphic layer (meters) |
| `isotopic_composition` | Proportion of isotopes in the fossil |
| `fossil_size` | Size of the fossil (cm) |
| `fossil_weight` | Weight of the fossil (grams) |
| `geological_period` | Geological period of fossil formation |
| `surrounding_rock_type` | Rock type surrounding the fossil |
| `paleomagnetic_data` | Paleomagnetic orientation at the site |
| `stratigraphic_position` | Position in the stratigraphic column |
| `inclusion_of_other_fossils` | Boolean — presence of other fossils |
| **`age`** | 🎯 **Target** — Fossil age in years |

---

## 🔬 Project Workflow

```
Raw Data
   │
   ├── 1. Data Cleaning & Preprocessing
   │       ├── Missing value imputation (median / mode)
   │       ├── Boolean column encoding
   │       └── Duplicate removal
   │
   ├── 2. Outlier Detection & Removal
   │       └── IQR method across all numeric features
   │
   ├── 3. Exploratory Data Analysis (EDA)
   │       ├── Distribution plots (histograms, KDE)
   │       ├── Pairplots for feature relationships
   │       └── Correlation heatmaps (Pearson + Spearman)
   │
   ├── 4. Feature Engineering
   │       ├── Isotope activity ratio
   │       ├── Log-transformed decay series
   │       ├── Composite isotope index
   │       └── Decay-isotope interaction term
   │
   ├── 5. Ordinal Encoding (Categorical Features)
   │       ├── Geological period → chronological order
   │       ├── Stratigraphic position → Top/Middle/Bottom
   │       ├── Rock type → depositional energy proxy
   │       └── Paleomagnetic data → Normal/Reversed polarity
   │
   ├── 6. Model Training & Evaluation
   │       └── 5 Regressors compared
   │
   └── 7. Final Model — XGBoost (Top 10 Features)
           └── Deployment-ready prediction pipeline
```

---

## 🧪 Feature Engineering

Four domain-informed features were engineered to enhance model performance:

| New Feature | Formula | Rationale |
|---|---|---|
| `isotope_activity_ratio` | `uranium_lead_ratio / carbon_14_ratio` | Captures relative isotope decay rates |
| `log_decay_series` | `log1p(radioactive_decay_series)` | Reduces skewness in decay measurements |
| `composite_isotope_index` | `mean(U/Pb + C14 + decay_series)` | Aggregated isotopic signal |
| `decay_isotope_interaction` | `radioactive_decay_series × isotopic_composition` | Interaction effect between decay and composition |

---

## 📈 Models & Results

Five regression algorithms were benchmarked with StandardScaler preprocessing:

| Model | MAE | R² Score |
|---|---|---|
| Linear Regression | 1799.61 | 0.9761 |
| Decision Tree Regressor | 2783.32 | 0.9444 |
| Random Forest Regressor | 1726.15 | 0.9783 |
| K-Nearest Neighbors | 4903.93 | 0.8280 |
| **XGBoost Regressor** ✅ | **1369.75** | **0.9865** |

### 🏆 Winner: XGBoost with Top-10 Features

After feature importance analysis, training on the **top 10 features** yielded a significant drop in MAE — proving that thoughtful feature selection further boosts performance.

**Top 10 Most Important Features:**
```
1. uranium_lead_ratio
2. stratigraphic_position
3. stratigraphic_layer_depth
4. composite_isotope_index
5. geological_period
6. paleomagnetic_data
7. carbon_14_ratio
8. surrounding_rock_type
9. isotope_activity_ratio
10. radioactive_decay_series
```

---

## 💡 Key Insights

- **`uranium_lead_ratio`** is the single strongest predictor of fossil age — consistent with the known half-life precision of uranium-lead radiometric dating
- **`stratigraphic_layer_depth`** and **`carbon_14_ratio`** are the next most influential — deeper layers and lower C14 signal older specimens
- Engineered features (especially `composite_isotope_index`) contributed meaningfully to the model's predictive power
- XGBoost's gradient boosting framework handles the non-linear interactions between geological features far better than linear approaches
- The residuals from the final model are approximately normally distributed — confirming a well-fitted, unbiased model

---

## 🛠 Tech Stack

```
Language     Python 3.10+
ML Library   scikit-learn, XGBoost
Data         pandas, NumPy
Viz          matplotlib, seaborn
Environment  Google Colab / Jupyter Notebook
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/fossil-age-prediction.git
cd fossil-age-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# 3. Launch the notebook
jupyter notebook Fossil_Age_Prediction_Regression.ipynb
```

> 📂 Place `train_data.csv` and `test_data.csv` in the same directory before running.

---

## 🗂 Project Structure

```
fossil-age-prediction/
│
├── Fossil_Age_Prediction_Regression.ipynb   # Main notebook (EDA → Model → Deployment)
├── Fossil_Age_Prediction_Project_.pdf       # Project report
├── README.md                                # You are here
├── train_data.csv                           # Training data
└── test_data.csv                            # Test data
```

---

<div align="center">

**Built with curiosity 🦖 | ML meets Paleontology**

*If this project helped you, feel free to ⭐ the repository!*

</div>
