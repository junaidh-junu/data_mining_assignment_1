# 🎯 JUNAIDH'S COMPLETE IMPLEMENTATION GUIDE
## Heart Disease Classification Project - Step-by-Step from Scratch

---

## 📋 TABLE OF CONTENTS

1. [Your Role & Responsibilities](#your-role--responsibilities)
2. [STEP 1: Environment Setup](#step-1-environment-setup)
3. [STEP 2: Create the Jupyter Notebook](#step-2-create-the-jupyter-notebook)
4. [STEP 3: Business Understanding Section](#step-3-business-understanding-section)
5. [STEP 4: Load and Explore Data](#step-4-load-and-explore-data)
6. [STEP 5: Exploratory Data Analysis (EDA)](#step-5-exploratory-data-analysis-eda)
7. [STEP 6: Data Preprocessing](#step-6-data-preprocessing)
8. [STEP 7: Logistic Regression Model](#step-7-logistic-regression-model)
9. [STEP 8: Random Forest Model](#step-8-random-forest-model)
10. [STEP 9: ROC Curve Visualization](#step-9-roc-curve-visualization)
11. [STEP 10: Report Writing - Your Sections](#step-10-report-writing---your-sections)
12. [STEP 11: Video Presentation - Your Slides](#step-11-video-presentation---your-slides)
13. [STEP 12: Learning Journal](#step-12-learning-journal)
14. [Troubleshooting Common Errors](#troubleshooting-common-errors)

---

## YOUR ROLE & RESPONSIBILITIES

### What You Are Responsible For:
| Task | Description | Estimated Time |
|------|-------------|----------------|
| Business Understanding | Write problem statement, SDG connection | 3 hours |
| Data Loading | Load dataset, initial exploration | 2 hours |
| EDA Visualizations | Create 5+ charts with interpretations | 5 hours |
| Logistic Regression | Build and evaluate the model | 3 hours |
| Random Forest | Build, tune, and evaluate the model | 4 hours |
| ROC Curves | Create comparative visualization | 2 hours |
| Report Sections | Introduction + Data sections (~1500 words) | 6 hours |
| Video Slides 1-5 | Opening + methodology slides | 3 hours |
| Learning Journal | Personal reflections (500 words) | 2 hours |

**Total: ~30 hours**

### What Alfin Is Doing:
- Literature Review (finding research papers)
- Data Quality Report
- Feature Engineering (encoding, scaling)
- SVM Model
- Decision Tree Model
- Confusion Matrices
- Feature Importance Analysis
- Report: Modeling + Conclusions sections
- Video Slides 6-10

---

## STEP 1: ENVIRONMENT SETUP

### 1.1 What You Need Installed

Before you start coding, you need Python and some libraries. Here's how to check and install everything.

#### Check if Python is Installed
Open your terminal (Command Prompt on Windows, Terminal on Mac) and type:
```bash
python --version
```
You should see something like `Python 3.9.x` or `Python 3.10.x`. If not, download Python from [python.org](https://python.org).

#### Install Required Libraries
Copy and paste this entire command into your terminal:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo jupyter
```

**What each library does:**
| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation - working with tables/spreadsheets |
| `numpy` | Mathematical operations on arrays |
| `matplotlib` | Creating charts and graphs |
| `seaborn` | Making prettier statistical visualizations |
| `scikit-learn` | Machine learning algorithms |
| `ucimlrepo` | Downloading the heart disease dataset |
| `jupyter` | Running interactive notebooks |

### 1.2 Starting Jupyter Notebook

After installing, start Jupyter by typing in your terminal:
```bash
jupyter notebook
```

This will open a browser window. Navigate to where you want to save your project and click **"New" → "Python 3"** to create a new notebook.

**Save your notebook as:** `heart_disease_classification.ipynb`

---

## STEP 2: CREATE THE JUPYTER NOTEBOOK

### 2.1 Understanding Jupyter Notebooks

A Jupyter Notebook has **cells**. Each cell can contain:
- **Code** (Python code that runs)
- **Markdown** (Text with formatting for explanations)

To run a cell: Press `Shift + Enter`
To add a new cell: Press `B` (when not editing a cell)
To change cell type to Markdown: Press `M`
To change cell type to Code: Press `Y`

### 2.2 Create Your First Cell - Title

**Cell Type: Markdown**

Copy this into your first cell:

```markdown
# Heart Disease Classification Using Machine Learning

## MSc Data Mining and Machine Learning - Project Assignment

**Authors:** Junaidh & Alfin  
**Date:** December 2024  
**Module:** Data Mining and Machine Learning

---

### Project Overview
This project implements machine learning classification algorithms to predict heart disease presence using the UCI Heart Disease dataset. The work aligns with **UN Sustainable Development Goal 3: Good Health and Well-being**, specifically targeting cardiovascular disease prevention through early detection.

### Dataset
- **Source:** UCI Machine Learning Repository
- **Instances:** 303 patients
- **Features:** 13 clinical attributes
- **Target:** Binary classification (heart disease present/absent)

### Algorithms Implemented
1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)
4. Decision Tree
```

Press `Shift + Enter` to run and see the formatted text.

### 2.3 Create Import Cell

**Cell Type: Code**

Add a new cell and paste:

```python
# ============================================
# LIBRARY IMPORTS
# ============================================
# Run this cell first - it loads all necessary libraries

# Data manipulation and analysis
import pandas as pd                    # For working with dataframes (tables)
import numpy as np                     # For numerical operations

# Visualization libraries
import matplotlib.pyplot as plt        # Basic plotting
import seaborn as sns                  # Statistical visualizations

# Dataset source
from ucimlrepo import fetch_ucirepo    # To download the heart disease data

# Machine learning tools
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score, 
    roc_curve,
    confusion_matrix, 
    classification_report
)

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
# This ensures you get the same results every time you run the code
RANDOM_STATE = 42

print("✅ All libraries imported successfully!")
print(f"📊 Pandas version: {pd.__version__}")
print(f"🔢 NumPy version: {np.__version__}")
```

**Expected Output:**
```
✅ All libraries imported successfully!
📊 Pandas version: 2.x.x
🔢 NumPy version: 1.x.x
```

---

## STEP 3: BUSINESS UNDERSTANDING SECTION

This is YOUR section to write. It explains WHY we're doing this project.

### 3.1 Add Business Understanding Cell

**Cell Type: Markdown**

```markdown
## 1. Business Understanding

### 1.1 Problem Statement

Cardiovascular diseases (CVDs) are the **leading cause of death globally**, accounting for approximately 17.9 million deaths annually according to the World Health Organization. Early detection of heart disease risk factors can significantly reduce mortality rates through timely intervention.

**The Problem:** Healthcare providers need efficient tools to identify patients at risk of heart disease based on clinical measurements. Manual assessment is time-consuming and may miss subtle patterns in patient data.

**Our Solution:** Develop a machine learning classification system that predicts heart disease presence using readily available clinical measurements.

### 1.2 Alignment with UN Sustainable Development Goal 3

This project directly supports **SDG 3: Good Health and Well-being**, specifically:

- **Target 3.4:** By 2030, reduce by one third premature mortality from non-communicable diseases through prevention and treatment
- **Target 3.8:** Achieve universal health coverage, including access to quality essential healthcare services

**How our project contributes:**
1. **Early Detection:** ML models can identify at-risk patients before symptoms appear
2. **Accessibility:** Automated screening can be deployed in resource-limited settings
3. **Cost Reduction:** Reduces need for expensive diagnostic procedures for initial screening

### 1.3 Success Criteria (SMART Goals)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **Specific** | Predict heart disease presence | Binary classification |
| **Measurable** | Achieve ≥80% accuracy | Accuracy score |
| **Achievable** | Based on published benchmarks (78-93%) | Literature review |
| **Relevant** | Supports SDG 3 healthcare goals | Impact assessment |
| **Time-bound** | Complete by December 11, 2024 | Project deadline |

### 1.4 Stakeholders

| Stakeholder | Interest | Benefit |
|-------------|----------|---------|
| Healthcare providers | Efficient screening tool | Faster patient triage |
| Patients | Early detection | Better health outcomes |
| Health systems | Cost reduction | Resource optimization |
| Researchers | Validated methodology | Reproducible results |
```

Press `Shift + Enter` to format this cell.

---

## STEP 4: LOAD AND EXPLORE DATA

This is where you load the actual data and look at it for the first time.

### 4.1 Load the Dataset

**Cell Type: Code**

```python
# ============================================
# DATA LOADING
# ============================================

# Download the Heart Disease dataset from UCI Repository
# The ID 45 refers to the Heart Disease dataset
print("📥 Downloading Heart Disease dataset from UCI Repository...")
heart_disease = fetch_ucirepo(id=45)

# Extract features (X) and target variable (y)
X = heart_disease.data.features
y = heart_disease.data.targets

print("✅ Dataset loaded successfully!")
print(f"\n📊 Dataset Shape: {X.shape[0]} patients, {X.shape[1]} features")
```

**Expected Output:**
```
📥 Downloading Heart Disease dataset from UCI Repository...
✅ Dataset loaded successfully!

📊 Dataset Shape: 303 patients, 13 features
```

### 4.2 Understand the Target Variable

**Cell Type: Code**

```python
# ============================================
# UNDERSTANDING THE TARGET VARIABLE
# ============================================

# Look at the original target values
print("Original target values:")
print(y['num'].value_counts())
print()

# The original data has values 0-4:
# 0 = no heart disease
# 1-4 = varying degrees of heart disease
# We need to convert this to BINARY: 0 (no disease) or 1 (disease present)

# Convert to binary classification
# If value is 0, keep it as 0 (no disease)
# If value is 1, 2, 3, or 4, convert to 1 (disease present)
y_binary = y['num'].apply(lambda x: 1 if x > 0 else 0)

print("After binary conversion:")
print(y_binary.value_counts())
print()
print(f"Class distribution:")
print(f"  No Disease (0): {(y_binary == 0).sum()} patients ({(y_binary == 0).mean()*100:.1f}%)")
print(f"  Disease (1): {(y_binary == 1).sum()} patients ({(y_binary == 1).mean()*100:.1f}%)")
```

**Expected Output:**
```
Original target values:
0    164
1     55
2     36
3     35
4     13
Name: num, dtype: int64

After binary conversion:
0    164
1    139
Name: num, dtype: int64

Class distribution:
  No Disease (0): 164 patients (54.1%)
  Disease (1): 139 patients (45.9%)
```

**What this tells us:** The dataset is fairly balanced - about 54% have no heart disease and 46% have heart disease. This is good because it means our models won't be biased toward one class.

### 4.3 Explore the Features

**Cell Type: Code**

```python
# ============================================
# EXPLORING THE FEATURES
# ============================================

# Display the first 5 rows of data
print("📋 First 5 rows of the dataset:")
print(X.head())
print()

# Display basic information about each column
print("\n📊 Dataset Information:")
print(X.info())
```

### 4.4 Feature Descriptions

**Cell Type: Markdown**

```markdown
## 2. Data Understanding

### 2.1 Dataset Overview

The UCI Heart Disease dataset contains medical records from 303 patients collected in Cleveland, Ohio in 1988. Each patient has 13 clinical attributes measured during examination.

### 2.2 Feature Descriptions

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| `age` | Age in years | Numeric | 29-77 |
| `sex` | Sex (1 = male, 0 = female) | Binary | 0, 1 |
| `cp` | Chest pain type | Categorical | 1-4 |
| `trestbps` | Resting blood pressure (mm Hg) | Numeric | 94-200 |
| `chol` | Serum cholesterol (mg/dl) | Numeric | 126-564 |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary | 0, 1 |
| `restecg` | Resting ECG results | Categorical | 0-2 |
| `thalach` | Maximum heart rate achieved | Numeric | 71-202 |
| `exang` | Exercise induced angina | Binary | 0, 1 |
| `oldpeak` | ST depression induced by exercise | Numeric | 0-6.2 |
| `slope` | Slope of peak exercise ST segment | Categorical | 1-3 |
| `ca` | Number of major vessels (0-3) | Numeric | 0-3 |
| `thal` | Thalassemia | Categorical | 3, 6, 7 |

### 2.3 Chest Pain Types (cp)
- **1:** Typical angina - chest pain related to heart
- **2:** Atypical angina - chest pain not related to heart
- **3:** Non-anginal pain - typically esophageal pain
- **4:** Asymptomatic - no symptoms

### 2.4 Target Variable
- **0:** No heart disease detected
- **1:** Heart disease present (any level)
```

### 4.5 Statistical Summary

**Cell Type: Code**

```python
# ============================================
# STATISTICAL SUMMARY
# ============================================

# Get descriptive statistics for all columns
print("📊 Statistical Summary of All Features:")
print("=" * 60)

# Using describe() gives us count, mean, std, min, 25%, 50%, 75%, max
summary = X.describe()
print(summary.round(2))

print("\n" + "=" * 60)
print("🔍 Key Observations:")
print("=" * 60)
print(f"• Average age: {X['age'].mean():.1f} years")
print(f"• Age range: {X['age'].min():.0f} to {X['age'].max():.0f} years")
print(f"• Average cholesterol: {X['chol'].mean():.1f} mg/dl")
print(f"• Average max heart rate: {X['thalach'].mean():.1f} bpm")
print(f"• Male patients: {(X['sex'] == 1).sum()} ({(X['sex'] == 1).mean()*100:.1f}%)")
print(f"• Female patients: {(X['sex'] == 0).sum()} ({(X['sex'] == 0).mean()*100:.1f}%)")
```

### 4.6 Check for Missing Values

**Cell Type: Code**

```python
# ============================================
# MISSING VALUE ANALYSIS
# ============================================

print("🔍 Missing Value Analysis:")
print("=" * 60)

# Count missing values in each column
missing_counts = X.isnull().sum()
missing_percent = (X.isnull().sum() / len(X)) * 100

# Create a summary dataframe
missing_df = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing %': missing_percent.round(2)
})

# Only show columns with missing values
missing_df = missing_df[missing_df['Missing Count'] > 0]

if len(missing_df) > 0:
    print("Columns with missing values:")
    print(missing_df)
else:
    print("✅ No missing values in features!")

# Also check target variable
print(f"\n📌 Target variable missing values: {y_binary.isnull().sum()}")

# Total missing
total_missing = X.isnull().sum().sum()
total_cells = X.shape[0] * X.shape[1]
print(f"\n📊 Total: {total_missing} missing values out of {total_cells} cells ({total_missing/total_cells*100:.2f}%)")
```

**Expected Output:**
```
🔍 Missing Value Analysis:
============================================================
Columns with missing values:
       Missing Count  Missing %
ca                 4       1.32
thal               2       0.66

📌 Target variable missing values: 0

📊 Total: 6 missing values out of 3939 cells (0.15%)
```

---

## STEP 5: EXPLORATORY DATA ANALYSIS (EDA)

This is YOUR main visualization section. You'll create charts to understand the data.

### 5.1 Target Distribution Visualization

**Cell Type: Code**

```python
# ============================================
# VISUALIZATION 1: TARGET DISTRIBUTION
# ============================================

# Set the style for all plots
plt.style.use('seaborn-v0_8-whitegrid')  # Clean, professional look

# Create figure with specific size
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Count plot
colors = ['#2ecc71', '#e74c3c']  # Green for healthy, Red for disease
bars = axes[0].bar(['No Disease (0)', 'Disease (1)'], 
                   [sum(y_binary == 0), sum(y_binary == 1)],
                   color=colors, edgecolor='black', linewidth=1.5)

# Add count labels on bars
for bar in bars:
    height = bar.get_height()
    axes[0].annotate(f'{int(height)}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=14, fontweight='bold')

axes[0].set_xlabel('Heart Disease Status', fontsize=12)
axes[0].set_ylabel('Number of Patients', fontsize=12)
axes[0].set_title('Distribution of Heart Disease in Dataset', fontsize=14, fontweight='bold')

# Right plot: Pie chart
sizes = [sum(y_binary == 0), sum(y_binary == 1)]
labels = [f'No Disease\n{sizes[0]} patients\n({sizes[0]/len(y_binary)*100:.1f}%)',
          f'Disease Present\n{sizes[1]} patients\n({sizes[1]/len(y_binary)*100:.1f}%)']
explode = (0.02, 0.02)  # Slightly separate both slices

axes[1].pie(sizes, explode=explode, colors=colors, 
            startangle=90, shadow=True,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
axes[1].legend(labels, loc='lower right', fontsize=10)
axes[1].set_title('Heart Disease Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Figure saved as 'target_distribution.png'")
```

### 5.2 Age Distribution Analysis

**Cell Type: Code**

```python
# ============================================
# VISUALIZATION 2: AGE DISTRIBUTION
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Age distribution by disease status
colors = {'No Disease': '#2ecc71', 'Disease': '#e74c3c'}

# Create a combined dataframe for easier plotting
plot_df = X.copy()
plot_df['Heart Disease'] = y_binary.map({0: 'No Disease', 1: 'Disease'})

# Histogram with KDE (Kernel Density Estimation)
for status in ['No Disease', 'Disease']:
    data = plot_df[plot_df['Heart Disease'] == status]['age']
    axes[0].hist(data, bins=15, alpha=0.6, label=status, 
                 color=colors[status], edgecolor='black')

axes[0].set_xlabel('Age (years)', fontsize=12)
axes[0].set_ylabel('Number of Patients', fontsize=12)
axes[0].set_title('Age Distribution by Heart Disease Status', fontsize=14, fontweight='bold')
axes[0].legend()

# Right: Box plot of age by disease status
box_data = [plot_df[plot_df['Heart Disease'] == 'No Disease']['age'],
            plot_df[plot_df['Heart Disease'] == 'Disease']['age']]

bp = axes[1].boxplot(box_data, labels=['No Disease', 'Disease'], 
                      patch_artist=True, widths=0.6)

# Color the boxes
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
for box in bp['boxes']:
    box.set_edgecolor('black')
    box.set_linewidth(1.5)

axes[1].set_xlabel('Heart Disease Status', fontsize=12)
axes[1].set_ylabel('Age (years)', fontsize=12)
axes[1].set_title('Age Comparison: Disease vs No Disease', fontsize=14, fontweight='bold')

# Add mean markers
for i, data in enumerate(box_data, 1):
    axes[1].scatter(i, data.mean(), color='yellow', s=100, zorder=5, 
                    marker='D', edgecolors='black', label='Mean' if i == 1 else '')

plt.tight_layout()
plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Print statistics
print("📊 Age Statistics by Heart Disease Status:")
print("-" * 50)
print(f"No Disease - Mean: {plot_df[plot_df['Heart Disease'] == 'No Disease']['age'].mean():.1f} years")
print(f"Disease    - Mean: {plot_df[plot_df['Heart Disease'] == 'Disease']['age'].mean():.1f} years")
```

### 5.3 Correlation Heatmap

**Cell Type: Code**

```python
# ============================================
# VISUALIZATION 3: CORRELATION HEATMAP
# ============================================

# Create a copy of features and add target
corr_df = X.copy()
corr_df['target'] = y_binary

# Calculate correlation matrix
correlation_matrix = corr_df.corr()

# Create the heatmap
fig, ax = plt.subplots(figsize=(14, 12))

# Create heatmap with seaborn
heatmap = sns.heatmap(correlation_matrix, 
                       annot=True,           # Show correlation values
                       fmt='.2f',            # Format to 2 decimal places
                       cmap='RdBu_r',        # Red-Blue diverging colormap
                       center=0,             # Center colormap at 0
                       square=True,          # Make cells square
                       linewidths=0.5,       # Add lines between cells
                       cbar_kws={'shrink': 0.8},
                       annot_kws={'size': 9})

plt.title('Correlation Matrix: All Features and Target Variable', 
          fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Print top correlations with target
print("\n📊 Top Correlations with Heart Disease (Target):")
print("=" * 50)
target_corr = correlation_matrix['target'].drop('target').sort_values(key=abs, ascending=False)
for feature, corr in target_corr.head(5).items():
    direction = "↑ positive" if corr > 0 else "↓ negative"
    print(f"  {feature}: {corr:.3f} ({direction})")
```

### 5.4 Feature Distribution by Class

**Cell Type: Code**

```python
# ============================================
# VISUALIZATION 4: KEY FEATURES BY DISEASE STATUS
# ============================================

# Select the most important numeric features
important_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feature in enumerate(important_features):
    # Create violin plot
    data_no_disease = X[y_binary == 0][feature].dropna()
    data_disease = X[y_binary == 1][feature].dropna()
    
    parts = axes[i].violinplot([data_no_disease, data_disease], 
                                positions=[1, 2], showmeans=True, showmedians=True)
    
    # Color the violins
    parts['bodies'][0].set_facecolor('#2ecc71')
    parts['bodies'][0].set_alpha(0.7)
    parts['bodies'][1].set_facecolor('#e74c3c')
    parts['bodies'][1].set_alpha(0.7)
    
    axes[i].set_xticks([1, 2])
    axes[i].set_xticklabels(['No Disease', 'Disease'])
    axes[i].set_title(f'{feature.upper()}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Value')

# Remove empty subplot
axes[5].axis('off')

# Add legend in empty space
axes[5].text(0.5, 0.7, '🟢 No Disease', fontsize=14, ha='center', 
             color='#2ecc71', fontweight='bold', transform=axes[5].transAxes)
axes[5].text(0.5, 0.5, '🔴 Disease Present', fontsize=14, ha='center', 
             color='#e74c3c', fontweight='bold', transform=axes[5].transAxes)

plt.suptitle('Feature Distributions by Heart Disease Status', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 5.5 Chest Pain Type Analysis

**Cell Type: Code**

```python
# ============================================
# VISUALIZATION 5: CHEST PAIN TYPE ANALYSIS
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Chest pain type labels
cp_labels = {1: 'Typical\nAngina', 2: 'Atypical\nAngina', 
             3: 'Non-anginal\nPain', 4: 'Asymptomatic'}

# Left: Count by chest pain type
cp_counts = X['cp'].value_counts().sort_index()
bars = axes[0].bar([cp_labels[i] for i in cp_counts.index], 
                   cp_counts.values, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Chest Pain Type', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distribution of Chest Pain Types', fontsize=14, fontweight='bold')

# Add value labels
for bar in bars:
    height = bar.get_height()
    axes[0].annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center')

# Right: Heart disease rate by chest pain type
cp_disease_rate = []
cp_types = sorted(X['cp'].dropna().unique())
for cp_type in cp_types:
    mask = X['cp'] == cp_type
    rate = y_binary[mask].mean() * 100
    cp_disease_rate.append(rate)

bars2 = axes[1].bar([cp_labels[i] for i in cp_types], cp_disease_rate, 
                    color=['#2ecc71' if r < 50 else '#e74c3c' for r in cp_disease_rate],
                    edgecolor='black')
axes[1].axhline(y=50, color='gray', linestyle='--', label='50% threshold')
axes[1].set_xlabel('Chest Pain Type', fontsize=12)
axes[1].set_ylabel('Heart Disease Rate (%)', fontsize=12)
axes[1].set_title('Heart Disease Rate by Chest Pain Type', fontsize=14, fontweight='bold')
axes[1].legend()

# Add value labels
for bar, rate in zip(bars2, cp_disease_rate):
    axes[1].annotate(f'{rate:.1f}%', xy=(bar.get_x() + bar.get_width()/2, rate),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('chest_pain_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 5.6 EDA Summary Cell

**Cell Type: Markdown**

```markdown
### 2.5 EDA Key Findings Summary

Based on our exploratory data analysis, we identified several important patterns:

#### Target Variable Balance
- The dataset is reasonably balanced with 54.1% no disease and 45.9% disease present
- This balance is favorable for classification algorithms

#### Age Patterns
- Patients with heart disease tend to be older on average
- Age shows a moderate positive correlation with disease presence

#### Key Predictive Features
1. **thalach (max heart rate):** Strong negative correlation - lower max HR associated with disease
2. **oldpeak (ST depression):** Strong positive correlation - higher values indicate disease
3. **cp (chest pain type):** Asymptomatic patients show highest disease rates
4. **exang (exercise angina):** Presence strongly associated with disease

#### Correlation Insights
- Maximum heart rate (thalach) has the strongest negative correlation with disease
- ST depression (oldpeak) has a notable positive correlation
- Age, sex, and exercise angina also show meaningful correlations

These findings will guide our feature selection and model interpretation in the modeling phase.
```

---

## STEP 6: DATA PREPROCESSING

### 6.1 Handle Missing Values

**Cell Type: Code**

```python
# ============================================
# DATA PREPROCESSING
# ============================================

print("=" * 60)
print("STEP 1: HANDLING MISSING VALUES")
print("=" * 60)

# Check missing values before
print("\nMissing values BEFORE imputation:")
print(X.isnull().sum()[X.isnull().sum() > 0])

# Create imputer - fills missing values with the most frequent value in each column
# This is appropriate for categorical-like features (ca, thal)
imputer = SimpleImputer(strategy='most_frequent')

# Apply imputation
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),  # fit_transform learns patterns and applies them
    columns=X.columns           # Keep original column names
)

# Verify no missing values remain
print("\nMissing values AFTER imputation:")
print(f"Total missing: {X_imputed.isnull().sum().sum()}")
print("✅ All missing values handled!")
```

### 6.2 Split Data

**Cell Type: Code**

```python
# ============================================
# TRAIN-TEST SPLIT
# ============================================

print("\n" + "=" * 60)
print("STEP 2: SPLITTING DATA INTO TRAINING AND TESTING SETS")
print("=" * 60)

# Split data: 80% for training, 20% for testing
# stratify=y_binary ensures same proportion of disease/no-disease in both sets
# random_state=42 ensures reproducibility (same split every time)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed,           # Features
    y_binary,            # Target
    test_size=0.2,       # 20% for testing
    random_state=RANDOM_STATE,  # For reproducibility (we set this to 42 earlier)
    stratify=y_binary    # Keep class proportions
)

print(f"\n📊 Training set size: {len(X_train)} samples ({len(X_train)/len(X_imputed)*100:.0f}%)")
print(f"📊 Testing set size: {len(X_test)} samples ({len(X_test)/len(X_imputed)*100:.0f}%)")

# Verify stratification worked
print(f"\n🎯 Class distribution in training set:")
print(f"   No Disease: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
print(f"   Disease: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")

print(f"\n🎯 Class distribution in test set:")
print(f"   No Disease: {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
print(f"   Disease: {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")
```

### 6.3 Feature Scaling

**Cell Type: Code**

```python
# ============================================
# FEATURE SCALING
# ============================================

print("\n" + "=" * 60)
print("STEP 3: FEATURE SCALING (STANDARDIZATION)")
print("=" * 60)

# StandardScaler transforms features to have mean=0 and std=1
# This is important because some algorithms (like SVM, Logistic Regression) 
# are sensitive to feature scales

scaler = StandardScaler()

# IMPORTANT: fit on training data only, then transform both
# This prevents "data leakage" - test data influencing the model
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),  # Learn scaling from training data
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),       # Apply same scaling to test data
    columns=X_test.columns
)

# Show example of scaling effect
print("\nExample: 'age' feature before and after scaling")
print(f"   Before - Mean: {X_train['age'].mean():.2f}, Std: {X_train['age'].std():.2f}")
print(f"   After  - Mean: {X_train_scaled['age'].mean():.2f}, Std: {X_train_scaled['age'].std():.2f}")

print("\n✅ Feature scaling complete!")
print("\n📋 Final preprocessed datasets ready for modeling:")
print(f"   X_train_scaled: {X_train_scaled.shape}")
print(f"   X_test_scaled: {X_test_scaled.shape}")
```

---

## STEP 7: LOGISTIC REGRESSION MODEL

This is YOUR model to build. Logistic Regression is a simple but effective classification algorithm.

### 7.1 Explanation Cell

**Cell Type: Markdown**

```markdown
## 4. Modeling

### 4.1 Algorithm Selection Rationale

We selected four diverse classification algorithms to compare their performance:

| Algorithm | Strengths | Why Selected |
|-----------|-----------|--------------|
| **Logistic Regression** | Interpretable, fast, probabilistic outputs | Baseline model, feature importance |
| **Random Forest** | Handles non-linear relationships, robust | Expected best performer |
| **SVM** | Effective in high dimensions, kernel flexibility | Different decision boundary approach |
| **Decision Tree** | Fully interpretable, visual rules | Explainability for stakeholders |

### 4.2 Model 1: Logistic Regression

**What is Logistic Regression?**

Despite its name, Logistic Regression is a **classification** algorithm (not regression). It predicts the **probability** that an instance belongs to a particular class.

**How it works:**
1. Calculates a weighted sum of input features
2. Applies the sigmoid function to convert this to a probability (0-1)
3. If probability > 0.5, predicts class 1 (disease); otherwise class 0

**Why it's good for this problem:**
- Produces probability scores (useful for risk assessment)
- Coefficients show feature importance
- Fast to train and predict
- Works well when relationships are roughly linear
```

### 7.2 Build Logistic Regression Model

**Cell Type: Code**

```python
# ============================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================

print("=" * 60)
print("BUILDING LOGISTIC REGRESSION MODEL")
print("=" * 60)

# Create the Logistic Regression model
# max_iter=1000: maximum iterations for the solver to converge
# random_state=42: for reproducibility
lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

# Train the model on training data
print("\n🔄 Training model...")
lr_model.fit(X_train_scaled, y_train)
print("✅ Model trained!")

# Make predictions on test data
y_pred_lr = lr_model.predict(X_test_scaled)

# Get probability predictions (for ROC curve later)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1

# Calculate metrics
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_prob_lr)

# Print results
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)
print(f"\n📊 Performance Metrics:")
print(f"   Accuracy:  {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")
print(f"   Precision: {precision_lr:.4f}")
print(f"   Recall:    {recall_lr:.4f}")
print(f"   F1-Score:  {f1_lr:.4f}")
print(f"   ROC-AUC:   {roc_auc_lr:.4f}")

# Detailed classification report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['No Disease', 'Disease']))
```

### 7.3 Interpret Logistic Regression Coefficients

**Cell Type: Code**

```python
# ============================================
# LOGISTIC REGRESSION COEFFICIENT ANALYSIS
# ============================================

print("=" * 60)
print("FEATURE IMPORTANCE (LOGISTIC REGRESSION COEFFICIENTS)")
print("=" * 60)

# Get feature names and coefficients
feature_names = X_train_scaled.columns
coefficients = lr_model.coef_[0]

# Create dataframe of coefficients
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("\n📊 Feature Coefficients (sorted by importance):")
print("-" * 50)
for _, row in coef_df.iterrows():
    direction = "↑ increases risk" if row['Coefficient'] > 0 else "↓ decreases risk"
    print(f"   {row['Feature']:12}: {row['Coefficient']:+.4f}  ({direction})")

# Visualize coefficients
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in coef_df['Coefficient']]
bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Coefficient Value', fontsize=12)
ax.set_title('Logistic Regression Coefficients\n(Positive = Increases Disease Risk)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('lr_coefficients.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## STEP 8: RANDOM FOREST MODEL

This is YOUR second model. Random Forest is an ensemble method that usually performs well.

### 8.1 Explanation Cell

**Cell Type: Markdown**

```markdown
### 4.3 Model 2: Random Forest

**What is Random Forest?**

Random Forest is an **ensemble learning method** that builds multiple decision trees and combines their predictions.

**How it works:**
1. Creates many decision trees (typically 100-500)
2. Each tree is trained on a random subset of data (bootstrap sampling)
3. Each tree considers only a random subset of features at each split
4. Final prediction is the **majority vote** of all trees

**Why it's good for this problem:**
- Handles non-linear relationships automatically
- Robust to outliers and noise
- Provides feature importance rankings
- Less prone to overfitting than single decision trees
- Often achieves high accuracy without much tuning

**Key Parameters:**
- `n_estimators`: Number of trees in the forest
- `max_depth`: Maximum depth of each tree
- `min_samples_split`: Minimum samples required to split a node
```

### 8.2 Build Random Forest Model

**Cell Type: Code**

```python
# ============================================
# MODEL 2: RANDOM FOREST
# ============================================

print("=" * 60)
print("BUILDING RANDOM FOREST MODEL")
print("=" * 60)

# Create Random Forest model with initial parameters
# n_estimators=100: number of trees
# max_depth=10: maximum depth of each tree (prevents overfitting)
# random_state=42: for reproducibility
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=RANDOM_STATE
)

# Train the model
print("\n🔄 Training model (this may take a few seconds)...")
rf_model.fit(X_train_scaled, y_train)
print("✅ Model trained!")

# Make predictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

# Print results
print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)
print(f"\n📊 Performance Metrics:")
print(f"   Accuracy:  {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")
print(f"   Precision: {precision_rf:.4f}")
print(f"   Recall:    {recall_rf:.4f}")
print(f"   F1-Score:  {f1_rf:.4f}")
print(f"   ROC-AUC:   {roc_auc_rf:.4f}")

# Detailed classification report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['No Disease', 'Disease']))
```

### 8.3 Hyperparameter Tuning for Random Forest

**Cell Type: Code**

```python
# ============================================
# RANDOM FOREST HYPERPARAMETER TUNING
# ============================================

print("=" * 60)
print("HYPERPARAMETER TUNING WITH GRID SEARCH")
print("=" * 60)
print("\n🔍 Finding optimal parameters (this may take 1-2 minutes)...")

# Define parameter grid to search
param_grid = {
    'n_estimators': [100, 200],      # Number of trees
    'max_depth': [5, 10, 15],        # Maximum depth
    'min_samples_split': [2, 5],     # Minimum samples to split
    'min_samples_leaf': [1, 2]       # Minimum samples in leaf
}

# Create GridSearchCV object
# cv=5: 5-fold cross-validation
# scoring='accuracy': optimize for accuracy
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1  # Use all available cores
)

# Fit grid search
grid_search.fit(X_train_scaled, y_train)

# Get best parameters
print("\n✅ Grid Search Complete!")
print(f"\n🏆 Best Parameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")
print(f"\n📊 Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Use the best model
rf_best = grid_search.best_estimator_

# Evaluate on test set
y_pred_rf_tuned = rf_best.predict(X_test_scaled)
y_prob_rf_tuned = rf_best.predict_proba(X_test_scaled)[:, 1]

accuracy_rf_tuned = accuracy_score(y_test, y_pred_rf_tuned)
roc_auc_rf_tuned = roc_auc_score(y_test, y_prob_rf_tuned)

print(f"\n📊 Tuned Model Test Performance:")
print(f"   Accuracy: {accuracy_rf_tuned:.4f} ({accuracy_rf_tuned*100:.2f}%)")
print(f"   ROC-AUC:  {roc_auc_rf_tuned:.4f}")
```

---

## STEP 9: ROC CURVE VISUALIZATION

This is YOUR visualization task - comparing all models' ROC curves.

**Cell Type: Code**

```python
# ============================================
# ROC CURVE COMPARISON (ALL 4 MODELS)
# ============================================

# Note: You'll need to run Alfin's SVM and Decision Tree code first
# For now, we'll create ROC curves for your two models

print("=" * 60)
print("ROC CURVE VISUALIZATION")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 8))

# Calculate ROC curves
# Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
ax.plot(fpr_lr, tpr_lr, 'b-', linewidth=2, 
        label=f'Logistic Regression (AUC = {roc_auc_lr:.3f})')

# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
ax.plot(fpr_rf, tpr_rf, 'g-', linewidth=2,
        label=f'Random Forest (AUC = {roc_auc_rf:.3f})')

# Plot diagonal line (random classifier)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

# Formatting
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.set_title('ROC Curves Comparison\n(Higher curve = Better model)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

# Set axis limits
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n📊 ROC-AUC Scores Summary:")
print(f"   Logistic Regression: {roc_auc_lr:.4f}")
print(f"   Random Forest: {roc_auc_rf:.4f}")
```

---

## STEP 10: REPORT WRITING - YOUR SECTIONS

You are responsible for writing approximately **1500 words** covering:

### Your Sections:
1. **Introduction** (~300 words)
2. **Business Understanding** (~400 words)
3. **Data Understanding** (~800 words)

### 10.1 Introduction Template

```
## 1. Introduction

### 1.1 Background and Motivation
[Write 2-3 paragraphs about:]
- Global burden of cardiovascular disease (use WHO statistics)
- Importance of early detection
- Role of machine learning in healthcare

### 1.2 Project Objectives
The primary objectives of this project are:
1. [First objective]
2. [Second objective]
3. [Third objective]

### 1.3 UN Sustainable Development Goal Alignment
This project aligns with SDG 3...
[Explain the connection in 1 paragraph]

### 1.4 Report Structure
This report is organized as follows:
- Section 2: Business Understanding
- Section 3: Data Understanding
- Section 4: Data Preparation
- Section 5: Modeling
- Section 6: Evaluation
- Section 7: Conclusions
```

### 10.2 Business Understanding Template

```
## 2. Business Understanding

### 2.1 Problem Definition
[Write 2 paragraphs explaining the problem in business terms]

### 2.2 Stakeholder Analysis
[Create a table and explanation of who benefits]

### 2.3 Success Metrics
[Define SMART criteria]

### 2.4 Ethical Considerations
[Discuss privacy, fairness, and responsible AI use]
```

### 10.3 Data Understanding Template

```
## 3. Data Understanding

### 3.1 Data Source
The UCI Heart Disease dataset was obtained from...
[Describe source, history, credibility]

### 3.2 Dataset Overview
The dataset contains [X] instances with [Y] features...
[Include a table of features]

### 3.3 Exploratory Data Analysis
[Describe your EDA findings with references to figures]

### 3.4 Data Quality Assessment
[Discuss missing values, outliers, class balance]
```

---

## STEP 11: VIDEO PRESENTATION - YOUR SLIDES

You are responsible for **Slides 1-5** (approximately 5 minutes of the 10-minute video).

### Slide 1: Title Slide (30 seconds)
- Project title
- Your names
- Module name
- Date

### Slide 2: Problem & SDG (1 minute)
- What problem are we solving?
- Why does it matter?
- Connection to SDG 3

### Slide 3: Dataset Overview (1 minute)
- Source and size
- Key features
- Target variable explanation

### Slide 4: EDA Highlights (1.5 minutes)
- Show 2-3 key visualizations
- Explain what they tell us
- Main insights

### Slide 5: Methodology Overview (1 minute)
- CRISP-DM framework
- Preprocessing steps
- Why these algorithms?

---

## STEP 12: LEARNING JOURNAL

Write **500 words** reflecting on your experience. Include:

### Structure:
```
# Individual Learning Journal

## My Contribution
[List what you specifically did]

## What I Learned
[Technical skills, soft skills, challenges overcome]

## Challenges Faced
[Problems you encountered and how you solved them]

## Team Collaboration
[How you worked with Alfin]

## Contribution Split
- Junaidh: [X]%
- Alfin: [Y]%

## Future Improvements
[What would you do differently?]
```

---

## TROUBLESHOOTING COMMON ERRORS

### Error: "ModuleNotFoundError: No module named 'ucimlrepo'"
**Solution:**
```bash
pip install ucimlrepo
```

### Error: "fetch_ucirepo() returned empty data"
**Solution:** Check your internet connection. The function downloads data from the internet.

### Error: "ValueError: could not convert string to float"
**Solution:** Make sure you're using `X_imputed` (after handling missing values) not the original `X`.

### Error: "ConvergenceWarning" for Logistic Regression
**Solution:** Increase `max_iter`:
```python
lr_model = LogisticRegression(max_iter=2000, random_state=42)
```

### Plots not showing
**Solution:** Add this after your import statements:
```python
%matplotlib inline
```

---

## ✅ CHECKLIST BEFORE SUBMISSION

- [ ] All code cells run without errors
- [ ] All visualizations are saved as .png files
- [ ] Logistic Regression model is trained and evaluated
- [ ] Random Forest model is trained and evaluated
- [ ] ROC curves are created and saved
- [ ] Report sections (Introduction, Business Understanding, Data Understanding) are written
- [ ] Video slides 1-5 are created
- [ ] Learning journal is written (500 words)
- [ ] All files are named correctly
- [ ] Notebook is organized with clear headings

---

**Good luck, Junaidh! You've got this! 🎉**

If you get stuck on any step, refer back to this guide or ask for help.
