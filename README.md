# Heart Disease Classification Using Machine Learning

## 🎯 Project Overview

This project implements a comprehensive heart disease classification system using machine learning algorithms on the UCI Heart Disease dataset. The work aligns with **UN Sustainable Development Goal 3: Good Health and Well-being**, specifically targeting cardiovascular disease prevention through early detection.

## 👥 Authors
- **Junaidh** 
- **Alfin**

**Course:** MSc Data Mining and Machine Learning  
**Date:** December 2024  
**Institution:** Griffith University

## 📊 Dataset Information

- **Source:** UCI Machine Learning Repository
- **Instances:** 303 patients
- **Features:** 13 clinical attributes
- **Target:** Binary classification (heart disease present/absent)

### Key Features:
- `age`: Age in years
- `sex`: Sex (1 = male, 0 = female)
- `cp`: Chest pain type (1-4)
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl
- `restecg`: Resting ECG results
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of peak exercise ST segment
- `ca`: Number of major vessels (0-3)
- `thal`: Thalassemia

## 🤖 Machine Learning Models

We implemented and compared four classification algorithms:

1. **Logistic Regression** - Baseline interpretable model
2. **Random Forest** - Ensemble method with feature importance
3. **Support Vector Machine (SVM)** - Non-linear classification with RBF kernel
4. **Decision Tree** - Fully interpretable model with visualization

## 📈 Key Results

### Model Performance
- **All models achieved >80% accuracy benchmark**
- **Random Forest**: Best overall performer (highest accuracy)
- **SVM**: Excellent discrimination ability (high ROC-AUC)
- **Decision Tree**: Most interpretable with clear decision rules

### Top Predictive Features
1. **Chest Pain Type (cp)** - Asymptomatic patients show highest risk
2. **Maximum Heart Rate (thalach)** - Lower values associated with disease
3. **ST Depression (oldpeak)** - Exercise-induced changes indicate cardiac stress
4. **Exercise Angina (exang)** - Strong predictor of disease presence
5. **Number of Major Vessels (ca)** - Coronary artery blockage indicator

## 🔬 Methodology

Following the **CRISP-DM** framework:

1. **Business Understanding** - Problem definition and SDG alignment
2. **Data Understanding** - Exploratory data analysis and statistical summary
3. **Data Preparation** - Missing value imputation, feature scaling, train-test split
4. **Modeling** - Implementation of 4 ML algorithms with hyperparameter tuning
5. **Evaluation** - Comprehensive performance assessment and comparison
6. **Deployment Considerations** - Clinical recommendations and future work

## 📊 Visualizations Generated

The project includes 10 comprehensive visualizations:

1. Target distribution analysis
2. Age distribution by disease status
3. Correlation heatmap
4. Feature distributions by class
5. Chest pain type analysis
6. Confusion matrices for all models
7. ROC curves comparison
8. Decision tree visualization
9. Feature importance analysis
10. Model performance comparison

## 🏥 Clinical Impact

### UN SDG 3 Alignment
- **Early Detection Capability**: Models can identify high-risk patients for preventive intervention
- **Accessible Screening**: Automated tools can be deployed in resource-limited healthcare settings
- **Cost-Effective Healthcare**: Reduces need for expensive initial diagnostic procedures
- **Evidence-Based Medicine**: Provides quantitative risk assessment for clinical decision-making

### Key Clinical Insights
- **Counterintuitive Finding**: Asymptomatic patients (chest pain type 4) have 83.3% disease rate
- **Exercise Stress Parameters**: Highly predictive indicators (thalach, oldpeak, exang)
- **Screening Importance**: Highlights need for comprehensive cardiac evaluation beyond symptoms

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo jupyter
```

### Running the Project
1. Clone this repository
2. Open `Heart_Disease_Classification.ipynb` in Jupyter Notebook
3. Run all cells sequentially
4. Generated visualizations will be saved as PNG files

### Repository Structure
```
├── Heart_Disease_Classification.ipynb    # Main implementation notebook
├── JUNAIDH_Complete_Implementation_Guide.md    # Detailed implementation guide
├── ALFIN_Complete_Implementation_Guide.md      # Comprehensive methodology guide
├── Assignment.pdf                        # Original project requirements
├── README.md                            # This file
└── Generated Visualizations/            # Output PNG files
```

## 📊 Technical Specifications

- **Programming Language**: Python 3.9+
- **Key Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **Development Environment**: Jupyter Notebook
- **Data Source**: UCI ML Repository (ucimlrepo package)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## 🔮 Future Work

### Technical Improvements
- Larger, diverse datasets from multiple centers
- Advanced feature engineering with domain knowledge
- Deep learning approaches for automatic feature discovery
- Uncertainty quantification for confidence intervals

### Clinical Validation
- Prospective clinical trials in real-world settings
- Multi-center validation across different populations
- Cost-effectiveness analysis vs traditional screening
- Integration with electronic health records

## 📚 References

- UCI Machine Learning Repository: Heart Disease Dataset
- World Health Organization: Cardiovascular Disease Statistics
- UN Sustainable Development Goals: Goal 3 - Good Health and Well-being

## 📄 License

This project is developed for academic purposes as part of the MSc Data Mining and Machine Learning course at Griffith University.

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the authors.

---

**Project Status**: ✅ Complete  
**Last Updated**: December 2024  
**Academic Institution**: Griffith University
