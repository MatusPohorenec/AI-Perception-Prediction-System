# AI Perception Prediction System

A machine learning system for predicting AI adoption perceptions in the construction industry, featuring an interactive Streamlit web application.

## 🎯 Project Overview

This project analyzes survey data from construction industry professionals to predict their perceptions of AI adoption. It addresses the hypothesis: *"For small companies, it's redundant to invest in AI solutions, and would hurt their finances."*

### Key Features

- **ML Pipeline**: Optimized for small-sample datasets (n=52) using LOOCV validation
- **Regression Models**: Lasso regression achieving R² = 0.501
- **Classification Models**: k-NN tuned classifiers with average F1 = 0.681
- **Interactive Dashboard**: Streamlit app with individual and company profile analysis

## 📁 Project Structure

```
├── app.py                 # Streamlit web application
├── train.py               # Main training pipeline
├── save_models.py         # Model persistence utility
├── requirements.txt       # Python dependencies
├── Survey in 2025.csv     # Survey dataset
├── src/
│   ├── __init__.py
│   ├── data_prep.py       # Data preprocessing & feature engineering
│   └── models.py          # ML model definitions & training
└── saved_models/
    ├── regression_model.pkl
    └── classification_models.pkl
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone git@github.com:MatusPohorenec/AI-Perception-Prediction-System.git
cd AI-Perception-Prediction-System

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

### Training Models

```bash
python train.py
```

## 📊 Features

### Individual Profile Assessment
- Job Position (7 levels)
- Work Experience (5 levels)
- Age Group
- Digital Competencies
- Personal AI Usage
- AI Training Level
- ICT Utilization

### Company Profile Assessment
- Company Size (Micro/Small/Medium/Large)
- Digitalization Level
- Company AI Usage
- Expected AI Impact (5 areas)

### Analysis & Predictions
- AI Perception Score predictions
- Hypothesis testing by company size
- ROI recommendations

## 🔬 Model Performance

| Model Type | Metric | Score |
|------------|--------|-------|
| Regression (Lasso) | R² | 0.501 |
| Classification (k-NN) | F1 | 0.681 |

## 📝 Dataset

Survey data from 52 construction industry professionals in Slovakia (2025), covering:
- Demographics & job characteristics
- AI usage and training levels
- Perceptions of AI impact on various business aspects

## 🛠️ Technologies

- **Python 3.14+**
- **scikit-learn**: Machine learning models
- **Streamlit**: Interactive web application
- **Plotly**: Data visualization
- **Pandas/NumPy**: Data processing
- **Optuna**: Hyperparameter optimization

## 📄 License

MIT License

## 👤 Author

Matus Pohorenec
