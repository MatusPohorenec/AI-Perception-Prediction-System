"""
Save trained models for use in the interactive frontend.
"""
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data_prep import (
    load_raw_dataframe,
    preprocess,
    build_regression_frame,
    build_classification_frame,
    INPUT_FEATURES_BASELINE,
    TARGET_CLASSIFICATION_3CLASS,
)
from src.models import (
    train_small_sample_regressors,
    train_small_sample_classifiers,
    train_advanced_classifiers,
    select_features_by_rfe,
)

DATA_PATH = Path(__file__).resolve().parent / "Survey in 2025.csv"
MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    print("Loading and preprocessing data...")
    df_raw = load_raw_dataframe(DATA_PATH)
    df = preprocess(df_raw)
    
    # =========================================================================
    # REGRESSION MODEL (Cost Reduction)
    # =========================================================================
    print("\nTraining regression model...")
    df_reg, X_base, y_base = build_regression_frame(df, use_optimized=False)
    
    # RFE feature selection
    selected_feats, _ = select_features_by_rfe(X_base, y_base, n_features_to_select=6)
    X_reg = X_base[selected_feats]
    
    # Train and get best model (Lasso)
    reg_results = train_small_sample_regressors(X_reg, y_base)
    
    # Get the best model (Lasso with R2=0.501)
    best_reg_name = 'lasso'
    best_reg = reg_results[best_reg_name]
    
    # Create scaler fitted on full data
    scaler_reg = StandardScaler()
    scaler_reg.fit(X_reg)
    
    # Retrain model on scaled data
    from sklearn.linear_model import Lasso
    final_reg_model = Lasso(alpha=0.1, max_iter=10000)
    X_reg_scaled = scaler_reg.transform(X_reg)
    final_reg_model.fit(X_reg_scaled, y_base)
    
    regression_package = {
        'model': final_reg_model,
        'scaler': scaler_reg,
        'features': selected_feats,
        'feature_names_display': {
            'Age_Numeric': 'Age Group',
            'Company_Size_Numeric': 'Company Size',
            'ICT_Utilization_Numeric': 'ICT Utilization Level',
            'AI_Util_Personal_Numeric': 'Personal AI Utilization',
            'AI_Training_Numeric': 'AI Training Level',
            'AI_Impact_Budgeting': 'AI Impact on Budgeting',
        },
        'target_name': 'Perception_CostReduction',
        'r2_score': reg_results[best_reg_name]['r2'],
    }
    
    with open(MODEL_DIR / 'regression_model.pkl', 'wb') as f:
        pickle.dump(regression_package, f)
    print(f"  Saved regression model (R2={regression_package['r2_score']:.3f})")
    
    # =========================================================================
    # CLASSIFICATION MODELS (6 Perception Targets)
    # =========================================================================
    print("\nTraining classification models...")
    
    classification_packages = {}
    
    target_display_names = {
        'Perception_Automation_3Class': 'AI for Automation',
        'Perception_Materials_3Class': 'AI for Materials Management',
        'Perception_ProjectMonitor_3Class': 'AI for Project Monitoring',
        'Perception_HR_3Class': 'AI for Human Resources',
        'Perception_Admin_3Class': 'AI for Administration',
        'Perception_Planning_3Class': 'AI for Planning',
    }
    
    # Use optimized features for classification
    from src.data_prep import INPUT_FEATURES_OPTIMIZED
    
    for target_col in TARGET_CLASSIFICATION_3CLASS:
        target_name = target_col.replace('Perception_', '').replace('_3Class', '')
        
        df_clf, X_clf, y_clf = build_classification_frame(df, target_col, use_optimized=True)
        
        if len(df_clf) < 10:
            continue
        
        # Train classifiers
        clf_results = train_small_sample_classifiers(X_clf, y_clf, target_name)
        
        # Try advanced classifiers
        try:
            clf_advanced = train_advanced_classifiers(X_clf, y_clf)
            clf_results.update(clf_advanced)
        except:
            pass
        
        # Find best model
        best_name = max(
            [k for k in clf_results.keys() if k != 'baseline_stratified'],
            key=lambda k: clf_results[k].get('f1_weighted', 0)
        )
        best_clf = clf_results[best_name]
        
        # Create scaler and retrain
        scaler_clf = StandardScaler()
        X_clf_scaled = scaler_clf.fit_transform(X_clf)
        
        # Retrain best model type
        if 'knn' in best_name:
            from sklearn.neighbors import KNeighborsClassifier
            k = best_clf.get('best_k', 5)
            final_clf = KNeighborsClassifier(n_neighbors=k, weights='distance')
        elif 'logistic' in best_name or 'logreg' in best_name:
            from sklearn.linear_model import LogisticRegression
            final_clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
        elif 'ridge' in best_name:
            from sklearn.linear_model import RidgeClassifier
            final_clf = RidgeClassifier(alpha=1.0, class_weight='balanced')
        elif 'ordinal' in best_name:
            try:
                from mord import LogisticAT
                final_clf = LogisticAT(alpha=1.0)
            except:
                from sklearn.linear_model import LogisticRegression
                final_clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
        else:
            from sklearn.neighbors import KNeighborsClassifier
            final_clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
        
        final_clf.fit(X_clf_scaled, y_clf)
        
        classification_packages[target_col] = {
            'model': final_clf,
            'scaler': scaler_clf,
            'features': list(X_clf.columns),
            'target_name': target_col,
            'display_name': target_display_names.get(target_col, target_name),
            'best_model_name': best_name,
            'f1_score': best_clf['f1_weighted'],
            'classes': sorted(y_clf.unique()),
            'class_labels': {1: 'Low', 2: 'Medium', 3: 'High'},
        }
        
        print(f"  {target_name}: {best_name} (F1={best_clf['f1_weighted']:.3f})")
    
    with open(MODEL_DIR / 'classification_models.pkl', 'wb') as f:
        pickle.dump(classification_packages, f)
    
    print(f"\nModels saved to {MODEL_DIR}")
    print("Run 'streamlit run app.py' to start the interactive frontend.")


if __name__ == "__main__":
    main()
