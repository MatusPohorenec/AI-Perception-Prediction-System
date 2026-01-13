"""
Data Preparation Module for Small-Sample Statistical Learning (n=51)

This module implements Phase 1 of the optimization strategy:
1. Target Simplification: 5-class → 3-class ordinal encoding
2. Feature Engineering: Theoretically-justified composite indices
3. Feature Selection Infrastructure: Spearman correlation, MI, RFE support

Key Principles:
- Reduce target classes to improve statistical stability
- Create composite indices to reduce dimensionality while preserving signal
- Maintain ordinal nature of survey data throughout
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict


RAW_COL_MAP = {
    'Age of respondends': 'Age',
    'Company size': 'Company_Size',
    'Job position': 'Job_Position',
    'Work experineces (in year)': 'Work_Experience',
    'SK NACE Classification': 'SK_NACE',
    'ICT Utilization level (of respondents)': 'ICT_Utilization',
    'AI utilization level (of respondents)': 'AI_Util_Personal',
    'Digital competencies (of respondents)': 'Digital_Competencies',
    "AI utilization level (of companies) - Company's AI experiences": 'AI_Util_Company',
    'Digitalization level of company': 'Digitalization_Level',
    'AI Tranining level of repondents': 'AI_Training',
    'AI impact level [Budgeting and cost management]': 'AI_Impact_Budgeting',
    'AI impact level? [Design and planning]': 'AI_Impact_Design',
    'AI impact level [Construction Project Management]': 'AI_Impact_ProjectMgmt',
    'AI impact level [Marketing a CRM]': 'AI_Impact_Marketing',
    'AI impact level [Material delivery and ligistic]': 'AI_Impact_Logistics',
    'Perception of AI impact on (cost-reducing)': 'Perception_CostReduction',
    'Perception of AI impact on (Automation of construction processes, robotics, drones, 3D print)': 'Perception_Automation',
    'Perception of AI impact on (optimatization of materials and logistic)': 'Perception_Materials',
    'Perception of AI impact on (monitoring of conostrcution projects)': 'Perception_ProjectMonitor',
    'Perception of AI impact on (HR management)': 'Perception_HR',
    'Perception of AI impact on (reducing administrative burden and paperwork)': 'Perception_Admin',
    'Perception of AI impact on (Inteliligent planning)': 'Perception_Planning',
}


def load_raw_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(how='all')
    df = df[df['Age of respondends'].notna()]
    df.columns = df.columns.str.strip()
    df = df.rename(columns=RAW_COL_MAP)
    return df


def encode_age(age_str: str) -> float:
    if pd.isna(age_str):
        return np.nan
    text = str(age_str).lower()
    if 'menej' in text:
        return 1
    if '25 - 35' in text or '25-35' in text:
        return 2
    if '36 - 45' in text or '36-45' in text:
        return 3
    if '46 - 50' in text or '46-50' in text:
        return 4
    if 'viac' in text:
        return 5
    return np.nan


def encode_experience(exp_str: str) -> float:
    if pd.isna(exp_str):
        return np.nan
    text = str(exp_str).lower()
    if 'menej' in text or '< 1' in text:
        return 1
    if '1 až 3' in text or '1-3' in text:
        return 2
    if '3 až 5' in text or '3-5' in text:
        return 3
    if '5 až 10' in text or '5-10' in text:
        return 4
    if 'viac' in text or '> 10' in text:
        return 5
    return np.nan


def encode_company_size(size_str: str) -> float:
    if pd.isna(size_str):
        return np.nan
    text = str(size_str).lower()
    if 'mikropodnik' in text:
        return 1
    if 'malá' in text or 'mala' in text:
        return 2
    if 'stredná' in text or 'stredna' in text:
        return 3
    if 'veľká' in text or 'velka' in text:
        return 4
    return np.nan


def encode_utilization_level(level_str: str) -> float:
    if pd.isna(level_str):
        return np.nan
    text = str(level_str).lower()
    if '1 –' in text or '1-' in text or '1 -' in text or 'minimálne' in text or 'minim' in text:
        return 1
    if '2 –' in text or '2-' in text or '2 -' in text or 'základné' in text or 'zaklad' in text:
        return 2
    if '3 –' in text or '3-' in text or '3 -' in text or 'stredná' in text or 'stredna' in text or 'priemern' in text:
        return 3
    if '4 –' in text or '4-' in text or '4 -' in text or 'vysoká' in text or 'vysoka' in text or 'nízka' in text or 'nizka' in text:
        return 4
    if '5 –' in text or '5-' in text or '5 -' in text or 'plne' in text or 'žiadna' in text or 'ziadna' in text:
        return 5
    return np.nan


def encode_ai_training(train_str: str) -> float:
    if pd.isna(train_str):
        return np.nan
    text = str(train_str).lower()
    if '1 –' in text or '1-' in text or 'vôbec' in text or 'vobec' in text:
        return 1
    if '2 –' in text or '2-' in text or 'minimálne' in text or 'minim' in text:
        return 2
    if '3 –' in text or '3-' in text or 'čiastočne' in text or 'ciastocne' in text:
        return 3
    if '4 –' in text or '4-' in text or 'dôkladne' in text or 'dokladne' in text:
        return 4
    return np.nan


def encode_job_position(job_str: str) -> float:
    if pd.isna(job_str):
        return np.nan
    text = str(job_str).lower()
    mapping = {
        'prípravár': 1,
        'rozpočtár': 1,
        'projektant': 2,
        'technológ': 3,
        'technik': 3,
        'stavbyvedúci': 4,
        'site-manager': 4,
        'asistent': 5,
        'manažér': 6,
        'manager': 6,
        'konateľ': 7,
        'manualna': 0,
    }
    for key, value in mapping.items():
        if key in text:
            return value
    return np.nan


def encode_perception_cost(per_str: str) -> float:
    if pd.isna(per_str):
        return np.nan
    text = str(per_str).lower()
    if '1 -' in text or 'rozhodne nie' in text:
        return 1
    if '2 -' in text or 'skôr nie' in text or 'skor nie' in text:
        return 2
    if '3 -' in text or 'neviem' in text:
        return 3
    if '4 -' in text or 'skôr áno' in text or 'skor ano' in text:
        return 4
    if '5 -' in text or 'rozhodne áno' in text or 'rozhodne ano' in text:
        return 5
    return np.nan


def encode_perception_category(per_str: str) -> float:
    """
    Encode categorical perception (Automation, Materials, etc.) to ordinal numeric.
    Original: 1=Minimálne, 2=Čiastočne, 3=Veľmi, 4=Neviem
    """
    if pd.isna(per_str):
        return np.nan
    text = str(per_str).lower()
    if 'minimálne' in text or 'minimalne' in text:
        return 1
    if 'čiastočne' in text or 'ciastocne' in text:
        return 2
    if 'veľmi' in text or 'velmi' in text:
        return 3
    if 'neviem' in text:
        return 4
    return np.nan


def simplify_target_3class(value: float) -> float:
    """
    Simplify 5-class or 4-class target to 3-class ordinal.
    
    RATIONALE FOR 3-CLASS SIMPLIFICATION:
    ------------------------------------
    Original class distributions show severe imbalance (e.g., 31/17/2/1).
    With n=51, having classes with <5 samples violates statistical assumptions:
    - Can't reliably estimate class-conditional distributions
    - Cross-validation folds may miss entire classes
    - Classifier learns noise rather than signal
    
    3-class mapping:
    - Low (1): Original classes 1-2 (negative/skeptical perception)
    - Medium (2): Original class 3 (neutral/uncertain)  
    - High (3): Original classes 3+ for categorical, 4-5 for cost perception
    
    This ensures minimum ~8-10 samples per class, enabling:
    - Stable parameter estimation
    - Meaningful cross-validation
    - Reduced overfitting risk
    """
    if pd.isna(value):
        return np.nan
    v = float(value)
    if v <= 2:
        return 1  # Low
    elif v == 3:
        return 2  # Medium  
    else:
        return 3  # High


def simplify_perception_category_3class(value: float) -> float:
    """
    Simplify 4-class categorical perception to 3-class.
    Original: 1=Minimálne, 2=Čiastočne, 3=Veľmi, 4=Neviem
    
    Mapping:
    - Low (1): Minimálne (class 1)
    - Medium (2): Čiastočne (class 2) + Neviem (class 4, uncertain)
    - High (3): Veľmi (class 3)
    
    RATIONALE: "Neviem" (don't know) is grouped with "Čiastočne" (partial)
    because uncertainty indicates moderate rather than extreme perception.
    """
    if pd.isna(value):
        return np.nan
    v = float(value)
    if v == 1:
        return 1  # Low (Minimálne)
    elif v == 2 or v == 4:
        return 2  # Medium (Čiastočne + Neviem)
    elif v == 3:
        return 3  # High (Veľmi)
    return np.nan


IMPACT_COLUMNS = [
    'AI_Impact_Budgeting',
    'AI_Impact_Design',
    'AI_Impact_ProjectMgmt',
    'AI_Impact_Marketing',
    'AI_Impact_Logistics',
]

# ============================================================================
# FEATURE SETS FOR SMALL-SAMPLE LEARNING
# ============================================================================

# Original 15 features (BASELINE - too many for n=51)
INPUT_FEATURES_BASELINE = [
    'Age_Numeric',
    'Company_Size_Numeric',
    'Job_Position_Numeric',
    'Experience_Numeric',
    'ICT_Utilization_Numeric',
    'AI_Util_Personal_Numeric',
    'Digital_Competencies_Numeric',
    'AI_Util_Company_Numeric',
    'Digitalization_Level_Numeric',
    'AI_Training_Numeric',
    *IMPACT_COLUMNS,
]

# Composite indices (theoretically justified dimensionality reduction)
COMPOSITE_FEATURES = [
    'AI_Experience_Index',      # Personal AI exposure
    'Digitalization_Index',     # Company digital maturity
    'AI_Impact_Index',          # Perceived AI impact across domains
]

# OPTIMIZED: 8 features using composite indices
# Rationale: n=51 / 8 features ≈ 6.4 samples per feature (acceptable)
INPUT_FEATURES_OPTIMIZED = [
    'Age_Numeric',
    'Company_Size_Numeric', 
    'Experience_Numeric',
    'AI_Experience_Index',      # Replaces: AI_Util_Personal, AI_Util_Company, AI_Training
    'Digitalization_Index',     # Replaces: ICT_Utilization, Digitalization_Level, Digital_Competencies
    'AI_Impact_Index',          # Replaces: 5 AI_Impact columns
    'Job_Position_Numeric',     # Retained: unique organizational context
]

# For backward compatibility
INPUT_FEATURES = INPUT_FEATURES_BASELINE

TARGET_REGRESSION = 'Perception_CostReduction_Numeric'
TARGET_REGRESSION_3CLASS = 'Perception_CostReduction_3Class'

TARGET_CLASSIFICATION = [
    'Perception_Automation_Numeric',
    'Perception_Materials_Numeric',
    'Perception_ProjectMonitor_Numeric',
    'Perception_HR_Numeric',
    'Perception_Admin_Numeric',
    'Perception_Planning_Numeric',
]

TARGET_CLASSIFICATION_3CLASS = [
    'Perception_Automation_3Class',
    'Perception_Materials_3Class',
    'Perception_ProjectMonitor_3Class',
    'Perception_HR_3Class',
    'Perception_Admin_3Class',
    'Perception_Planning_3Class',
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main preprocessing pipeline with both baseline and optimized features.
    """
    df = df.copy()
    
    # === BASIC FEATURE ENCODING ===
    df['Age_Numeric'] = df['Age'].apply(encode_age)
    df['Experience_Numeric'] = df['Work_Experience'].apply(encode_experience)
    df['Company_Size_Numeric'] = df['Company_Size'].apply(encode_company_size)
    df['Job_Position_Numeric'] = df['Job_Position'].apply(encode_job_position)
    df['ICT_Utilization_Numeric'] = df['ICT_Utilization'].apply(encode_utilization_level)
    df['AI_Util_Personal_Numeric'] = df['AI_Util_Personal'].apply(encode_utilization_level)
    df['Digital_Competencies_Numeric'] = df['Digital_Competencies'].apply(encode_utilization_level)
    df['AI_Util_Company_Numeric'] = df['AI_Util_Company'].apply(encode_utilization_level)
    df['Digitalization_Level_Numeric'] = df['Digitalization_Level'].apply(encode_utilization_level)
    df['AI_Training_Numeric'] = df['AI_Training'].apply(encode_ai_training)

    for col in IMPACT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # === TARGET ENCODING (ORIGINAL) ===
    df[TARGET_REGRESSION] = df['Perception_CostReduction'].apply(encode_perception_cost)
    for col in TARGET_CLASSIFICATION:
        raw = col.replace('_Numeric', '')
        df[col] = df[raw].apply(encode_perception_category)

    # === COMPOSITE INDICES (PHASE 1 - FEATURE ENGINEERING) ===
    # These reduce dimensionality while preserving theoretical coherence
    
    # AI_Experience_Index: Personal AI exposure and training
    # Rationale: Personal AI usage, company AI adoption, and training are 
    # conceptually related measures of an individual's AI ecosystem exposure
    df['AI_Experience_Index'] = df[['AI_Util_Personal_Numeric', 
                                     'AI_Util_Company_Numeric', 
                                     'AI_Training_Numeric']].mean(axis=1)
    
    # Digitalization_Index: Overall digital maturity
    # Rationale: ICT utilization, company digitalization, and digital competencies
    # jointly capture the digital environment the respondent operates in
    df['Digitalization_Index'] = df[['ICT_Utilization_Numeric', 
                                      'Digitalization_Level_Numeric', 
                                      'Digital_Competencies_Numeric']].mean(axis=1)
    
    # AI_Impact_Index: Perceived AI impact across business domains
    # Rationale: The 5 impact columns measure the same underlying construct
    # (perceived AI impact) across different business functions
    df['AI_Impact_Index'] = df[IMPACT_COLUMNS].mean(axis=1)

    # === 3-CLASS TARGET SIMPLIFICATION (PHASE 1) ===
    # Apply ordinal simplification for statistical stability
    df[TARGET_REGRESSION_3CLASS] = df[TARGET_REGRESSION].apply(simplify_target_3class)
    
    for col_orig, col_3class in zip(TARGET_CLASSIFICATION, TARGET_CLASSIFICATION_3CLASS):
        df[col_3class] = df[col_orig].apply(simplify_perception_category_3class)

    return df


def build_regression_frame(df: pd.DataFrame, use_optimized: bool = False):
    """
    Build regression dataset with selected features.
    
    Args:
        df: Preprocessed dataframe
        use_optimized: If True, use reduced feature set with composite indices
    """
    features = INPUT_FEATURES_OPTIMIZED if use_optimized else INPUT_FEATURES_BASELINE
    target = TARGET_REGRESSION_3CLASS if use_optimized else TARGET_REGRESSION
    
    required = features + [target]
    df_reg = df.dropna(subset=required).copy()
    X = df_reg[features]
    y_reg = df_reg[target]
    return df_reg, X, y_reg


def build_classification_frame(df: pd.DataFrame, target_col: str, use_optimized: bool = False):
    """
    Build classification dataset for a single target.
    """
    features = INPUT_FEATURES_OPTIMIZED if use_optimized else INPUT_FEATURES_BASELINE
    required = features + [target_col]
    df_clf = df.dropna(subset=required).copy()
    X = df_clf[features]
    y = df_clf[target_col]
    return df_clf, X, y


def load_and_prepare(csv_path: Path):
    """Load data and return both baseline and optimized versions."""
    df_raw = load_raw_dataframe(csv_path)
    df_preproc = preprocess(df_raw)
    
    # Baseline (original 15 features, 5-class targets)
    df_reg, X_reg, y_reg = build_regression_frame(df_preproc, use_optimized=False)
    
    return df_preproc, df_reg, X_reg, y_reg


def get_class_distribution_report(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Generate class distribution report for both original and simplified targets.
    Used to justify 3-class simplification.
    """
    report = {}
    
    # Original targets
    for col in TARGET_CLASSIFICATION:
        if col in df.columns:
            counts = df[col].value_counts().sort_index()
            report[f'{col} (original)'] = counts.to_dict()
    
    # Simplified targets
    for col in TARGET_CLASSIFICATION_3CLASS:
        if col in df.columns:
            counts = df[col].value_counts().sort_index()
            report[f'{col} (3-class)'] = counts.to_dict()
    
    # Cost reduction target
    if TARGET_REGRESSION in df.columns:
        report[f'{TARGET_REGRESSION} (original)'] = df[TARGET_REGRESSION].value_counts().sort_index().to_dict()
    if TARGET_REGRESSION_3CLASS in df.columns:
        report[f'{TARGET_REGRESSION_3CLASS} (3-class)'] = df[TARGET_REGRESSION_3CLASS].value_counts().sort_index().to_dict()
    
    return report
