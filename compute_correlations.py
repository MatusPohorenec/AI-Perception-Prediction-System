"""
Compute and save correlation matrix and feature importance for the app.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.data_prep import load_raw_dataframe, preprocess

def compute_correlations():
    """Compute correlation matrix from survey data."""
    # Load data
    df = load_raw_dataframe(Path("Survey in 2025.csv"))
    df = preprocess(df)
    
    # Select numeric columns for correlation
    feature_cols = [
        'Age', 'Company_Size', 'Job_Position', 'Work_Experience',
        'ICT_Utilization', 'Personal_AI_Usage', 'Digital_Competencies',
        'Company_AI_Usage', 'Digitalization_Level', 'AI_Training',
        'AI_Impact_Productivity', 'AI_Impact_Job_Security', 
        'AI_Impact_Skills', 'AI_Impact_Work_Quality', 'AI_Impact_Cost'
    ]
    
    target_cols = [
        'Target_Productivity', 'Target_Job_Security', 'Target_Skills',
        'Target_Quality', 'Target_Cost', 'Target_Innovation', 'Target_Readiness'
    ]
    
    # Get available columns
    available_features = [c for c in feature_cols if c in df.columns]
    available_targets = [c for c in target_cols if c in df.columns]
    
    print(f"Available features: {available_features}")
    print(f"Available targets: {available_targets}")
    
    # If we don't have exact column names, use what we have
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"All numeric columns: {numeric_cols}")
    
    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Save correlation matrix
    output_path = Path("saved_models")
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / "correlation_matrix.pkl", "wb") as f:
        pickle.dump({
            'correlation_matrix': corr_matrix,
            'columns': numeric_cols
        }, f)
    
    print(f"\nCorrelation matrix saved to {output_path / 'correlation_matrix.pkl'}")
    print(f"Shape: {corr_matrix.shape}")
    
    # Print top correlations
    print("\n=== Top Feature Correlations ===")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(corr_pairs)
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    
    print(corr_df.head(20).to_string(index=False))
    
    return corr_matrix

if __name__ == "__main__":
    compute_correlations()
