# pipelines/data_pipeline.py
import pandas as pd
from utils.helpers import load_data_from_db

def clean_risk_data(risk_df):
    # Handle missing values
    risk_df['riskDescription'].fillna("No description", inplace=True)
    risk_df['riskResponse'].fillna("No response", inplace=True)
    
    # Convert riskScore to a numerical value if it's not already
    risk_df['riskScore'] = pd.to_numeric(risk_df['riskScore'], errors='coerce')
    
    # Drop rows with missing riskScore
    risk_df.dropna(subset=['riskScore'], inplace=True)
    
    return risk_df

def clean_risk_mitigation_data(risk_mitigation_df):
    # Handle missing values
    risk_mitigation_df['mitigationCost'].fillna(0, inplace=True)
    risk_mitigation_df['mitigationEffort'].fillna("Low", inplace=True)
    
    return risk_mitigation_df

def merge_data(risk_df, risk_mitigation_df):
    # Merge Risk and RiskMitigation tables on riskId
    merged_df = pd.merge(risk_df, risk_mitigation_df, left_on='id', right_on='riskId', how='left')
    return merged_df

def run_pipeline():
    # Load data
    risk_df = load_data_from_db("Risk")
    risk_mitigation_df = load_data_from_db("RiskMitigation")
    
    # Clean data
    risk_df = clean_risk_data(risk_df)
    risk_mitigation_df = clean_risk_mitigation_data(risk_mitigation_df)
    
    # Merge data
    merged_df = merge_data(risk_df, risk_mitigation_df)
    
    # Save processed data
    merged_df.to_csv("/workspaces/risk-model/data/processed/processed_data.csv", index=False)
    print("Data pipeline completed. Processed data saved to data/processed/processed_data.csv.")

if __name__ == "__main__":
    run_pipeline()
    