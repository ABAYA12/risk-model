#!/usr/bin/env python3
"""
RiskGuard AI/ML Pipeline for a Risk Management Application.
This single-file solution includes:
 - Direct DB connection using SQLAlchemy.
 - Data extraction (with capitalized table names quoted) from the key tables.
 - Data cleaning and transformation using Pandas.
 - Training of ML models for risk prediction, anomaly detection, etc.
 - Generation of visualizations (histograms, scatter plots, line charts, etc.).
 - Aggregation into a Risk Advice engine that provides context-aware recommendations.
 - A simulated monitoring function for continuous data updates.

Note:
- In production, responsibilities might be split between Data Engineers (ingestion/cleaning)
  and Data Scientists/ML Engineers (modeling/analysis).
- This code uses only free open-source tools.
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, accuracy_score

import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Configuration and DB Connection
# -----------------------------
# Set your database URL (update with your credentials)
DB_URL = os.getenv('DB_URL',"postgresql://postgres.njsusvrlnigkiduefyax:GfSNK6BnVuBUYEf7@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
)

def get_db_engine(db_url: str) -> Engine:
    """Create and return a SQLAlchemy engine."""
    try:
        engine = create_engine(db_url)
        print("Database engine created successfully.")
        return engine
    except SQLAlchemyError as e:
        print(f"Error connecting to the database: {e}")
        raise

# -----------------------------
# Data Extraction and Cleaning
# -----------------------------
class DataPipeline:
    def __init__(self, engine: Engine, organization_id: int):
        self.engine = engine
        self.organization_id = organization_id  # Filter to get company-specific data
        
        # Initialize dataframes (we only load the tables we need for predictions)
        self.df_risk = pd.DataFrame()
        self.df_risk_mitigation = pd.DataFrame()
        self.df_risk_status = pd.DataFrame()
        self.df_risk_monitoring = pd.DataFrame()
        self.df_user = pd.DataFrame()
        self.df_user_role = pd.DataFrame()

    def load_data(self):
        """Extract the required tables from the DB for the given organization."""
        print("Loading data from database ...")
        try:
            # 1. "Risk" Table
            query_risk = f'SELECT * FROM "Risk" WHERE "organizationId" = {self.organization_id}'
            self.df_risk = pd.read_sql(query_risk, self.engine)
            print(f'"Risk" table loaded: {self.df_risk.shape[0]} records')

            # 2. "RiskMitigation" Table
            query_risk_mitigation = (f'SELECT * FROM "RiskMitigation" WHERE "riskId" IN '
                                     f'(SELECT "id" FROM "Risk" WHERE "organizationId" = {self.organization_id})')
            self.df_risk_mitigation = pd.read_sql(query_risk_mitigation, self.engine)
            print(f'"RiskMitigation" table loaded: {self.df_risk_mitigation.shape[0]} records')

            # 3. "RiskStatus" Table
            query_risk_status = (f'SELECT * FROM "RiskStatus" WHERE "risk_id" IN '
                                 f'(SELECT "id" FROM "Risk" WHERE "organizationId" = {self.organization_id})')
            self.df_risk_status = pd.read_sql(query_risk_status, self.engine)
            print(f'"RiskStatus" table loaded: {self.df_risk_status.shape[0]} records')

            # 4. "RiskMonitoring" Table
            query_risk_monitoring = (f'SELECT * FROM "RiskMonitoring" WHERE "riskId" IN '
                                     f'(SELECT "id" FROM "Risk" WHERE "organizationId" = {self.organization_id})')
            self.df_risk_monitoring = pd.read_sql(query_risk_monitoring, self.engine)
            print(f'"RiskMonitoring" table loaded: {self.df_risk_monitoring.shape[0]} records')

            # 5. "User" Table
            query_user = f'SELECT * FROM "User" WHERE "organizationId" = {self.organization_id}'
            self.df_user = pd.read_sql(query_user, self.engine)
            print(f'"User" table loaded: {self.df_user.shape[0]} records')

            # 6. "UserRole" Table
            query_user_role = (
                'SELECT ur.* FROM "UserRole" ur '
                'JOIN "User" u ON ur."userEmail" = u."email" '
                f'WHERE u."organizationId" = {self.organization_id}'
            )
            self.df_user_role = pd.read_sql(query_user_role, self.engine)
            print(f'"UserRole" table loaded: {self.df_user_role.shape[0]} records')

        except SQLAlchemyError as e:
            print(f"Error loading data: {e}")
            raise

    def clean_data(self):
        """Perform basic data cleaning and type conversion for each DataFrame."""
        print("Cleaning data ...")
        # Cleaning "Risk" table
        if not self.df_risk.empty:
            # Convert createdAt/updatedAt columns to datetime
            self.df_risk['createdAt'] = pd.to_datetime(self.df_risk['createdAt'], errors='coerce')
            self.df_risk['updatedAt'] = pd.to_datetime(self.df_risk['updatedAt'], errors='coerce')
            # Drop rows missing critical columns
            self.df_risk = self.df_risk.dropna(subset=['riskScore', 'riskCategory'])
            # Ensure numeric fields are numeric
            self.df_risk['riskScore'] = pd.to_numeric(self.df_risk['riskScore'], errors='coerce')
            self.df_risk['riskImpactLevel'] = pd.to_numeric(self.df_risk['riskImpactLevel'], errors='coerce')
            self.df_risk['riskProbabilityLevel'] = pd.to_numeric(self.df_risk['riskProbabilityLevel'], errors='coerce')
        
        # Cleaning "RiskMitigation" table
        if not self.df_risk_mitigation.empty:
            self.df_risk_mitigation['mitigatedRiskScore'] = pd.to_numeric(self.df_risk_mitigation['mitigatedRiskScore'], errors='coerce')
            self.df_risk_mitigation['mitigatedRiskProbabilityLevel'] = pd.to_numeric(self.df_risk_mitigation['mitigatedRiskProbabilityLevel'], errors='coerce')
            self.df_risk_mitigation['mitigatedRiskImpactLevel'] = pd.to_numeric(self.df_risk_mitigation['mitigatedRiskImpactLevel'], errors='coerce')
        
        # Cleaning "RiskStatus" table
        if not self.df_risk_status.empty:
            self.df_risk_status['createdAt'] = pd.to_datetime(self.df_risk_status['createdAt'], errors='coerce')
        
        # Cleaning "RiskMonitoring" table
        if not self.df_risk_monitoring.empty:
            self.df_risk_monitoring['createdAt'] = pd.to_datetime(self.df_risk_monitoring['createdAt'], errors='coerce')
            self.df_risk_monitoring['updatedAt'] = pd.to_datetime(self.df_risk_monitoring['updatedAt'], errors='coerce')
        
        # Cleaning "User" table
        if not self.df_user.empty:
            self.df_user['createdAt'] = pd.to_datetime(self.df_user['createdAt'], errors='coerce')
            self.df_user['updatedAt'] = pd.to_datetime(self.df_user['updatedAt'], errors='coerce')
        
        # Cleaning "UserRole" table
        if not self.df_user_role.empty:
            self.df_user_role['createdAt'] = pd.to_datetime(self.df_user_role['createdAt'], errors='coerce')
            self.df_user_role['updatedAt'] = pd.to_datetime(self.df_user_role['updatedAt'], errors='coerce')
        
        print("Data cleaning completed.")

# -----------------------------
# ML Model Training and Predictions
# -----------------------------
class RiskModels:
    def __init__(self, data_pipeline: DataPipeline):
        self.data_pipeline = data_pipeline
        # Models that will be trained:
        self.risk_regressor = None  # Predict continuous riskScore
        self.risk_classifier = None # Classify riskCategory (low/medium/high)
        self.anomaly_detector = None  # Isolation Forest for anomaly detection

    def train_risk_prediction_models(self):
        """
        Train two models on the "Risk" table:
         - A regression model to predict riskScore.
         - A classification model to predict riskCategory.
        """
        print("Training Risk Prediction Models ...")
        df = self.data_pipeline.df_risk.copy()
        if df.empty:
            print("Risk data is empty. Cannot train models.")
            return

        # Use riskProbabilityLevel and riskImpactLevel as features to predict riskScore
        features = ['riskProbabilityLevel', 'riskImpactLevel']
        df = df.dropna(subset=features + ['riskScore', 'riskCategory'])
        X = df[features]
        y_score = df['riskScore']

        # Train a simple linear regressor
        X_train, X_test, y_train, y_test = train_test_split(X, y_score, test_size=0.2, random_state=42)
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        self.risk_regressor = regressor
        pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        print(f"Risk Score Regression MSE: {mse:.2f}")

        # For classification, assume riskCategory is one of ('low', 'medium', 'high')
        df['riskCategoryNumeric'] = df['riskCategory'].map({'low': 0, 'medium': 1, 'high': 2})
        df = df.dropna(subset=['riskCategoryNumeric'])
        y_cat = df['riskCategoryNumeric']
        X_cat = df[features]
        X_train, X_test, y_train, y_test = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42)
        classifier = LogisticRegression(multi_class='multinomial', max_iter=200)
        classifier.fit(X_train, y_train)
        self.risk_classifier = classifier
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Risk Category Classification Accuracy: {acc:.2f}")

    def train_anomaly_detector(self):
        """
        Train an anomaly detection model using Isolation Forest on the riskScore.
        """
        print("Training Anomaly Detection Model ...")
        df = self.data_pipeline.df_risk.copy()
        if df.empty:
            print("Risk data is empty. Cannot train anomaly detector.")
            return

        # Use riskScore as the input for anomaly detection (reshape to 2D)
        X = df[['riskScore']].dropna()
        isolation_forest = IsolationForest(contamination=0.05, random_state=42)
        isolation_forest.fit(X)
        self.anomaly_detector = isolation_forest
        # Optionally, add anomaly scores to the DataFrame for visualization.
        df['anomaly_score'] = isolation_forest.decision_function(X)
        df['anomaly'] = isolation_forest.predict(X)
        self.data_pipeline.df_risk = df  # update with anomaly results
        print("Anomaly Detection model trained.")

    def generate_predictions(self):
        """
        Generate predictions using the trained models and add results to the data.
        """
        print("Generating predictions ...")
        df = self.data_pipeline.df_risk.copy()
        if self.risk_regressor is None or self.risk_classifier is None:
            print("Models are not trained.")
            return df
        
        features = ['riskProbabilityLevel', 'riskImpactLevel']
        df = df.dropna(subset=features)
        df['predicted_riskScore'] = self.risk_regressor.predict(df[features])
        df['predicted_riskCategoryNumeric'] = self.risk_classifier.predict(df[features])
        mapping = {0: 'low', 1: 'medium', 2: 'high'}
        df['predicted_riskCategory'] = df['predicted_riskCategoryNumeric'].map(mapping)
        self.data_pipeline.df_risk = df
        print("Predictions generated.")
        return df

# -----------------------------
# Visualization Functions
# -----------------------------
class Visualizations:
    def __init__(self, dp: DataPipeline):
        self.dp = dp

    def plot_risk_score_distribution(self):
        """Histogram: Distribution of Risk Scores by Category."""
        df = self.dp.df_risk.copy()
        if df.empty:
            print("No risk data for plotting.")
            return
        plt.figure(figsize=(8,6))
        sns.histplot(data=df, x='riskScore', hue='riskCategory', multiple='stack', bins=20)
        plt.title("Distribution of Risk Scores by Category (Low/Medium/High)")
        plt.xlabel("Risk Score")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("risk_score_distribution.png")
        plt.close()
        print('Risk Score Distribution plot saved as "risk_score_distribution.png"')

    def plot_risk_probability_vs_impact(self):
        """Scatter Plot: Correlation between Risk Probability and Impact Levels."""
        df = self.dp.df_risk.copy()
        if df.empty:
            print("No risk data for plotting.")
            return
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x='riskProbabilityLevel', y='riskImpactLevel', hue='riskCategory')
        plt.title("Correlation Between Risk Probability and Impact Levels")
        plt.xlabel("Risk Probability Level")
        plt.ylabel("Risk Impact Level")
        plt.tight_layout()
        plt.savefig("risk_prob_vs_impact.png")
        plt.close()
        print('Risk Probability vs Impact plot saved as "risk_prob_vs_impact.png"')

    def plot_risk_trends_over_time(self):
        """Line Chart: Risk Score Trends Over Time."""
        df = self.dp.df_risk.copy()
        if df.empty:
            print("No risk data for plotting.")
            return
        df = df.dropna(subset=['createdAt'])
        df['Month'] = df['createdAt'].dt.to_period('M')
        trend = df.groupby('Month')['riskScore'].mean().reset_index()
        trend['Month'] = trend['Month'].dt.to_timestamp()
        plt.figure(figsize=(10,6))
        sns.lineplot(data=trend, x='Month', y='riskScore', marker="o")
        plt.title("Risk Score Trends Over Time (Monthly Average)")
        plt.xlabel("Month")
        plt.ylabel("Average Risk Score")
        plt.tight_layout()
        plt.savefig("risk_trends_over_time.png")
        plt.close()
        print('Risk Trends Over Time plot saved as "risk_trends_over_time.png"')

    def plot_mitigation_effectiveness(self):
        """Bar Chart: Impact of Mitigation Strategies on Reducing Risk Scores."""
        df = self.dp.df_risk_mitigation.copy()
        if df.empty:
            print("No risk mitigation data for plotting.")
            return
        plt.figure(figsize=(8,6))
        sns.barplot(data=df, x='mitigationControl', y='mitigatedRiskScore')
        plt.title("Impact of Mitigation Strategies on Reducing Risk Scores")
        plt.xlabel("Mitigation Control")
        plt.ylabel("Mitigated Risk Score")
        plt.tight_layout()
        plt.savefig("mitigation_effectiveness.png")
        plt.close()
        print('Mitigation Effectiveness plot saved as "mitigation_effectiveness.png"')

    def plot_cost_vs_effort_of_mitigation(self):
        """Heatmap: Cost vs. Effort of Mitigation Strategies."""
        df = self.dp.df_risk_mitigation.copy()
        if df.empty:
            print("No risk mitigation data for plotting.")
            return
        pivot = df.pivot_table(index='mitigationEffort', columns='mitigationCost', aggfunc='size', fill_value=0)
        plt.figure(figsize=(8,6))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")
        plt.title("Cost vs. Effort of Mitigation Strategies")
        plt.xlabel("Mitigation Cost")
        plt.ylabel("Mitigation Effort")
        plt.tight_layout()
        plt.savefig("cost_vs_effort_mitigation.png")
        plt.close()
        print('Cost vs Effort Mitigation heatmap saved as "cost_vs_effort_mitigation.png"')

    def plot_mitigation_by_risk_category(self):
        """Stacked Bar Chart: Most Common Mitigation Strategies by Risk Category."""
        df_risk = self.dp.df_risk[['id', 'riskCategory']]
        df_mitig = self.dp.df_risk_mitigation.copy()
        if df_risk.empty or df_mitig.empty:
            print("Insufficient data for mitigation by risk category plotting.")
            return
        merged = pd.merge(df_mitig, df_risk, left_on='riskId', right_on='id', how='inner')
        count_df = merged.groupby(['riskCategory', 'mitigationControl']).size().unstack(fill_value=0)
        count_df.plot(kind='bar', stacked=True, figsize=(10,6))
        plt.title("Most Common Mitigation Strategies by Risk Category")
        plt.xlabel("Risk Category")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("mitigation_by_risk_category.png")
        plt.close()
        print('Mitigation by Risk Category plot saved as "mitigation_by_risk_category.png"')

    def plot_outlier_detection(self):
        """Box Plot: Outliers in Risk Scores by Category."""
        df = self.dp.df_risk.copy()
        if df.empty:
            print("No risk data for plotting outliers.")
            return
        plt.figure(figsize=(8,6))
        sns.boxplot(data=df, x='riskCategory', y='riskScore')
        plt.title("Outliers in Risk Scores by Category")
        plt.xlabel("Risk Category")
        plt.ylabel("Risk Score")
        plt.tight_layout()
        plt.savefig("outlier_detection.png")
        plt.close()
        print('Outlier Detection plot saved as "outlier_detection.png"')

    def plot_sudden_risk_spikes(self):
        """Time-Series Anomaly: Unusual Risk Score Spikes Over Time."""
        df = self.dp.df_risk.copy()
        if df.empty:
            print("No risk data for plotting sudden spikes.")
            return
        df = df.dropna(subset=['createdAt'])
        df = df.sort_values('createdAt')
        plt.figure(figsize=(10,6))
        plt.plot(df['createdAt'], df['riskScore'], label='Risk Score')
        anomalies = df[df.get('anomaly', 1) == -1]
        plt.scatter(anomalies['createdAt'], anomalies['riskScore'], color='red', label='Anomaly')
        plt.title("Unusual Risk Score Spikes Over Time (Anomalies Highlighted)")
        plt.xlabel("Time")
        plt.ylabel("Risk Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig("sudden_risk_spikes.png")
        plt.close()
        print('Sudden Risk Spikes plot saved as "sudden_risk_spikes.png"')

    def plot_risk_status_distribution(self):
        """Pie Chart: Proportion of Risks by Status (Monitored vs. Closed)."""
        df = self.dp.df_risk_status.copy()
        if df.empty:
            print("No risk status data for plotting.")
            return
        status_counts = df['statusType'].value_counts()
        plt.figure(figsize=(6,6))
        plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title("Proportion of Risks by Status")
        plt.tight_layout()
        plt.savefig("risk_status_distribution.png")
        plt.close()
        print('Risk Status Distribution pie chart saved as "risk_status_distribution.png"')

    def plot_risk_response_activity(self):
        """Gantt Chart (Simulated): Timeline of Risk Response Activities and Implementation Progress."""
        df = self.dp.df_risk_monitoring.copy()
        if df.empty:
            print("No risk monitoring data for Gantt chart.")
            return
        
        df['start'] = pd.to_datetime(df['createdAt'], errors='coerce')
        df['end'] = pd.to_datetime(df.get('endDate', None), errors='coerce')
        df['end'] = df['end'].fillna(df['start'] + pd.Timedelta(days=7))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, row in df.iterrows():
            ax.barh(row['riskId'], (row['end'] - row['start']).days, left=row['start'], color='skyblue')
            ax.text(row['start'] + (row['end']-row['start'])/2, idx, str(row['riskResponseActivitiyStatus']), 
                    va='center', ha='center', color='black')
        ax.set_xlabel("Date")
        ax.set_ylabel("Risk ID")
        plt.title("Timeline of Risk Response Activities and Implementation Progress")
        plt.tight_layout()
        plt.savefig("risk_response_activity.png")
        plt.close()
        print('Risk Response Activity Gantt chart saved as "risk_response_activity.png"')

    def plot_user_risk_submission_patterns(self):
        """Line Chart: Frequency of Risk Submissions by User Role/Department."""
        df_user = self.dp.df_user[['id', 'email']]
        df_risk = self.dp.df_risk.copy()
        if df_user.empty or df_risk.empty:
            print("Insufficient data for user risk submission patterns.")
            return
        merged = pd.merge(df_risk, df_user, left_on='submittedBy', right_on='email', how='inner')
        merged = merged.dropna(subset=['createdAt'])
        merged['createdAt'] = pd.to_datetime(merged['createdAt'], errors='coerce')
        merged['Month'] = merged['createdAt'].dt.to_period('M')
        submission_counts = merged.groupby('Month').size().reset_index(name='count')
        submission_counts['Month'] = submission_counts['Month'].dt.to_timestamp()
        plt.figure(figsize=(10,6))
        sns.lineplot(data=submission_counts, x='Month', y='count', marker="o")
        plt.title("Frequency of Risk Submissions Over Time")
        plt.xlabel("Month")
        plt.ylabel("Number of Submissions")
        plt.tight_layout()
        plt.savefig("user_risk_submission_patterns.png")
        plt.close()
        print('User Risk Submission Patterns plot saved as "user_risk_submission_patterns.png"')

    def plot_user_role_vs_risk_severity(self):
        """Violin Plot: Distribution of Risk Scores by User Role."""
        df_risk = self.dp.df_risk.copy()
        df_user = self.dp.df_user[['email', 'organizationId']]
        df_role = self.dp.df_user_role.copy()
        if df_risk.empty or df_user.empty or df_role.empty:
            print("Insufficient data for user role vs risk severity plot.")
            return
        merged = pd.merge(df_risk, df_user, left_on='submittedBy', right_on='email', how='inner')
        merged = pd.merge(merged, df_role, left_on='submittedBy', right_on='userEmail', how='inner')
        plt.figure(figsize=(10,6))
        sns.violinplot(data=merged, x='position', y='riskScore')
        plt.title("Distribution of Risk Scores by User Role")
        plt.xlabel("User Role")
        plt.ylabel("Risk Score")
        plt.tight_layout()
        plt.savefig("user_role_vs_risk_severity.png")
        plt.close()
        print('User Role vs Risk Severity plot saved as "user_role_vs_risk_severity.png"')

# -----------------------------
# Risk Advice Engine
# -----------------------------
class RiskAdviceEngine:
    def __init__(self, dp: DataPipeline, models: RiskModels):
        self.dp = dp
        self.models = models

    def aggregate_advice(self):
        """
        Combine insights from the five predictions into risk advice.
        Provides context-aware, prioritized recommendations.
        """
        print("Aggregating risk advice ...")
        advice_list = []
        df_risk = self.dp.df_risk.copy()

        # 1. Risk Prediction advice: High-risk alerts.
        high_risk = df_risk[df_risk['riskScore'] > 80]  # Threshold can be adjusted
        for idx, row in high_risk.iterrows():
            advice_list.append(
                f"High-risk alert: {row.get('riskName', 'Unnamed Risk')} (Score: {row['riskScore']}). "
                "Consider immediate review and resource allocation."
            )

        # 2. Mitigation Suggestions advice.
        df_mitig = self.dp.df_risk_mitigation.copy()
        if not df_mitig.empty:
            effective = df_mitig[df_mitig['mitigatedRiskScore'] < (df_mitig['mitigatedRiskScore'].mean())]
            for idx, row in effective.iterrows():
                advice_list.append(
                    f"Mitigation suggestion: Strategy '{row['mitigationControl']}' shows effective risk reduction. "
                    "Consider scaling this approach."
                )

        # 3. Anomaly Detection advice.
        if 'anomaly' in df_risk.columns:
            anomalies = df_risk[df_risk['anomaly'] == -1]
            for idx, row in anomalies.iterrows():
                advice_list.append(
                    f"Anomaly detected: Unusual risk score for {row.get('riskName', 'Unnamed Risk')} (Score: {row['riskScore']}). "
                    "Recommend an audit."
                )

        # 4. Risk Monitoring advice.
        df_status = self.dp.df_risk_status.copy()
        if not df_status.empty:
            monitored = df_status[df_status['isMonitored'] == True]
            for idx, row in monitored.iterrows():
                advice_list.append(
                    f"Monitoring update: Risk with ID {row['risk_id']} is under active monitoring."
                )

        # 5. User Behavior Analysis advice.
        df_user = self.dp.df_user.copy()
        if not df_user.empty:
            submission_counts = df_risk.groupby('submittedBy').size()
            for user_email, count in submission_counts.items():
                if count > 5:  # Arbitrary threshold
                    advice_list.append(
                        f"User behavior: {user_email} has submitted {count} risks recently. Consider training or process review."
                    )

        if not advice_list:
            advice_list.append("No immediate risk actions are required at this time.")

        print("Risk Advice Aggregated:")
        for advice in advice_list:
            print(" -", advice)
        return advice_list

# -----------------------------
# Main Pipeline Class
# -----------------------------
class RiskGuardPipeline:
    def __init__(self, db_url: str, organization_id: int):
        self.engine = get_db_engine(db_url)
        self.dp = DataPipeline(self.engine, organization_id)
        self.models = RiskModels(self.dp)
        self.viz = Visualizations(self.dp)
        self.advice_engine = RiskAdviceEngine(self.dp, self.models)

    def run_pipeline(self):
        print("Starting RiskGuard AI/ML Pipeline ...")
        # Step 1: Data Extraction and Cleaning
        self.dp.load_data()
        self.dp.clean_data()

        # Step 2: Train ML Models (Risk Prediction and Anomaly Detection)
        self.models.train_risk_prediction_models()
        self.models.train_anomaly_detector()
        self.models.generate_predictions()

        # Step 3: Generate Visualizations for each prediction/analysis
        self.viz.plot_risk_score_distribution()
        self.viz.plot_risk_probability_vs_impact()
        self.viz.plot_risk_trends_over_time()
        self.viz.plot_mitigation_effectiveness()
        self.viz.plot_cost_vs_effort_of_mitigation()
        self.viz.plot_mitigation_by_risk_category()
        self.viz.plot_outlier_detection()
        self.viz.plot_sudden_risk_spikes()
        self.viz.plot_risk_status_distribution()
        self.viz.plot_risk_response_activity()
        self.viz.plot_user_risk_submission_patterns()
        self.viz.plot_user_role_vs_risk_severity()

        # Step 4: Generate aggregated Risk Advice
        advice = self.advice_engine.aggregate_advice()
        print("RiskGuard AI/ML Pipeline completed.")

    def monitor_updates(self, poll_interval: int = 60):
        """
        Monitor the app for new data updates.
        In production, this might use a real streaming solution.
        """
        print("Starting monitoring for new data ...")
        while True:
            print(f"Polling for updates at {datetime.datetime.now()} ...")
            self.dp.load_data()
            self.dp.clean_data()
            self.models.train_risk_prediction_models()
            self.models.train_anomaly_detector()
            self.models.generate_predictions()
            self.advice_engine.aggregate_advice()
            time.sleep(poll_interval)

# -----------------------------
# Main Function
# -----------------------------
def main():
    # Example: Company A's organizationId = 1
    organization_id = 1
    pipeline = RiskGuardPipeline(DB_URL, organization_id)
    pipeline.run_pipeline()

    # Uncomment the line below to enable continuous monitoring:
    # pipeline.monitor_updates(poll_interval=300)  # Poll every 5 minutes

if __name__ == "__main__":
    main()

# -----------------------------
# Architecture Diagram (Text Format)
# -----------------------------
"""
Architecture Diagram: RiskGuard AI/ML Pipeline

+------------------------------------------------------------+
|                Risk Management Application                 |
|  (Users interact with the app, upload files, input risks)  |
+-------------------------------+----------------------------+
                                |
                                v
+------------------------------------------------------------+
|              Data Extraction & Ingestion Layer             |
|  - Direct DB connection using SQLAlchemy                   |
|  - Queries using quoted, capitalized table names           |
|  - Real-time & scheduled data polling                      |
+-------------------------------+----------------------------+
                                |
                                v
+------------------------------------------------------------+
|        Data Cleaning & Transformation (Pandas)             |
|  - Convert date/time formats                               |
|  - Handle missing values and type conversions              |
+-------------------------------+----------------------------+
                                |
                                v
+------------------------------------------------------------+
|            Machine Learning & Prediction Models            |
|  - Risk Prediction (Regression & Classification)           |
|  - Anomaly Detection (Isolation Forest)                    |
|  - Mitigation Suggestions & User Behavior Analysis         |
+-------------------------------+----------------------------+
                                |
                                v
+------------------------------------------------------------+
|           Visualization & Analysis Layer                   |
|  - Histograms, Scatter Plots, Line Charts, etc.             |
+-------------------------------+----------------------------+
                                |
                                v
+------------------------------------------------------------+
|         Aggregation & Risk Advice Engine                   |
|  - Combine insights from all models                        |
|  - Generate context-aware recommendations per company      |
+-------------------------------+----------------------------+
                                |
                                v
+------------------------------------------------------------+
|          Integration & Delivery Layer                    |
|  - API endpoints / direct integration with the app         |
|  - Real-time updates and dashboard refreshes               |
+------------------------------------------------------------+
"""
