# Group Mate 2's contribution - Random Forest Model for Fraud Detection

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import plotly.express as px
import joblib
import os

class RandomForestFraudDetector:
    def __init__(self):
        # Load the model and label encoders
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        try:
            self.model = joblib.load('models/random_forest_model.pkl')
            self.label_encoders = joblib.load('models/rf_label_encoders.pkl')
        except:
            # If models don't exist, train them
            from e_commerce_random_forest import train_model
            self.model, self.label_encoders, _ = train_model()
    
    def _preprocess_data(self, transaction_data):
        """Preprocess transaction data for prediction"""
        df = pd.DataFrame([transaction_data])
        
        # Convert transaction time to datetime and extract month
        if 'TransactionTime' in df.columns:
            df['InvoiceMonth'] = pd.to_datetime(df['TransactionTime']).dt.month
            df = df.drop(['TransactionTime'], axis=1)
        
        # Ensure we have all required features
        required_features = ['StockCode', 'Country', 'UnitPrice', 'InvoiceMonth']
        
        # Encode categorical variables
        categorical_cols = ['StockCode', 'Country']
        for col in categorical_cols:
            if col in df.columns:
                val = str(df[col].iloc[0])
                try:
                    df[col] = self.label_encoders[col].transform([val])[0]
                except (ValueError, KeyError):
                    # If category not seen during training, use the 'unknown' category
                    df[col] = self.label_encoders[col].transform(['unknown'])[0]
        
        # Convert all features to float32
        for col in required_features:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
        
        # Select only required features in correct order
        return df[required_features]
    
    def detect_fraud(self, transaction_data):
        """Detect fraud in a transaction"""
        # Preprocess the data
        processed_data = self._preprocess_data(transaction_data)
        
        # Get prediction probability
        fraud_probability = self.model.predict_proba(processed_data)[0][1]
        
        # Get feature importance
        feature_importance = dict(zip(
            processed_data.columns,
            self.model.feature_importances_
        ))
        
        return {
            'fraud_probability': fraud_probability,
            'model_name': 'Random Forest',
            'feature_importance': feature_importance
        }
    
    def show_model_performance(self):
        """Display Random Forest model performance metrics"""
        # Load test data
        from e_commerce_random_forest import load_and_prepare_data
        _, X_test, _, y_test, _ = load_and_prepare_data()
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Create ROC curve plot
        fig = px.line(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC = {roc_auc:.2f})',
            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        return fig 