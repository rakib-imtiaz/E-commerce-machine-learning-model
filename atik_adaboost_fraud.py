# Atik's contribution - AdaBoost Model for Fraud Detection

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib
import os

class AdaBoostFraudDetector:
    def __init__(self):
        # Load the model and label encoder
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        try:
            self.model = joblib.load('models/adaboost_model.pkl')
            self.label_encoders = joblib.load('models/label_encoders.pkl')
        except:
            # If models don't exist, train them
            from adaboost_e_commerce import train_model
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
                except ValueError:
                    # If category not seen during training, use the 'unknown' category
                    df[col] = self.label_encoders[col].transform(['unknown'])[0]
        
        # Convert all features to float32
        for col in required_features:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
        
        # Select only required features in correct order
        return df[required_features]
    
    def detect_fraud(self, transaction_data):
        """Detect fraud using AdaBoost model"""
        processed_data = self._preprocess_data(transaction_data)
        fraud_probability = self.model.predict_proba(processed_data)[:, 1]
        
        return {
            'fraud_probability': fraud_probability[0],
            'model_name': 'AdaBoost',
            'confidence_score': self._calculate_confidence(fraud_probability[0])
        }
    
    def _calculate_confidence(self, probability):
        """Calculate confidence score"""
        return abs(probability - 0.5) * 2  # Scale to 0-1
    
    def show_model_performance(self):
        """Display AdaBoost model performance metrics"""
        # Load test data
        from adaboost_e_commerce import load_and_prepare_data
        _, X_test, _, y_test, _ = load_and_prepare_data()
        
        # Calculate predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred)
        }
        
        return metrics