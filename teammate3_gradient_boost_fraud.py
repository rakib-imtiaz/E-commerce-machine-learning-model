# Group Mate 3's contribution - Gradient Boosting Model for Fraud Detection

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

class GradientBoostFraudDetector:
    def __init__(self):
        # Load the model and label encoders
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        try:
            self.model = joblib.load('models/gradient_boost_model.pkl')
            self.label_encoders = joblib.load('models/gb_label_encoders.pkl')
        except:
            # If models don't exist, train them
            from gradient_boost_e_commerce import train_model
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
        
        return {
            'fraud_probability': fraud_probability,
            'model_name': 'Gradient Boosting',
            'prediction_confidence': abs(fraud_probability - 0.5) * 2
        }
    
    def show_model_performance(self):
        """Display Gradient Boosting model performance metrics"""
        # Load test data
        from gradient_boost_e_commerce import load_and_prepare_data
        _, X_test, _, y_test, _ = load_and_prepare_data()
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Gradient Boosting Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Convert plot to image for Streamlit
        fig = plt.gcf()
        plt.close()
        
        return fig 