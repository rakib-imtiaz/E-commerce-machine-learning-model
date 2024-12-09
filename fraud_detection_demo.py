import streamlit as st
from atik_adaboost_fraud import AdaBoostFraudDetector
from teammate2_random_forest_fraud import RandomForestFraudDetector
from teammate3_gradient_boost_fraud import GradientBoostFraudDetector
import plotly.express as px
import pandas as pd

@st.cache_data
def load_stock_data():
    try:
        df = pd.read_csv('E-Commerce Data.csv', encoding='latin-1')
    except FileNotFoundError:
        df = pd.read_csv('datasets/E-Commerce Data.csv', encoding='latin-1')
    
    # Get unique stock codes and their descriptions
    stock_data = df[['StockCode', 'Description']].drop_duplicates()
    stock_codes = stock_data['StockCode'].astype(str) + ' - ' + stock_data['Description'].astype(str)
    return sorted(stock_codes.unique())

class FraudDetectionDemo:
    def __init__(self):
        # Initialize models only when needed
        self._ada_detector = None
        self._rf_detector = None
        self._gb_detector = None
    
    @property
    def ada_detector(self):
        if self._ada_detector is None:
            self._ada_detector = AdaBoostFraudDetector()
        return self._ada_detector
    
    @property
    def rf_detector(self):
        if self._rf_detector is None:
            self._rf_detector = RandomForestFraudDetector()
        return self._rf_detector
    
    @property
    def gb_detector(self):
        if self._gb_detector is None:
            self._gb_detector = GradientBoostFraudDetector()
        return self._gb_detector

    def get_transaction_input(self):
        stock_codes = load_stock_data()

        col1, col2 = st.columns(2)
        with col1:
            quantity = st.number_input("Quantity", min_value=1)
            unit_price = st.number_input("Unit Price", min_value=0.0)
            country = st.selectbox("Country", ["UK", "Germany", "France", "Others"])
        
        with col2:
            stock_code_full = st.selectbox("Stock Code", stock_codes)
            stock_code = stock_code_full.split(' - ')[0]
            
            customer_id = st.text_input("Customer ID")
            transaction_date = st.date_input("Transaction Date")
            transaction_time = st.time_input("Transaction Time")
            
            transaction_datetime = pd.Timestamp.combine(transaction_date, transaction_time)
            
        return {
            'Quantity': quantity,
            'UnitPrice': unit_price,
            'Country': country,
            'StockCode': stock_code,
            'CustomerID': customer_id,
            'TransactionTime': transaction_datetime
        }

    def run(self):
        st.title("Fraud Detection System - Model Comparison")
        
        model_choice = st.sidebar.selectbox(
            "Select Model to Demonstrate",
            ["AdaBoost (Atik)", "Random Forest (Teammate 2)", 
             "Gradient Boosting (Teammate 3)", "Compare All"]
        )
        
        transaction_data = self.get_transaction_input()
        
        if st.button("Analyze Transaction"):
            with st.spinner('Analyzing transaction...'):
                if model_choice == "AdaBoost (Atik)":
                    self.show_adaboost_analysis(transaction_data)
                elif model_choice == "Random Forest (Teammate 2)":
                    self.show_random_forest_analysis(transaction_data)
                elif model_choice == "Gradient Boosting (Teammate 3)":
                    self.show_gradient_boost_analysis(transaction_data)
                else:
                    self.show_model_comparison(transaction_data)
    
    def show_adaboost_analysis(self, transaction_data):
        st.subheader("AdaBoost Model Analysis")
        
        # Get predictions
        result = self.ada_detector.detect_fraud(transaction_data)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Fraud Probability",
                value=f"{result['fraud_probability']:.2%}"
            )
        with col2:
            st.metric(
                label="Confidence Score",
                value=f"{result['confidence_score']:.2f}"
            )
        
        # Show model performance
        st.subheader("Model Performance Metrics")
        metrics = self.ada_detector.show_model_performance()
        metrics_df = pd.DataFrame([metrics])
        st.dataframe(metrics_df)
    
    def show_random_forest_analysis(self, transaction_data):
        st.subheader("Random Forest Model Analysis")
        
        # Get predictions
        result = self.rf_detector.detect_fraud(transaction_data)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Fraud Probability",
                value=f"{result['fraud_probability']:.2%}"
            )
        
        # Show feature importance
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame(
            result['feature_importance'].items(),
            columns=['Feature', 'Importance']
        )
        fig = px.bar(importance_df, x='Importance', y='Feature')
        st.plotly_chart(fig)
    
    def show_gradient_boost_analysis(self, transaction_data):
        st.subheader("Gradient Boosting Model Analysis")
        
        # Get predictions
        result = self.gb_detector.detect_fraud(transaction_data)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Fraud Probability",
                value=f"{result['fraud_probability']:.2%}"
            )
        with col2:
            st.metric(
                label="Prediction Confidence",
                value=f"{result['prediction_confidence']:.2f}"
            )
        
        # Show confusion matrix
        st.subheader("Model Performance")
        fig = self.gb_detector.show_model_performance()
        st.pyplot(fig)
    
    def show_model_comparison(self, transaction_data):
        st.subheader("Model Comparison")
        
        # Get predictions from all models
        ada_result = self.ada_detector.detect_fraud(transaction_data)
        rf_result = self.rf_detector.detect_fraud(transaction_data)
        gb_result = self.gb_detector.detect_fraud(transaction_data)
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Model': ['AdaBoost', 'Random Forest', 'Gradient Boosting'],
            'Fraud Probability': [
                ada_result['fraud_probability'],
                rf_result['fraud_probability'],
                gb_result['fraud_probability']
            ]
        })
        
        # Display comparison
        fig = px.bar(
            comparison,
            x='Model',
            y='Fraud Probability',
            title='Fraud Probability by Model'
        )
        st.plotly_chart(fig)

if __name__ == "__main__":
    demo = FraudDetectionDemo()
    demo.run() 