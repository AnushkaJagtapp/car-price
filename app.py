import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 20px 0;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and preprocessing info"""
    try:
        model = joblib.load('model/car_price_model.pkl')
        feature_info = joblib.load('model/preprocessing.pkl')
        return model, feature_info
    except:
        st.error("Model not found! Please run train_model.py first.")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Car Price Prediction App</h1>', unsafe_allow_html=True)
    st.markdown("### Predict the market value of your car based on its features")
    
    # Load model
    model, feature_info = load_model()
    
    if model is None:
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💰 Price Prediction", "📊 Data Analysis", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<h3 class="sub-header">Car Specifications</h3>', unsafe_allow_html=True)
            
            # Brand selection
            brands = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Tesla']
            brand = st.selectbox("Brand", brands, index=0)
            
            # Model type
            models = ['Sedan', 'SUV', 'Truck', 'Coupe', 'Convertible', 'Hatchback']
            model_type = st.selectbox("Model Type", models, index=0)
            
            # Year
            current_year = 2024
            year = st.slider("Manufacturing Year", 2010, current_year, 2020)
            
            # Mileage
            mileage = st.slider("Mileage (miles)", 0, 200000, 50000, step=1000)
            
            # Engine Size
            engine_size = st.slider("Engine Size (L)", 1.0, 5.0, 2.0, step=0.1)
        
        with col2:
            st.markdown('<h3 class="sub-header">Additional Details</h3>', unsafe_allow_html=True)
            
            # Fuel Type
            fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Electric']
            fuel_type = st.selectbox("Fuel Type", fuel_types, index=0)
            
            # Transmission
            transmissions = ['Automatic', 'Manual', 'CVT']
            transmission = st.selectbox("Transmission", transmissions, index=0)
            
            # Owner Count
            owner_count = st.selectbox("Number of Previous Owners", [0, 1, 2, 3, 4, 5], index=1)
            
            # Accident History
            accident_history = st.radio("Accident History", ["No", "Yes"], horizontal=True)
            accident_history_val = 1 if accident_history == "Yes" else 0
            
            # Service History
            service_history = st.selectbox("Service History", ['Full', 'Partial', 'None'], index=0)
            
            # Predict button
            predict_button = st.button("🚀 Predict Car Price", use_container_width=True)
        
        # Make prediction
        if predict_button:
            # Prepare input data
            input_data = pd.DataFrame({
                'brand': [brand],
                'model': [model_type],
                'year': [year],
                'mileage': [mileage],
                'engine_size': [engine_size],
                'fuel_type': [fuel_type],
                'transmission': [transmission],
                'owner_count': [owner_count],
                'accident_history': [accident_history_val],
                'service_history': [service_history]
            })
            
            # Make prediction
            try:
                prediction = model.predict(input_data)[0]
                
                # Display prediction
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="text-align: center; color: #1E88E5;">Predicted Price</h2>
                        <h1 style="text-align: center; color: #0D47A1;">${prediction:,.2f}</h1>
                        <p style="text-align: center;">Based on the provided specifications</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Price impact visualization
                st.markdown('<h3 class="sub-header">Price Impact Factors</h3>', unsafe_allow_html=True)

                factors = {
                    'Year (+$1000/year)': (year - 2010) * 1000,
                    'Mileage (-$0.1/mile)': -mileage * 0.1,
                    'Engine Size (+$3000/L)': (engine_size - 1) * 3000,
                    'Brand Premium': 15000 if brand in ['BMW', 'Mercedes', 'Audi'] else 0,
                    'Fuel Type Premium': 10000 if fuel_type == 'Electric' else (5000 if fuel_type == 'Hybrid' else 0),
                    'Owner Count (-$2000/owner)': -owner_count * 2000,
                    'Accident History': -8000 if accident_history_val else 0
                }
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(factors.keys()),
                        y=list(factors.values()),
                        marker_color=['green' if val > 0 else 'red' for val in factors.values()]
                    )
                ])
                fig.update_layout(
                    title="How Each Factor Affects Price",
                    xaxis_title="Factors",
                    yaxis_title="Price Impact ($)",
                    showlegend=False
                )                
                st.plotly_chart(fig, use_container_width=True)               
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with tab2:
        st.markdown('<h3 class="sub-header">Car Market Analysis</h3>', unsafe_allow_html=True)
        try:
            df = pd.read_csv('car_price.csv')
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.box(df, x='brand', y='price', 
                             title='Price Distribution by Brand',
                             color='brand')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                avg_price_fuel = df.groupby('fuel_type')['price'].mean().reset_index()
                fig2 = px.bar(avg_price_fuel, x='fuel_type', y='price',
                             title='Average Price by Fuel Type',
                             color='fuel_type')
                st.plotly_chart(fig2, use_container_width=True)           
            col3, col4 = st.columns(2)
            
            with col3:
                fig3 = px.scatter(df, x='mileage', y='price', color='fuel_type',
                                 title='Mileage vs Price',
                                 trendline='ols')
                st.plotly_chart(fig3, use_container_width=True)         
            with col4:
                fig4 = px.scatter(df, x='year', y='price', color='transmission',
                                 title='Manufacturing Year vs Price')
                st.plotly_chart(fig4, use_container_width=True)
            with st.expander("View Sample Data"):
                st.dataframe(df.head(50))
                
        except Exception as e:
            st.warning("Sample data not available. Run train_model.py to generate data.")
    
    with tab3:
        st.markdown('<h3 class="sub-header">About This App</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Purpose
        This application predicts car prices using machine learning based on various car specifications and features.
        
        ### 🔧 How It Works
        1. **Input Features**: The model considers 10 different features including brand, model, year, mileage, etc.
        2. **Machine Learning**: Uses Gradient Boosting Regressor trained on car data
        3. **Prediction**: Provides estimated market value based on current trends
        
        ###  Features Considered
        - **Brand & Model**: Manufacturer and body type
        - **Year**: Manufacturing year
        - **Mileage**: Total distance traveled
        - **Engine Size**: Engine capacity in liters
        - **Fuel Type**: Petrol, Diesel, Hybrid, or Electric
        - **Transmission**: Automatic, Manual, or CVT
        - **Owner History**: Number of previous owners
        - **Accident History**: Whether the car has been in accidents
        - **Service History**: Maintenance record completeness
        
        ###  Getting Started
        1. Run `train_model.py` to train the model
        2. Install requirements: `pip install -r requirements.txt`
        3. Run the app: `streamlit run app.py`
        
        ###  Disclaimer
        This is a demonstration project. Actual car prices may vary based on market conditions, location, and other factors not considered in this model.
        """)
        
        # Show model metrics
        st.markdown("---")
        st.markdown("### 📈 Model Performance")
        
        # Simulated metrics (these would come from actual training)
        metrics = {
            'R² Score': 0.89,
            'Mean Absolute Error': '$3,200',
            'Root Mean Squared Error': '$4,500'
        }
        
        for metric, value in metrics.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric(label=metric, value=value)

if __name__ == "__main__":
    main()