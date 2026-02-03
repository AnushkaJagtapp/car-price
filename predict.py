import joblib
import pandas as pd
import numpy as np

def predict_single_car(model_path='model/car_price_model.pkl', input_data=None):
    """
    Predict car price for a single car
    """
    # Load the trained model
    model = joblib.load(model_path)
    
    # If no input data provided, use sample data
    if input_data is None:
        input_data = {
            'brand': 'Toyota',
            'model': 'Sedan',
            'year': 2020,
            'mileage': 50000,
            'engine_size': 2.0,
            'fuel_type': 'Petrol',
            'transmission': 'Automatic',
            'owner_count': 1,
            'accident_history': 0,
            'service_history': 'Full'
        }
    
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(df)[0]
    
    print("=" * 50)
    print("CAR PRICE PREDICTION")
    print("=" * 50)
    print("\nInput Features:")
    for key, value in input_data.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print(f"\n💰 Predicted Price: ${prediction:,.2f}")
    print("=" * 50)
    
    return prediction

def predict_multiple_cars(model_path='model/car_price_model.pkl', csv_path=None):
    """
    Predict prices for multiple cars from CSV file
    """
    model = joblib.load(model_path)
    
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        # Create sample cars
        data = {
            'brand': ['Toyota', 'BMW', 'Tesla'],
            'model': ['Sedan', 'SUV', 'Sedan'],
            'year': [2018, 2021, 2022],
            'mileage': [60000, 25000, 15000],
            'engine_size': [1.8, 3.0, 0.0],  # Tesla has electric motor
            'fuel_type': ['Petrol', 'Diesel', 'Electric'],
            'transmission': ['Automatic', 'Automatic', 'Automatic'],
            'owner_count': [2, 1, 1],
            'accident_history': [0, 0, 0],
            'service_history': ['Full', 'Full', 'Full']
        }
        df = pd.DataFrame(data)
    
    predictions = model.predict(df)
    df['predicted_price'] = predictions
    
    print("\n" + "=" * 50)
    print("BATCH PREDICTION RESULTS")
    print("=" * 50)
    print(df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    # Predict single car
    print("Single Car Prediction:")
    predict_single_car()
    
    print("\n\n" + "=" * 70 + "\n")
    
    # Predict multiple cars
    print("Batch Prediction (3 cars):")
    predict_multiple_cars()