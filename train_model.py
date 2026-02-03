import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create synthetic car data if no dataset exists
def create_sample_data():
    """Generate sample car data for training"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'brand': np.random.choice(['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Tesla'], n_samples),
        'model': np.random.choice(['Sedan', 'SUV', 'Truck', 'Coupe', 'Convertible', 'Hatchback'], n_samples),
        'year': np.random.randint(2010, 2024, n_samples),
        'mileage': np.random.randint(5000, 150000, n_samples),
        'engine_size': np.round(np.random.uniform(1.0, 5.0, n_samples), 1),
        'fuel_type': np.random.choice(['Petrol', 'Diesel', 'Hybrid', 'Electric'], n_samples),
        'transmission': np.random.choice(['Automatic', 'Manual', 'CVT'], n_samples),
        'owner_count': np.random.randint(0, 5, n_samples),
        'accident_history': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'service_history': np.random.choice(['Full', 'Partial', 'None'], n_samples),
    }
    
    # Calculate price based on features
    base_price = 20000
    data['price'] = (
        base_price +
        (data['year'] - 2010) * 1000 +  # newer cars cost more
        -data['mileage'] * 0.1 +        # higher mileage reduces price
        data['engine_size'] * 3000 +    # larger engine increases price
        np.where(data['fuel_type'] == 'Electric', 10000, 
                np.where(data['fuel_type'] == 'Hybrid', 5000, 0)) +
        np.where(data['brand'].isin(['BMW', 'Mercedes', 'Audi']), 15000, 0) +
        -data['owner_count'] * 2000 +   # more owners reduces price
        -data['accident_history'] * 8000 +  # accident reduces price
        np.random.normal(0, 5000, n_samples)  # random noise
    )
    
    # Ensure price is positive
    data['price'] = np.maximum(data['price'], 5000)
    
    return pd.DataFrame(data)

def train_model():
    """Train and save the car price prediction model"""
    print("Loading/Creating dataset...")
    
    try:
        # Try to load existing dataset
        df = pd.read_csv('car_price.csv')
        print(f"Loaded dataset with {len(df)} records")
    except:
        # Create sample data if no dataset exists
        print("No dataset found. Creating sample data...")
        df = create_sample_data()
        df.to_csv('car_price.csv', index=False)
        print(f"Created sample dataset with {len(df)} records")
    
    print("\nDataset Preview:")
    print(df.head())
    
    # Prepare features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Define categorical and numerical features
    categorical_features = ['brand', 'model', 'fuel_type', 'transmission', 'service_history']
    numerical_features = ['year', 'mileage', 'engine_size', 'owner_count', 'accident_history']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
    
    # Create and train the model
    print("\nTraining model...")
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"MAE: ${mae:,.2f}")
    print(f"MSE: ${mse:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save the model and preprocessing information
    print("\nSaving model...")
    joblib.dump(model, 'model/car_price_model.pkl')
    
    # Save feature names for the app
    feature_info = {
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'all_features': list(X.columns)
    }
    joblib.dump(feature_info, 'model/preprocessing.pkl')
    
    print("Model training complete! Files saved in 'model/' directory")
    
    return model, feature_info

if __name__ == "__main__":
    train_model()