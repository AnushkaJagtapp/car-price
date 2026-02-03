import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(filepath='car_price.csv'):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df = df.dropna()
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR    
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]   
    return df

def evaluate_model(y_true, y_pred, model_name="Model"):
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }   
    print(f"\n{'='*40}")
    print(f"Performance Metrics for {model_name}")
    print(f"{'='*40}")
    for metric, value in metrics.items():
        if metric != 'R2':
            print(f"{metric}: ${value:,.2f}")
        else:
            print(f"{metric}: {value:.4f}")    
    return metrics

def plot_feature_importance(model, feature_names, top_n=10):
    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        importances = model.named_steps['regressor'].feature_importances_
        preprocessor = model.named_steps['preprocessor']
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[-top_n:]
        
        plt.title(f'Top {top_n} Feature Importances')
        plt.barh(range(top_n), importances[indices[-top_n:]])
        plt.yticks(range(top_n), [feature_names[i] for i in indices[-top_n:]])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.show()

def generate_sample_car():
    sample_car = {
        'brand': np.random.choice(['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes']),
        'model': np.random.choice(['Sedan', 'SUV', 'Truck']),
        'year': np.random.randint(2015, 2024),
        'mileage': np.random.randint(10000, 100000),
        'engine_size': round(np.random.uniform(1.5, 3.5), 1),
        'fuel_type': np.random.choice(['Petrol', 'Diesel', 'Hybrid']),
        'transmission': np.random.choice(['Automatic', 'Manual']),
        'owner_count': np.random.randint(0, 4),
        'accident_history': np.random.choice([0, 1]),
        'service_history': np.random.choice(['Full', 'Partial'])
    }   
    return sample_car

def save_predictions(predictions, filename='predictions.csv'):
    pd.DataFrame(predictions).to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")