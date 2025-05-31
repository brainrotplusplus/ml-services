import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
from generate_synthetic_data import SyntheticTramDataGenerator


def prepare_features(df):
    """Prepare features for training."""
    feature_columns = [
        'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
        'temperature', 'precipitation', 'humidity', 'wind_speed',
        'line_number', 'planned_travel_time_min',
        'weather_condition', 'vehicle_type', 'route_segment', 'direction'
    ]
    
    X = df[feature_columns].copy()
    y = df['delay_minutes']
    
    # Identify categorical features
    categorical_features = ['weather_condition', 'vehicle_type', 'route_segment', 'direction']
    
    return X, y, categorical_features


def evaluate_model(model, X_test, y_test, dataset_name="Test"):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{dataset_name} Results:")
    print(f"MAE: {mae:.3f} minutes")
    print(f"RMSE: {rmse:.3f} minutes")
    print(f"R² Score: {r2:.3f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def print_feature_importance(model, feature_names, top_n=10):
    """Print top feature importances."""
    feature_importance = model.get_feature_importance()
    
    print(f"\nTop {top_n} Most Important Features:")
    sorted_features = sorted(zip(feature_names, feature_importance), 
                           key=lambda x: x[1], reverse=True)
    
    for i, (feat, imp) in enumerate(sorted_features[:top_n]):
        print(f"{i+1:2d}. {feat:25s}: {imp:6.2f}")


def main():
    parser = argparse.ArgumentParser(description='Train CatBoost model on tram delay data')
    parser.add_argument('--real-data', type=str, default='enhanced_tram_data.csv',
                       help='Path to real tram data CSV file')
    parser.add_argument('--synthetic-samples', type=int, default=500,
                       help='Number of synthetic samples to generate for testing')
    parser.add_argument('--model-output', type=str, default='tram_delay_model.cbm',
                       help='Output path for trained model')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of CatBoost iterations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("Loading real tram data...")
    try:
        df_real = pd.read_csv(args.real_data)
        print(f"Real data shape: {df_real.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find real data file '{args.real_data}'")
        return
    
    # Generate synthetic test data
    print(f"\nGenerating {args.synthetic_samples} synthetic test samples...")
    generator = SyntheticTramDataGenerator(random_seed=args.seed)
    df_synthetic = generator.generate_dataset(n_samples=args.synthetic_samples)
    
    # Prepare features
    print("\nPreparing features for training...")
    X_real, y_real, cat_features = prepare_features(df_real)
    X_synthetic, y_synthetic, _ = prepare_features(df_synthetic)
    
    # Split real data into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_real, y_real, test_size=0.2, random_state=args.seed
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Synthetic test set shape: {X_synthetic.shape}")
    
    # Train CatBoost model
    print(f"\nTraining CatBoost model with {args.iterations} iterations...")
    model = CatBoostRegressor(
        iterations=args.iterations,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,
        random_seed=args.seed,
        verbose=100
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        plot=False
    )
    
    # Evaluate on validation set (real data)
    val_results = evaluate_model(model, X_val, y_val, "Validation (Real Data)")
    
    # Test on synthetic data
    synthetic_results = evaluate_model(model, X_synthetic, y_synthetic, "Synthetic Test Data")
    
    # Feature importance
    print_feature_importance(model, X_train.columns)
    
    # Compare delay distributions
    print(f"\nDelay Distribution Comparison:")
    print(f"Real data - Mean: {y_real.mean():.2f}, Std: {y_real.std():.2f}")
    print(f"Synthetic data - Mean: {y_synthetic.mean():.2f}, Std: {y_synthetic.std():.2f}")
    
    # Save model
    model.save_model(args.model_output)
    print(f"\nModel saved to '{args.model_output}'")
    
    # Save synthetic test data
    synthetic_output = 'synthetic_test_data.csv'
    df_synthetic.to_csv(synthetic_output, index=False)
    print(f"Synthetic test data saved to '{synthetic_output}'")
    
    # Summary
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"Real data samples: {len(df_real)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Synthetic test samples: {len(df_synthetic)}")
    print(f"\nValidation Performance:")
    print(f"  MAE: {val_results['mae']:.3f} min, RMSE: {val_results['rmse']:.3f} min, R²: {val_results['r2']:.3f}")
    print(f"Synthetic Test Performance:")
    print(f"  MAE: {synthetic_results['mae']:.3f} min, RMSE: {synthetic_results['rmse']:.3f} min, R²: {synthetic_results['r2']:.3f}")


if __name__ == "__main__":
    main() 