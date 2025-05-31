# Kraków Tram Delay Prediction System

This project provides a complete machine learning pipeline for predicting tram delays in Kraków. It includes data enhancement with weather and temporal features, synthetic data generation, and CatBoost regression model training for accurate delay prediction.

## Features

### Data Enhancement

The system enhances tram delay data with comprehensive features for machine learning analysis:

#### Temporal Features

- `hour_of_day`: Hour of the day (0-23)
- `day_of_week`: Day of the week (0=Monday, 6=Sunday)
- `is_weekend`: Boolean indicating if it's weekend
- `month`: Month number
- `day_of_month`: Day of the month

#### Weather Features

- `temperature`: Temperature in Celsius
- `precipitation`: Precipitation in mm
- `weather_condition`: Weather condition (rain, snow, clear, cloudy, fog)
- `humidity`: Humidity percentage
- `wind_speed`: Wind speed in m/s

#### Route Features

- `route_segment`: Main route description
- `planned_travel_time_min`: Estimated travel time in minutes
- `delay_category`: Categorized delay (on_time, slight_delay, moderate_delay, significant_delay, major_delay)
- `delay_ratio`: Delay as a proportion of planned travel time
- `is_delayed`: Boolean indicating delay > 2 minutes

### Machine Learning Pipeline

- **CatBoost Regression**: Advanced gradient boosting for delay prediction
- **Cross-domain Training**: Train on synthetic data, test on real data
- **Feature Importance Analysis**: Identify key factors affecting delays
- **Comprehensive Evaluation**: RMSE, MAE, R² metrics with visualizations
- **Categorical Feature Handling**: Automatic processing of text-based features

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

Or manually install dependencies:

```bash
pip install pandas>=1.5.0 requests>=2.28.0 numpy>=1.21.0 catboost>=1.2.0
```

2. (Optional) Get a free OpenWeatherMap API key for real weather data:
   - Go to https://openweathermap.org/api
   - Sign up for a free account
   - Get your API key

## Usage

### Data Enhancement

#### Basic Usage (with mock weather data)

```bash
python3 weather_data.py
```

#### With Real Weather Data

1. Set your API key as environment variable:

```bash
export OPENWEATHER_API_KEY="your_api_key_here"
python3 weather_data.py
```

2. Or edit the script and set the API key directly:

```python
api_key = "your_api_key_here"  # Replace with your actual API key
```

### Machine Learning Training

#### Simple Training in Jupyter Notebook

```python
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load datasets
synthetic_df = pd.read_csv('synthetic_tram_data_2000.csv')
real_df = pd.read_csv('enhanced_tram_data.csv')

# Define categorical features
categorical_features = [
    'weather_condition', 'vehicle_type', 'brigade', 'route_segment',
    'direction', 'stop_name', 'delay_category'
]

# Prepare features
def prepare_features(df):
    features = [
        'hour_of_day', 'day_of_week', 'month', 'day_of_month',
        'temperature', 'precipitation', 'humidity', 'wind_speed',
        'line_number', 'planned_travel_time_min', 'is_weekend', 'is_delayed'
    ] + categorical_features

    features = [f for f in features if f in df.columns and f not in ['delay_minutes', 'delay_ratio']]

    X = df[features].copy()
    y = df['delay_minutes'].copy()

    return X, y

# Prepare training and test data
X_train, y_train = prepare_features(synthetic_df)
X_test, y_test = prepare_features(real_df)

# Ensure same features in both datasets
common_features = [col for col in X_train.columns if col in X_test.columns]
X_train = X_train[common_features]
X_test = X_test[common_features]
categorical_features = [f for f in categorical_features if f in common_features]

# Initialize and train CatBoost model
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    cat_features=categorical_features,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

# Train the model
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    plot=False
)

# Make predictions and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f} minutes")
print(f"MAE: {mae:.2f} minutes")
print(f"R² Score: {r2:.3f}")
```

## Data Files

- `synthetic_tram_data_2000.csv`: Synthetic training data (2000+ samples)
- `enhanced_tram_data.csv`: Real tram delay data with enhanced features
- `weather_data.py`: Data enhancement script
- `requirements.txt`: Python dependencies

## Model Performance

The CatBoost regression model provides:

- **RMSE**: Root Mean Square Error in minutes
- **MAE**: Mean Absolute Error in minutes
- **R² Score**: Coefficient of determination (goodness of fit)
- **Feature Importance**: Ranking of most predictive features
- **Delay Category Analysis**: Performance breakdown by delay severity

### Key Predictive Features (typical ranking):

1. Weather conditions (temperature, precipitation)
2. Time-based features (hour of day, day of week)
3. Route characteristics (planned travel time, line number)
4. Operational factors (vehicle type, direction)

## Output Structure

### Enhanced Data Example

```json
{
  "hour_of_day": 12,
  "day_of_week": 5,
  "is_weekend": true,
  "temperature": 12.4,
  "precipitation": 1.2,
  "weather_condition": "rain",
  "line_number": 10,
  "vehicle_type": "tramwaj",
  "route_segment": "Kurdwanów–Pleszów",
  "planned_travel_time_min": 25,
  "delay_minutes": 23.0,
  "delay_category": "major_delay",
  "delay_ratio": 0.92,
  "is_delayed": true
}
```

## Supported Tram Lines

The system includes route information for all major Kraków tram lines:

- **Main Lines**: 1, 3, 4, 5, 8, 9, 10, 11, 14, 18, 20, 21, 22, 24
- **Express Lines**: 50, 52
- **Additional Lines**: 72, 76, 77, 78

## Project Structure

```
ml-services/
├── requirements.txt              # Python dependencies
├── weather_data.py              # Data enhancement script
├── synthetic_tram_data_2000.csv # Synthetic training data
├── enhanced_tram_data.csv       # Real enhanced data
└── README.md                    # This file
```

## Machine Learning Workflow

1. **Data Enhancement**: Add weather and temporal features
2. **Feature Engineering**: Prepare categorical and numerical features
3. **Model Training**: Train CatBoost on synthetic data
4. **Validation**: Test on real tram delay data
5. **Evaluation**: Comprehensive performance analysis
6. **Feature Analysis**: Identify key delay predictors

## Customization

### Data Enhancement

- Add more route information
- Include additional weather parameters
- Change delay categorization thresholds
- Add custom temporal features (rush hour, seasons)

### Model Training

- Adjust CatBoost hyperparameters
- Add feature engineering steps
- Implement ensemble methods
- Include cross-validation strategies

## Notes

- **Weather Data**: Without an API key, mock weather data is used (12.4°C, rain, 1.2mm precipitation)
- **Data Source**: Designed to work with CSV files from czynaczas.pl
- **Route Information**: Based on typical tram routes in Kraków
- **Production Use**: Consider historical weather APIs for accurate timestamps
- **Model Generalization**: Synthetic data training allows robust real-world performance

## Performance Tips

- **Feature Selection**: Focus on weather and temporal features for best results
- **Data Quality**: Ensure consistent categorical feature encoding
- **Model Tuning**: Adjust iterations and learning_rate based on dataset size
- **Validation**: Use cross-domain validation (synthetic→real) for robust evaluation
