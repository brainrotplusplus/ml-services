# Kraków Tram Delay Data Enhancement

This project enhances tram delay data from Kraków with weather information and temporal features for machine learning analysis.

## Features

The script adds the following features to your tram delay data:

### Temporal Features

- `hour_of_day`: Hour of the day (0-23)
- `day_of_week`: Day of the week (0=Monday, 6=Sunday)
- `is_weekend`: Boolean indicating if it's weekend
- `month`: Month number
- `day_of_month`: Day of the month

### Weather Features

- `temperature`: Temperature in Celsius
- `precipitation`: Precipitation in mm
- `weather_condition`: Weather condition (rain, snow, clear, etc.)
- `humidity`: Humidity percentage
- `wind_speed`: Wind speed in m/s

### Route Features

- `route_segment`: Main route description
- `planned_travel_time_min`: Estimated travel time in minutes
- `delay_category`: Categorized delay (on_time, slight_delay, etc.)
- `delay_ratio`: Delay as a proportion of planned travel time
- `is_delayed`: Boolean indicating delay > 2 minutes

## Installation

1. Install required packages:

```bash
sudo apt update
sudo apt install -y python3-pandas python3-requests
```

2. (Optional) Get a free OpenWeatherMap API key:
   - Go to https://openweathermap.org/api
   - Sign up for a free account
   - Get your API key

## Usage

### Basic Usage (with mock weather data)

```bash
python3 weather_data.py
```

### With Real Weather Data

1. Set your API key as environment variable:

```bash
export OPENWEATHER_API_KEY="your_api_key_here"
python3 weather_data.py
```

2. Or edit the script and set the API key directly:

```python
api_key = "your_api_key_here"  # Replace with your actual API key
```

## Output

The script generates:

- `enhanced_tram_data.csv`: Enhanced dataset with all features
- Console output with data summary and sample records
- Example JSON record showing the data structure

## Example Output Structure

```json
{
  "hour_of_day": 12,
  "day_of_week": 5,
  "is_weekend": true,
  "temperature": 12.4,
  "precipitation": 1.2,
  "weather_condition": "rain",
  "line_number": "10",
  "vehicle_type": "tramwaj",
  "route_segment": "Kurdwanów–Pleszów",
  "planned_travel_time_min": 25,
  "delay_minutes": 23.0
}
```

## Supported Tram Lines

The script includes route information for all major Kraków tram lines:

- Lines 1, 3, 4, 5, 8, 9, 10, 11, 14, 18, 20, 21, 22, 24, 50, 52, 72, 76, 77, 78

## Notes

- Without an API key, the script uses mock weather data (12.4°C, rain, 1.2mm precipitation)
- The script is designed to work with CSV files from czynaczas.pl
- Route segments and travel times are based on typical tram routes in Kraków
- For production use, consider using historical weather data APIs for more accurate timestamps

## Files

- `weather_data.py`: Main enhancement script
- `requirements.txt`: Python dependencies
- `enhanced_tram_data.csv`: Output file with enhanced data
- Input: `Kraków - Opóźnienia pojazdów...csv`: Your tram delay data

## Customization

You can modify the script to:

- Add more route information
- Include additional weather parameters
- Change delay categorization thresholds
- Add custom temporal features (rush hour, seasons, etc.)
