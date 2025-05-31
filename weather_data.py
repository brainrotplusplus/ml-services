import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import os
from typing import Dict, Any
import time

class KrakowTramDataEnhancer:
    def __init__(self, api_key: str = None):
        """
        Initialize the data enhancer with weather API key.
        Get your free API key from: https://openweathermap.org/api
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_weather_url = "http://api.openweathermap.org/data/2.5/weather"
        self.historical_weather_url = "http://api.openweathermap.org/data/2.5/onecall/timemachine"
        
        # Kraków coordinates
        self.lat = 50.0647
        self.lon = 19.9450
        
        # Route segments mapping (simplified - in real scenario this would be more comprehensive)
        self.route_segments = {
            "1": {"main_route": "Salwator–Jarzębiny", "avg_travel_time": 45},
            "3": {"main_route": "Nowy Bieżanów–Krowodrza", "avg_travel_time": 35},
            "4": {"main_route": "Bronowice–Nowa Huta", "avg_travel_time": 40},
            "5": {"main_route": "Jarzębiny–Krowodrza", "avg_travel_time": 30},
            "8": {"main_route": "Bronowice–Borek Fałęcki", "avg_travel_time": 35},
            "9": {"main_route": "Rondo Hipokratesa–Nowy Bieżanów", "avg_travel_time": 40},
            "10": {"main_route": "Kurdwanów–Pleszów", "avg_travel_time": 25},
            "11": {"main_route": "Czerwone Maki–Mały Płaszów", "avg_travel_time": 30},
            "14": {"main_route": "Bronowice–Miśnieńska", "avg_travel_time": 25},
            "18": {"main_route": "Górka Narodowa–Czerwone Maki", "avg_travel_time": 35},
            "20": {"main_route": "Mały Płaszów–Cichy Kącik", "avg_travel_time": 20},
            "21": {"main_route": "Os.Piastów–Pleszów", "avg_travel_time": 30},
            "22": {"main_route": "Borek Fałęcki–Kopiec Wandy", "avg_travel_time": 40},
            "24": {"main_route": "Bronowice–Kurdwanów", "avg_travel_time": 35},
            "50": {"main_route": "Borek Fałęcki–Górka Narodowa", "avg_travel_time": 45},
            "52": {"main_route": "Czerwone Maki–Os.Piastów", "avg_travel_time": 30},
            "72": {"main_route": "Rondo Grunwaldzkie–Czerwone Maki", "avg_travel_time": 25},
            "76": {"main_route": "Mały Płaszów–Cmentarz Rakowicki", "avg_travel_time": 20},
            "77": {"main_route": "Nowy Bieżanów–Dworzec Towarowy", "avg_travel_time": 25},
            "78": {"main_route": "Salwator–Krowodrza Górka", "avg_travel_time": 30}
        }

    def get_current_weather(self) -> Dict[str, Any]:
        """Get current weather data for Kraków"""
        if not self.api_key:
            print("Warning: No API key provided. Using mock weather data.")
            return {
                "temperature": 12.4,
                "precipitation": 1.2,
                "weather_condition": "rain",
                "humidity": 75,
                "wind_speed": 3.2
            }
        
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_weather_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract precipitation (rain or snow)
            precipitation = 0.0
            if 'rain' in data and '1h' in data['rain']:
                precipitation = data['rain']['1h']
            elif 'snow' in data and '1h' in data['snow']:
                precipitation = data['snow']['1h']
            
            return {
                "temperature": data['main']['temp'],
                "precipitation": precipitation,
                "weather_condition": data['weather'][0]['main'].lower(),
                "humidity": data['main']['humidity'],
                "wind_speed": data['wind']['speed']
            }
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            # Return mock data as fallback
            return {
                "temperature": 12.4,
                "precipitation": 1.2,
                "weather_condition": "rain",
                "humidity": 75,
                "wind_speed": 3.2
            }

    def add_temporal_features(self, timestamp: datetime = None) -> Dict[str, Any]:
        """Add temporal features based on timestamp"""
        if timestamp is None:
            timestamp = datetime.now()
        
        return {
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.weekday(),  # 0=Monday, 6=Sunday
            "is_weekend": timestamp.weekday() >= 5,
            "month": timestamp.month,
            "day_of_month": timestamp.day
        }

    def get_route_info(self, line_number: str, direction: str) -> Dict[str, Any]:
        """Get route segment and planned travel time"""
        route_info = self.route_segments.get(line_number, {
            "main_route": f"Line {line_number}",
            "avg_travel_time": 30
        })
        
        return {
            "route_segment": route_info["main_route"],
            "planned_travel_time_min": route_info["avg_travel_time"]
        }

    def parse_delay(self, delay_str: str) -> float:
        """Parse delay string to float minutes"""
        try:
            return float(delay_str)
        except (ValueError, TypeError):
            return 0.0

    def enhance_dataframe(self, df: pd.DataFrame, timestamp: datetime = None) -> pd.DataFrame:
        """Enhance the dataframe with all additional features"""
        # Get weather data once for all records
        weather_data = self.get_current_weather()
        temporal_data = self.add_temporal_features(timestamp)
        
        # Create enhanced dataframe
        enhanced_df = df.copy()
        
        # Add temporal features
        enhanced_df['hour_of_day'] = temporal_data['hour_of_day']
        enhanced_df['day_of_week'] = temporal_data['day_of_week']
        enhanced_df['is_weekend'] = temporal_data['is_weekend']
        enhanced_df['month'] = temporal_data['month']
        enhanced_df['day_of_month'] = temporal_data['day_of_month']
        
        # Add weather features
        enhanced_df['temperature'] = weather_data['temperature']
        enhanced_df['precipitation'] = weather_data['precipitation']
        enhanced_df['weather_condition'] = weather_data['weather_condition']
        enhanced_df['humidity'] = weather_data['humidity']
        enhanced_df['wind_speed'] = weather_data['wind_speed']
        
        # Clean and enhance existing columns
        enhanced_df['vehicle_type'] = enhanced_df['Rodzaj pojazdu'].str.lower()
        enhanced_df['line_number'] = enhanced_df['Linia'].astype(str)
        enhanced_df['delay_minutes'] = enhanced_df['Opóźnienie'].apply(self.parse_delay)
        
        # Add route information
        route_info = enhanced_df.apply(
            lambda row: self.get_route_info(row['line_number'], row['Kierunek']), 
            axis=1
        )
        
        enhanced_df['route_segment'] = [info['route_segment'] for info in route_info]
        enhanced_df['planned_travel_time_min'] = [info['planned_travel_time_min'] for info in route_info]
        
        # Add additional useful features
        enhanced_df['vehicle_number'] = enhanced_df['Nr tab.']
        enhanced_df['brigade'] = enhanced_df['Brygada']
        enhanced_df['direction'] = enhanced_df['Kierunek']
        enhanced_df['stop_name'] = enhanced_df['Przystanek']
        enhanced_df['delay_category'] = pd.cut(
            enhanced_df['delay_minutes'], 
            bins=[-float('inf'), 0, 2, 5, 10, float('inf')],
            labels=['on_time', 'slight_delay', 'moderate_delay', 'significant_delay', 'major_delay']
        )
        
        # Calculate performance metrics
        enhanced_df['delay_ratio'] = enhanced_df['delay_minutes'] / enhanced_df['planned_travel_time_min']
        enhanced_df['is_delayed'] = enhanced_df['delay_minutes'] > 2  # More than 2 minutes is considered delayed
        
        return enhanced_df

    def save_enhanced_data(self, df: pd.DataFrame, output_file: str = "enhanced_tram_data_2.csv"):
        """Save enhanced dataframe to CSV"""
        # Select the most important columns for the output
        output_columns = [
            'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
            'temperature', 'precipitation', 'weather_condition', 'humidity', 'wind_speed',
            'line_number', 'vehicle_type', 'vehicle_number', 'brigade',
            'route_segment', 'direction', 'stop_name',
            'planned_travel_time_min', 'delay_minutes', 'delay_category', 'delay_ratio', 'is_delayed'
        ]
        
        output_df = df[output_columns].copy()
        output_df.to_csv(output_file, index=False)
        print(f"Enhanced data saved to {output_file}")
        
        # Print summary statistics
        print("\n=== Data Enhancement Summary ===")
        print(f"Total records: {len(output_df)}")
        print(f"Lines covered: {output_df['line_number'].nunique()}")
        print(f"Average delay: {output_df['delay_minutes'].mean():.1f} minutes")
        print(f"Delayed vehicles (>2min): {output_df['is_delayed'].sum()} ({output_df['is_delayed'].mean()*100:.1f}%)")
        print(f"Current weather: {output_df['weather_condition'].iloc[0]} at {output_df['temperature'].iloc[0]:.1f}°C")
        
        return output_df

def main():
    """Main function to process the tram delay data"""
    # Initialize the enhancer
    # You can set your OpenWeatherMap API key here or as environment variable
    api_key = None  # Replace with your API key or set OPENWEATHER_API_KEY environment variable
    enhancer = KrakowTramDataEnhancer(api_key="a8623a425e44e8fa82ace92a7e6ec4c2")
    
    # Load the CSV file
    csv_file = "Kraków - Opóźnienia pojazdów komunikacji miejskiej - Czynaczas.pl - 2025-05-31 12_47_24 .csv"
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records from {csv_file}")
        
        # Parse timestamp from filename (you can modify this based on your needs)
        # For now, using current time - in real scenario you'd parse from filename
        timestamp = datetime.now()
        
        # Enhance the data
        enhanced_df = enhancer.enhance_dataframe(df, timestamp)
        
        # Save enhanced data
        output_df = enhancer.save_enhanced_data(enhanced_df)
        
        # Display sample of enhanced data
        print("\n=== Sample Enhanced Data ===")
        sample_cols = ['line_number', 'delay_minutes', 'temperature', 'weather_condition', 
                      'hour_of_day', 'is_weekend', 'route_segment']
        print(output_df[sample_cols].head(10).to_string(index=False))
        
        # Create example JSON record as requested
        sample_record = {
            "hour_of_day": int(output_df['hour_of_day'].iloc[0]),
            "day_of_week": int(output_df['day_of_week'].iloc[0]),
            "is_weekend": bool(output_df['is_weekend'].iloc[0]),
            "temperature": float(output_df['temperature'].iloc[0]),
            "precipitation": float(output_df['precipitation'].iloc[0]),
            "weather_condition": str(output_df['weather_condition'].iloc[0]),
            "line_number": str(output_df['line_number'].iloc[0]),
            "vehicle_type": str(output_df['vehicle_type'].iloc[0]),
            "route_segment": str(output_df['route_segment'].iloc[0]),
            "planned_travel_time_min": int(output_df['planned_travel_time_min'].iloc[0]),
            "delay_minutes": float(output_df['delay_minutes'].iloc[0])
        }
        
        print("\n=== Example Enhanced Record (JSON format) ===")
        print(json.dumps(sample_record, indent=2))
        
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file '{csv_file}'")
        print("Please make sure the file is in the same directory as this script.")
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()
