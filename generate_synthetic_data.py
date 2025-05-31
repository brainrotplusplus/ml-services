import pandas as pd
import numpy as np
import random
import argparse
from typing import List, Dict, Any


class SyntheticTramDataGenerator:
    """Generator for synthetic tram delay data that mimics real patterns."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the generator with a random seed for reproducibility."""
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Define realistic data distributions based on real tram data
        self.weather_conditions = ['rain', 'sunny', 'cloudy', 'snow', 'fog']
        self.weather_probabilities = [0.3, 0.4, 0.2, 0.05, 0.05]
        
        self.vehicle_prefixes = ['HY', 'RY', 'RP', 'HG', 'HL', 'RF', 'RG', 'RZ']
        self.line_numbers = [1, 3, 4, 5, 8, 9, 10, 11, 14, 18, 20, 21, 22, 24, 50, 52, 72, 76, 77, 78]
        self.planned_travel_times = [20, 25, 30, 35, 40, 45]
        
        self.route_segments = [
            "Kurdwanów–Pleszów", 
            "Czerwone Maki–Os.Piastów", 
            "Borek Fałęcki–Kopiec Wandy",
            "Mały Płaszów–Cichy Kącik", 
            "Górka Narodowa–Czerwone Maki", 
            "Bronowice–Borek Fałęcki",
            "Salwator–Krowodrza Górka", 
            "Nowy Bieżanów–Krowodrza", 
            "Jarzębiny–Krowodrza",
            "Rondo Hipokratesa–Nowy Bieżanów",
            "Os.Piastów–Pleszów",
            "Rondo Grunwaldzkie–Czerwone Maki"
        ]
        
        self.directions = [
            "Kurdwanów P+R", "Czerwone Maki P+R", "Borek Fałęcki", 
            "Górka Narodowa P+R", "Salwator", "Bronowice Małe", 
            "Nowy Bieżanów P+R", "Mały Płaszów P+R", "Krowodrza G. P+R",
            "Os.Piastów", "Pleszów", "Rondo Hipokratesa"
        ]
        
        self.stop_names = [
            "Kurdwanów P+R 03", "Czerwone Maki P+R 01", "Borek Fałęcki 01", 
            "Mały Płaszów P+R 01", "Górka Narodowa P+R 01", "Salwator 01",
            "Bronowice Małe 01", "Nowy Bieżanów P+R 01", "Krowodrza Górka P+R 03",
            "Os. Piastów 01", "Pleszów 01", "Rondo Hipokratesa 02",
            "Teatr Słowackiego 01", "Rondo Grunwaldzkie 01", "Dworzec Towarowy 01"
        ]
        
    def _generate_vehicle_numbers(self, count: int = 50) -> List[str]:
        """Generate realistic vehicle numbers."""
        return [f"{random.choice(self.vehicle_prefixes)}{random.randint(100, 999)}" 
                for _ in range(count)]
    
    def _generate_brigades(self, count: int = 30) -> List[str]:
        """Generate realistic brigade identifiers."""
        return [f"{random.randint(1, 80):02d}-{random.randint(1, 10):02d}" 
                for _ in range(count)]
    
    def _calculate_delay(self, hour: int, weather: str, temperature: float, 
                        is_weekend: bool, precipitation: float) -> float:
        """Calculate realistic delay based on conditions."""
        base_delay = 0
        
        # Weather impact
        weather_multipliers = {
            'rain': 3.0,
            'snow': 5.0,
            'fog': 2.0,
            'cloudy': 0.5,
            'sunny': 0.0
        }
        base_delay += np.random.exponential(weather_multipliers.get(weather, 0))
        
        # Precipitation impact
        if precipitation > 2.0:
            base_delay += np.random.exponential(2)
        
        # Rush hour impact (7-9 AM, 4-6 PM)
        if hour in [7, 8, 9, 16, 17, 18]:
            base_delay += np.random.exponential(2.5)
        elif hour in [6, 10, 15, 19]:  # Semi-rush hours
            base_delay += np.random.exponential(1.0)
        
        # Weekend effect (usually less delay)
        if is_weekend:
            base_delay *= 0.6
        
        # Temperature effect (extreme temperatures cause delays)
        if temperature < -5:
            base_delay += np.random.exponential(2)
        elif temperature > 30:
            base_delay += np.random.exponential(1.5)
        elif temperature < 0:
            base_delay += np.random.exponential(0.5)
        
        # Late night/early morning effect
        if hour < 6 or hour > 22:
            base_delay *= 0.5
        
        # Add random noise
        delay = max(0, base_delay + np.random.normal(0, 1))
        
        return delay
    
    def _categorize_delay(self, delay_minutes: float) -> str:
        """Categorize delay into predefined categories."""
        if delay_minutes >= 10:
            return 'major_delay'
        elif delay_minutes >= 5:
            return 'significant_delay'
        elif delay_minutes >= 3:
            return 'moderate_delay'
        elif delay_minutes >= 1:
            return 'slight_delay'
        else:
            return 'on_time'
    
    def generate_sample(self) -> Dict[str, Any]:
        """Generate a single synthetic tram data sample."""
        # Temporal features
        hour_of_day = np.random.randint(5, 24)
        day_of_week = np.random.randint(0, 7)
        is_weekend = day_of_week >= 5
        month = np.random.randint(1, 13)
        day_of_month = np.random.randint(1, 29)
        
        # Weather features with seasonal variation
        seasonal_temp_adjust = {
            1: -5, 2: -3, 3: 0, 4: 5, 5: 10, 6: 15,
            7: 18, 8: 17, 9: 12, 10: 7, 11: 2, 12: -3
        }
        base_temp = 15 + seasonal_temp_adjust.get(month, 0)
        temperature = np.random.normal(base_temp, 6)
        
        # Precipitation more likely in certain months
        precip_multiplier = 1.5 if month in [11, 12, 1, 2, 4, 5] else 1.0
        precipitation = np.random.exponential(0.5 * precip_multiplier)
        
        weather_condition = np.random.choice(
            self.weather_conditions, 
            p=self.weather_probabilities
        )
        
        # Adjust weather based on temperature and season
        if temperature < 0 and weather_condition == 'rain':
            weather_condition = 'snow'
        elif month in [6, 7, 8] and weather_condition == 'snow':
            weather_condition = 'sunny'
        
        humidity = np.random.randint(30, 95)
        wind_speed = np.random.exponential(2)
        
        # Transport features
        line_number = np.random.choice(self.line_numbers)
        vehicle_type = 'tramwaj'
        vehicle_number = f"{random.choice(self.vehicle_prefixes)}{random.randint(100, 999)}"
        brigade = f"{random.randint(1, 80):02d}-{random.randint(1, 10):02d}"
        route_segment = np.random.choice(self.route_segments)
        direction = np.random.choice(self.directions)
        stop_name = np.random.choice(self.stop_names)
        
        planned_travel_time_min = np.random.choice(self.planned_travel_times)
        
        # Calculate delay based on conditions
        delay_minutes = self._calculate_delay(
            hour_of_day, weather_condition, temperature, 
            is_weekend, precipitation
        )
        
        # Calculate derived features
        delay_ratio = delay_minutes / planned_travel_time_min if planned_travel_time_min > 0 else 0
        delay_category = self._categorize_delay(delay_minutes)
        is_delayed = delay_minutes >= 3
        
        return {
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'month': month,
            'day_of_month': day_of_month,
            'temperature': round(temperature, 1),
            'precipitation': round(precipitation, 1),
            'weather_condition': weather_condition,
            'humidity': humidity,
            'wind_speed': round(wind_speed, 1),
            'line_number': line_number,
            'vehicle_type': vehicle_type,
            'vehicle_number': vehicle_number,
            'brigade': brigade,
            'route_segment': route_segment,
            'direction': direction,
            'stop_name': stop_name,
            'planned_travel_time_min': planned_travel_time_min,
            'delay_minutes': round(delay_minutes, 1),
            'delay_category': delay_category,
            'delay_ratio': round(delay_ratio, 6),
            'is_delayed': is_delayed
        }
    
    def generate_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate a complete synthetic dataset."""
        print(f"Generating {n_samples} synthetic tram data samples...")
        
        synthetic_data = []
        for i in range(n_samples):
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_samples} samples...")
            synthetic_data.append(self.generate_sample())
        
        df = pd.DataFrame(synthetic_data)
        
        # Print statistics
        print(f"\nGenerated dataset statistics:")
        print(f"Shape: {df.shape}")
        print(f"Average delay: {df['delay_minutes'].mean():.2f} minutes")
        print(f"Delayed percentage: {(df['is_delayed'].sum() / len(df) * 100):.1f}%")
        print(f"Delay categories distribution:")
        print(df['delay_category'].value_counts().to_string())
        
        return df


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Generate synthetic tram delay data')
    parser.add_argument('--samples', '-n', type=int, default=1000, 
                       help='Number of samples to generate (default: 1000)')
    parser.add_argument('--output', '-o', type=str, default='synthetic_tram_data.csv',
                       help='Output CSV file name (default: synthetic_tram_data.csv)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Generate synthetic data
    generator = SyntheticTramDataGenerator(random_seed=args.seed)
    df_synthetic = generator.generate_dataset(n_samples=args.samples)
    
    # Save to CSV
    df_synthetic.to_csv(args.output, index=False)
    print(f"\nSynthetic data saved to '{args.output}'")
    
    # Display sample
    print(f"\nSample of generated data:")
    print(df_synthetic.head().to_string())


if __name__ == "__main__":
    main() 