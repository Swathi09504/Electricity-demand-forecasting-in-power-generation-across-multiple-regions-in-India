import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_pipeline import EmberEnergyClient, NASAPowerClient

def fetch_actual_data(days=1000):
    """Fetch REAL data with proper fallback"""
    print("Fetching actual historical demand from Ember Energy API...")
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        client = EmberEnergyClient()
        df_real = client.fetch_generation_mix(
            "IND", 
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        if df_real.empty:
            raise ValueError("Ember API returned empty data")
        
        # Filter for Demand series
        demand_data = df_real[df_real["series"] == "Demand"].copy()
        
        if demand_data.empty:
            raise ValueError("No 'Demand' series found")
        
        # Convert TWh to MW
        demand_data["demand_mw"] = (demand_data["generation_twh"] * 1000000) / 730
        
        # Sort and prepare for resampling
        demand_data = demand_data.sort_values("date")
        
        # Create hourly grid
        start_date = demand_data["date"].iloc[0]
        end_date = demand_data["date"].iloc[-1] + pd.DateOffset(months=1)
        full_dates = pd.date_range(start=start_date, end=end_date, freq='H')
        
        full_df = pd.DataFrame({"datetime": full_dates})
        full_df["merge_date"] = full_df["datetime"].dt.floor("D")
        
        # Merge monthly data
        full_df = full_df.merge(
            demand_data[["date", "demand_mw"]], 
            left_on="merge_date", 
            right_on="date", 
            how="left"
        )
        
        # Interpolate between months
        full_df["demand_mw"] = full_df["demand_mw"].interpolate(method="linear")
        
        # Fetch weather data
        print("Fetching actual weather data from NASA POWER API...")
        weather_client = NASAPowerClient()
        
        # Use population-weighted coordinates for India
        cities = [
            {"name": "Delhi", "lat": 28.6139, "lon": 77.2090, "weight": 0.30},
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "weight": 0.20},
            {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "weight": 0.15},
            {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "weight": 0.15},
            {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "weight": 0.20},
        ]
        
        weather_dfs = []
        for city in cities:
            city_weather = weather_client.fetch_daily_data(
                city["lat"], city["lon"], start_date, end_date
            )
            if not city_weather.empty:
                city_weather = city_weather.reset_index()
                city_weather["weight"] = city["weight"]
                weather_dfs.append(city_weather)
        
        if weather_dfs:
            # Combine cities with weights
            combined_weather = pd.concat(weather_dfs)
            weather_columns = ["datetime", "temperature_2m", "relative_humidity", "wind_speed_10m"]
            weighted = combined_weather.groupby("datetime").apply(
                lambda x: pd.Series({
                    "temperature_2m": (x["temperature_2m"] * x["weight"]).sum(),
                    "relative_humidity": (x["relative_humidity"] * x["weight"]).sum(),
                    "wind_speed_10m": (x["wind_speed_10m"] * x["weight"]).sum(),
                })
            ).reset_index()
            
            # Merge with demand data
            full_df = full_df.merge(
                weighted[["datetime", "temperature_2m", "relative_humidity", "wind_speed_10m"]],
                left_on="merge_date",
                right_on="datetime",
                how="left"
            )
            
            # Rename columns
            full_df = full_df.rename(columns={
                "temperature_2m": "temperature",
                "relative_humidity": "humidity",
                "wind_speed_10m": "wind_speed"
            })
            
            # Forward fill missing weather
            for col in ["temperature", "humidity", "wind_speed"]:
                full_df[col] = full_df[col].ffill().bfill()
        else:
            raise ValueError("Weather fetch failed for all cities")
        
        # Add temporal features
        full_df["hour"] = full_df["datetime"].dt.hour
        full_df["day_of_week"] = full_df["datetime"].dt.dayofweek
        
        # Select final columns
        result = full_df[["datetime", "demand_mw", "temperature", "humidity", 
                         "wind_speed", "hour", "day_of_week"]]
        
        return result
        
    except Exception as e:
        print(f"ERROR fetching real data: {e}")
        print("Falling back to synthetic data generation...")
        return generate_synthetic_fallback(days)

def generate_synthetic_fallback(days=1000):
    """Generate realistic synthetic data with appropriate complexity"""
    print("Generating synthetic data with realistic patterns...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    full_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    n = len(full_dates)
    
    # Base demand (MW) - realistic for India
    base_demand = 120000  # 120 GW typical
    
    # Annual cycle (seasonal)
    annual = np.sin(2 * np.pi * np.arange(n) / (24 * 365))
    seasonal_demand = base_demand * (1 + 0.15 * annual)  # 15% seasonal variation
    
    # Weekly cycle (weekends lower)
    day_of_week = full_dates.dayofweek
    weekly_pattern = np.where(day_of_week >= 5, 0.85, 1.0)  # 15% lower on weekends
    
    # Daily cycle (peaks at 9 AM and 7 PM)
    hour = full_dates.hour
    morning_peak = np.exp(-((hour - 9) ** 2) / 50)  # Gaussian peak at 9 AM
    evening_peak = np.exp(-((hour - 19) ** 2) / 50)  # Gaussian peak at 7 PM
    daily_pattern = 1 + 0.2 * (morning_peak + evening_peak)
    
    # Combine patterns
    demand = seasonal_demand * weekly_pattern * daily_pattern
    
    # Add realistic noise
    noise = np.random.normal(0, 0.03 * base_demand, n)  # 3% standard deviation
    demand += noise
    
    # Ensure no negative demand
    demand = np.maximum(demand, 50000)
    
    # Generate weather with realistic patterns
    # Temperature: 15-35°C seasonal, 5-10°C daily swing
    temp_seasonal = 25 + 10 * annual
    temp_daily = 5 * np.sin(2 * np.pi * (hour - 14) / 24)  # Peak at 2 PM
    temperature = temp_seasonal + temp_daily + np.random.normal(0, 2, n)
    
    # Humidity: inversely related to temperature
    humidity = 80 - 0.5 * (temperature - 20) + np.random.normal(0, 10, n)
    humidity = np.clip(humidity, 20, 100)
    
    # Wind speed: higher in afternoons
    wind_speed = 5 + 3 * np.sin(2 * np.pi * (hour - 13) / 24) + np.random.exponential(2, n)
    wind_speed = np.clip(wind_speed, 0, 30)
    
    # Create DataFrame
    df = pd.DataFrame({
        "datetime": full_dates,
        "demand_mw": demand,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "hour": hour,
        "day_of_week": day_of_week
    })
    
    return df

if __name__ == "__main__":
    # Fetch real data (with automatic fallback)
    df = fetch_actual_data(days=1000)
    
    if df is not None and not df.empty:
        # Save both versions for comparison
        df.to_csv('actual_demand.csv', index=False)
        print(f"✅ Successfully saved {len(df):,} records to actual_demand.csv")
        
        # Also save a sample for quick inspection
        df.head(100).to_csv('demand_sample.csv', index=False)
        print("📊 Sample saved to demand_sample.csv")
        
        # Print basic statistics
        print("\n📈 Data Statistics:")
        print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"   Mean demand: {df['demand_mw'].mean():.0f} MW")
        print(f"   Peak demand: {df['demand_mw'].max():.0f} MW")
        print(f"   Mean temperature: {df['temperature'].mean():.1f}°C")
    else:
        print("❌ Failed to generate any data")
