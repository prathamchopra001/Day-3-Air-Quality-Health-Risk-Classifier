"""
Data Collection Script - Fetch real air quality data from OpenAQ API
Saves data to CSV for model training
"""

import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"

# Headers
headers = {"Accept": "application/json"}
if API_KEY:
    headers["X-API-Key"] = API_KEY

# Major cities with known air quality monitoring
CITIES = [
    # Asia (often high pollution)
    {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
    {"name": "Beijing", "lat": 39.9042, "lon": 116.4074},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Shanghai", "lat": 31.2304, "lon": 121.4737},
    {"name": "Dhaka", "lat": 23.8103, "lon": 90.4125},
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011},
    {"name": "Jakarta", "lat": -6.2088, "lon": 106.8456},
    {"name": "Seoul", "lat": 37.5665, "lon": 126.9780},
    {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
    {"name": "Bangkok", "lat": 13.7563, "lon": 100.5018},
    
    # Europe (generally moderate)
    {"name": "London", "lat": 51.5074, "lon": -0.1278},
    {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
    {"name": "Berlin", "lat": 52.5200, "lon": 13.4050},
    {"name": "Madrid", "lat": 40.4168, "lon": -3.7038},
    {"name": "Rome", "lat": 41.9028, "lon": 12.4964},
    {"name": "Warsaw", "lat": 52.2297, "lon": 21.0122},
    
    # Americas
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"name": "Mexico City", "lat": 19.4326, "lon": -99.1332},
    {"name": "Sao Paulo", "lat": -23.5505, "lon": -46.6333},
    {"name": "Lima", "lat": -12.0464, "lon": -77.0428},
    {"name": "Bogota", "lat": 4.7110, "lon": -74.0721},
    
    # Africa & Middle East
    {"name": "Cairo", "lat": 30.0444, "lon": 31.2357},
    {"name": "Lagos", "lat": 6.5244, "lon": 3.3792},
    {"name": "Dubai", "lat": 25.2048, "lon": 55.2708},
    
    # Oceania
    {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
    {"name": "Melbourne", "lat": -37.8136, "lon": 144.9631},
]

# Parameter name mapping
PARAM_MAP = {
    "pm25": "pm25", "pm2.5": "pm25",
    "pm10": "pm10",
    "o3": "o3", "ozone": "o3",
    "no2": "no2", "nitrogen dioxide": "no2",
    "so2": "so2", "sulfur dioxide": "so2", "sulphur dioxide": "so2",
    "co": "co", "carbon monoxide": "co",
}


def request_api(endpoint: str, params: dict = None) -> dict:
    """Make API request with rate limiting."""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        
        if resp.status_code == 429:
            print("  Rate limited, waiting 5s...")
            time.sleep(5)
            return request_api(endpoint, params)
        
        if resp.status_code != 200:
            return {"error": f"Status {resp.status_code}: {resp.text[:200]}"}
        
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def find_locations_near(lat: float, lon: float) -> list:
    """Find monitoring locations near coordinates."""
    data = request_api("/locations", {
        "coordinates": f"{lat},{lon}",
        "radius": 25000,
        "limit": 10
    })
    return data.get("results", [])


def get_location_measurements(location_id: int) -> dict:
    """Get latest measurements for a location."""
    data = request_api(f"/locations/{location_id}/latest")
    
    if "error" in data:
        return {}
    
    measurements = {}
    for item in data.get("results", []):
        param = item.get("parameter", {})
        name = param.get("name", "") if isinstance(param, dict) else str(param)
        name = name.lower().strip()
        
        std_name = PARAM_MAP.get(name)
        if std_name and item.get("value") is not None:
            measurements[std_name] = float(item["value"])
    
    return measurements


def get_sensor_measurements(sensor_id: int, limit: int = 100) -> list:
    """Get historical measurements from a sensor."""
    data = request_api(f"/sensors/{sensor_id}/measurements", {
        "limit": limit
    })
    return data.get("results", [])


def collect_data():
    """Main data collection function."""
    print("="*60)
    print("AIR QUALITY DATA COLLECTION")
    print("="*60)
    
    if not API_KEY:
        print("ERROR: OPENAQ_API_KEY not set!")
        return
    
    all_data = []
    
    for city in CITIES:
        print(f"\n📍 {city['name']}...")
        
        # Find locations near this city
        locations = find_locations_near(city["lat"], city["lon"])
        
        if not locations:
            print(f"  No stations found")
            continue
        
        print(f"  Found {len(locations)} stations")
        
        # Get data from each location
        for loc in locations[:5]:  # Max 5 stations per city
            loc_id = loc.get("id")
            loc_name = loc.get("name", "Unknown")
            
            # Get current measurements
            measurements = get_location_measurements(loc_id)
            
            if measurements:
                row = {
                    "city": city["name"],
                    "station_id": loc_id,
                    "station_name": loc_name,
                    "latitude": city["lat"],
                    "longitude": city["lon"],
                    "timestamp": datetime.now().isoformat(),
                    "pm25": measurements.get("pm25"),
                    "pm10": measurements.get("pm10"),
                    "o3": measurements.get("o3"),
                    "no2": measurements.get("no2"),
                    "so2": measurements.get("so2"),
                    "co": measurements.get("co"),
                }
                all_data.append(row)
                print(f"    ✓ {loc_name}: {measurements}")
            
            # Also try to get historical data from sensors
            sensors = loc.get("sensors", [])
            if sensors and len(all_data) < 5000:  # Limit total data
                for sensor in sensors[:3]:
                    sensor_id = sensor.get("id")
                    param = sensor.get("parameter", {})
                    param_name = param.get("name", "").lower() if isinstance(param, dict) else ""
                    std_name = PARAM_MAP.get(param_name)
                    
                    if sensor_id and std_name:
                        hist_data = get_sensor_measurements(sensor_id, limit=50)
                        
                        for item in hist_data:
                            value = item.get("value")
                            timestamp = item.get("period", {}).get("datetimeFrom", {}).get("utc")
                            
                            if value is not None:
                                hist_row = {
                                    "city": city["name"],
                                    "station_id": loc_id,
                                    "station_name": loc_name,
                                    "latitude": city["lat"],
                                    "longitude": city["lon"],
                                    "timestamp": timestamp,
                                    "pm25": value if std_name == "pm25" else None,
                                    "pm10": value if std_name == "pm10" else None,
                                    "o3": value if std_name == "o3" else None,
                                    "no2": value if std_name == "no2" else None,
                                    "so2": value if std_name == "so2" else None,
                                    "co": value if std_name == "co" else None,
                                }
                                all_data.append(hist_row)
            
            time.sleep(0.5)  # Rate limiting
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total records: {len(df)}")
    print(f"Cities covered: {df['city'].nunique()}")
    print(f"Stations: {df['station_id'].nunique()}")
    print(f"\nPollutant coverage:")
    for col in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
        count = df[col].notna().sum()
        print(f"  {col}: {count} readings ({100*count/len(df):.1f}%)")
    
    # Save to CSV
    output_path = Path("data/air_quality_raw.csv")
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Data saved to {output_path}")
    
    return df


def process_data(input_path: str = "data/air_quality_raw.csv"):
    """Process raw data for training."""
    print("\n" + "="*60)
    print("PROCESSING DATA FOR TRAINING")
    print("="*60)
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records")
    
    # Show what we got per city
    print("\nData per city:")
    for city in df['city'].unique():
        city_df = df[df['city'] == city]
        pm25_count = city_df['pm25'].notna().sum()
        print(f"  {city}: {len(city_df)} records, {pm25_count} with PM2.5")
    
    # Keep only records with PM2.5 (required)
    df_valid = df[df['pm25'].notna() & (df['pm25'] > 0)].copy()
    
    # Fill missing pollutants with realistic estimates based on PM2.5
    # These ratios are based on typical urban air quality relationships
    for idx, row in df_valid.iterrows():
        pm25 = row['pm25']
        
        # PM10 is typically 1.5-3x PM2.5
        if pd.isna(row['pm10']) or row['pm10'] == 0:
            df_valid.loc[idx, 'pm10'] = pm25 * np.random.uniform(1.5, 2.5)
        
        # O3 (in ppm) - inversely related to PM in urban areas
        if pd.isna(row['o3']) or row['o3'] == 0:
            # Higher PM usually means lower O3 (NOx titration)
            base_o3 = 0.03 + (0.05 * (1 - min(pm25, 200) / 200))
            df_valid.loc[idx, 'o3'] = base_o3 * np.random.uniform(0.7, 1.3)
        
        # SO2 (in ppb) - correlates with industrial pollution
        if pd.isna(row['so2']) or row['so2'] == 0:
            df_valid.loc[idx, 'so2'] = max(5, pm25 * np.random.uniform(0.1, 0.3))
        
        # CO (in ppm) - correlates with traffic/combustion
        if pd.isna(row['co']) or row['co'] == 0:
            df_valid.loc[idx, 'co'] = max(0.3, pm25 * np.random.uniform(0.005, 0.02))
    
    # Remove extreme outliers
    df_valid = df_valid[
        (df_valid["pm25"] < 800) &
        (df_valid["pm10"] < 1500) &
        (df_valid["o3"] < 0.3) &
        (df_valid["no2"] < 500) &
        (df_valid["so2"] < 300) &
        (df_valid["co"] < 30)
    ]
    
    print(f"\nProcessed records: {len(df_valid)}")
    print(f"\nPollutant ranges in processed data:")
    for col in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
        vals = df_valid[col]
        print(f"  {col}: {vals.min():.2f} - {vals.max():.2f} (mean: {vals.mean():.2f})")
    
    print(f"\nSample processed data:")
    print(df_valid[["city", "pm25", "pm10", "o3", "no2", "so2", "co"]].head(10).to_string())
    
    # Save processed data
    output_path = "data/air_quality_training.csv"
    df_valid.to_csv(output_path, index=False)
    print(f"\n✓ Training data saved to {output_path}")
    print(f"  Total records: {len(df_valid)}")
    
    return df_valid


if __name__ == "__main__":
    # Collect raw data
    collect_data()
    
    # Process for training
    process_data()