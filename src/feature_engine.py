"""
Feature Engineering - AQI calculations and feature preparation
"""

import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List

# EPA AQI Breakpoints
AQI_BREAKPOINTS = {
    "pm25": [
        (0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ],
    "pm10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ],
    "o3": [  # ppm, 8-hour
        (0, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
    ],
    "no2": [  # ppb
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500),
    ],
    "so2": [  # ppb
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 1004, 301, 500),
    ],
    "co": [  # ppm
        (0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 50.4, 301, 500),
    ],
}

# Health tier definitions
HEALTH_TIERS = {
    0: {"name": "Good", "color": "#00E400", "range": (0, 50)},
    1: {"name": "Moderate", "color": "#FFFF00", "range": (51, 100)},
    2: {"name": "Unhealthy for Sensitive Groups", "color": "#FF7E00", "range": (101, 150)},
    3: {"name": "Unhealthy", "color": "#FF0000", "range": (151, 200)},
    4: {"name": "Very Unhealthy", "color": "#8F3F97", "range": (201, 300)},
    5: {"name": "Hazardous", "color": "#7E0023", "range": (301, 500)},
}

# Feature names for model
FEATURE_NAMES = [
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "pm_ratio", "hour_sin", "hour_cos", "is_rush_hour"
]


def calculate_sub_aqi(concentration: float, pollutant: str) -> float:
    """Calculate AQI for a single pollutant."""
    if pollutant not in AQI_BREAKPOINTS:
        return 0
    
    for c_low, c_high, i_low, i_high in AQI_BREAKPOINTS[pollutant]:
        if c_low <= concentration <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return aqi
    
    # Above maximum range
    return 500


def calculate_aqi(measurements: Dict[str, float]) -> Tuple[float, str]:
    """Calculate overall AQI (max of all sub-indices)."""
    max_aqi = 0
    dominant = "pm25"
    
    for pollutant, value in measurements.items():
        if pollutant in AQI_BREAKPOINTS:
            sub_aqi = calculate_sub_aqi(value, pollutant)
            if sub_aqi > max_aqi:
                max_aqi = sub_aqi
                dominant = pollutant
    
    return max_aqi, dominant


def aqi_to_tier(aqi: float) -> int:
    """Convert AQI value to health tier (0-5)."""
    if aqi <= 50:
        return 0
    elif aqi <= 100:
        return 1
    elif aqi <= 150:
        return 2
    elif aqi <= 200:
        return 3
    elif aqi <= 300:
        return 4
    else:
        return 5


def prepare_features(measurements: Dict[str, float], hour: int = None) -> Tuple[np.ndarray, Dict]:
    """
    Prepare feature array for model prediction.
    
    Returns:
        Tuple of (feature_array, feature_dict)
    """
    if hour is None:
        hour = datetime.now().hour
    
    # Extract pollutant values (use 0 if missing)
    pm25 = measurements.get("pm25", 0)
    pm10 = measurements.get("pm10", 0)
    o3 = measurements.get("o3", 0)
    no2 = measurements.get("no2", 0)
    so2 = measurements.get("so2", 0)
    co = measurements.get("co", 0)
    
    # Calculate derived features
    pm_ratio = pm25 / pm10 if pm10 > 0 else 0
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    is_rush_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
    
    # Create feature array
    features = np.array([[
        pm25, pm10, o3, no2, so2, co,
        pm_ratio, hour_sin, hour_cos, is_rush_hour
    ]])
    
    # Create feature dict for explanations
    feature_dict = {
        "pm25": pm25, "pm10": pm10, "o3": o3,
        "no2": no2, "so2": so2, "co": co,
        "pm_ratio": pm_ratio, "hour_sin": hour_sin,
        "hour_cos": hour_cos, "is_rush_hour": is_rush_hour
    }
    
    return features, feature_dict


def get_recommendations(tier: int) -> List[str]:
    """Get health recommendations for a tier."""
    recommendations = {
        0: [
            "Air quality is satisfactory",
            "Great day for outdoor activities",
            "No precautions needed"
        ],
        1: [
            "Air quality is acceptable",
            "Unusually sensitive people should consider reducing prolonged outdoor exertion",
            "Most people can continue normal activities"
        ],
        2: [
            "Sensitive groups may experience health effects",
            "Children, elderly, and those with respiratory issues should limit prolonged outdoor exertion",
            "Consider indoor activities during peak pollution hours"
        ],
        3: [
            "Everyone may begin to experience health effects",
            "Avoid prolonged outdoor exertion",
            "Keep windows closed",
            "Consider wearing an N95 mask outdoors"
        ],
        4: [
            "Health alert: significant risk for everyone",
            "Avoid all outdoor physical activities",
            "Use air purifiers indoors",
            "Wear N95 mask if going outside is necessary"
        ],
        5: [
            "Emergency conditions - serious health effects for everyone",
            "Stay indoors with windows and doors closed",
            "Run air purifiers continuously",
            "Avoid any outdoor exposure",
            "Seek medical attention if experiencing symptoms"
        ],
    }
    return recommendations.get(tier, recommendations[0])