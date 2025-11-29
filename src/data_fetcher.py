"""
OpenAQ API Client - Fetches real-time air quality data
Fixed to correctly parse v3 API responses
"""

import os
import time
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List


class OpenAQClient:
    """Client for OpenAQ API v3"""
    
    BASE_URL = "https://api.openaq.org/v3"
    
    # Parameter name mapping
    PARAM_MAP = {
        "pm25": "pm25", "pm2.5": "pm25",
        "pm10": "pm10",
        "o3": "o3", "ozone": "o3",
        "no2": "no2", "nitrogen dioxide": "no2",
        "so2": "so2", "sulfur dioxide": "so2", "sulphur dioxide": "so2",
        "co": "co", "carbon monoxide": "co",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAQ_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        if self.api_key:
            self.session.headers.update({"X-API-Key": self.api_key})
    
    def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with error handling."""
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            resp = self.session.get(url, params=params, timeout=30)
            
            if resp.status_code == 401:
                return {"error": "API key required. Get one at: https://explore.openaq.org/register"}
            if resp.status_code == 429:
                time.sleep(2)
                return self._request(endpoint, params)
            if resp.status_code == 422:
                return {"error": f"Invalid request: {resp.text[:200]}"}
            
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def find_locations(self, lat: float, lon: float, radius: int = 25000) -> List[Dict]:
        """Find monitoring stations near coordinates (max radius: 25km)."""
        data = self._request("/locations", {
            "coordinates": f"{lat},{lon}",
            "radius": min(radius, 25000),
            "limit": 100,  # Get more to filter for recent ones
            "order_by": "lastUpdated",
            "sort_order": "desc"
        })
        
        if "error" in data:
            return []
        
        locations = data.get("results", [])
        
        # Filter for locations with recent data (within last 7 days)
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        recent_locations = []
        
        for loc in locations:
            last_updated = loc.get("datetimeLast", {})
            if isinstance(last_updated, dict):
                last_str = last_updated.get("utc", "")
            else:
                last_str = str(last_updated) if last_updated else ""
            
            if last_str:
                try:
                    # Parse datetime string
                    last_dt = datetime.fromisoformat(last_str.replace("Z", "+00:00"))
                    last_dt = last_dt.replace(tzinfo=None)  # Make naive for comparison
                    
                    if last_dt > recent_cutoff:
                        recent_locations.append(loc)
                except:
                    pass
        
        return recent_locations
    
    def get_latest(self, location_id: int, sensors: List[Dict]) -> Dict[str, float]:
        """
        Get latest measurements for a location.
        
        Args:
            location_id: The location ID
            sensors: List of sensor dicts from the location object
            
        Returns:
            Dict mapping parameter names to values
        """
        # Build sensor ID to parameter name mapping
        sensor_param_map = {}
        for sensor in sensors:
            sensor_id = sensor.get("id")
            param = sensor.get("parameter", {})
            param_name = param.get("name", "").lower() if isinstance(param, dict) else str(param).lower()
            std_name = self.PARAM_MAP.get(param_name, param_name)
            if sensor_id and std_name:
                sensor_param_map[sensor_id] = std_name
        
        # Get latest measurements
        data = self._request(f"/locations/{location_id}/latest")
        
        if "error" in data:
            return {}
        
        measurements = {}
        results = data.get("results", [])
        
        for item in results:
            sensor_id = item.get("sensorsId")
            value = item.get("value")
            
            if sensor_id in sensor_param_map and value is not None:
                param_name = sensor_param_map[sensor_id]
                measurements[param_name] = float(value)
        
        return measurements
    
    def get_measurements(self, lat: float, lon: float) -> Dict:
        """Get air quality measurements near coordinates."""
        
        # Find nearby stations with recent data
        locations = self.find_locations(lat, lon)
        
        if not locations:
            # Try with larger effective area by not filtering for recent
            data = self._request("/locations", {
                "coordinates": f"{lat},{lon}",
                "radius": 25000,
                "limit": 20
            })
            locations = data.get("results", []) if "error" not in data else []
            
            if not locations:
                return {"error": "No monitoring stations found within 25km"}
        
        # Try each station
        for loc in locations:
            loc_id = loc.get("id")
            loc_name = loc.get("name", "Unknown")
            sensors = loc.get("sensors", [])
            
            if not sensors:
                continue
            
            # Get measurements
            measurements = self.get_latest(loc_id, sensors)
            
            if measurements:
                country = loc.get("country", {})
                country_name = country.get("name") if isinstance(country, dict) else str(country or "")
                
                # Get last update time
                last_updated = loc.get("datetimeLast", {})
                if isinstance(last_updated, dict):
                    last_str = last_updated.get("utc", "")
                else:
                    last_str = ""
                
                return {
                    "station_id": loc_id,
                    "station_name": loc_name,
                    "city": loc.get("locality") or loc.get("city") or "",
                    "country": country_name,
                    "measurements": measurements,
                    "last_updated": last_str,
                    "timestamp": datetime.now().isoformat()
                }
        
        return {"error": "No recent measurements available from nearby stations"}
    
    def close(self):
        self.session.close()


def geocode(query: str) -> Dict:
    """Convert location name to coordinates using Nominatim."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "AirQualityClassifier/1.0"},
            timeout=10
        )
        results = resp.json()
        
        if results:
            return {
                "lat": float(results[0]["lat"]),
                "lon": float(results[0]["lon"]),
                "name": results[0]["display_name"]
            }
        return {"error": "Location not found"}
    except Exception as e:
        return {"error": str(e)}


# Quick test
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    client = OpenAQClient()
    
    # Test with Delhi
    print("Testing Delhi...")
    result = client.get_measurements(28.6139, 77.2090)
    print(f"Result: {result}")
    
    # Test with London
    print("\nTesting London...")
    result = client.get_measurements(51.5074, -0.1278)
    print(f"Result: {result}")
    
    # Test with Los Angeles
    print("\nTesting Los Angeles...")
    result = client.get_measurements(34.0522, -118.2437)
    print(f"Result: {result}")
    
    client.close()