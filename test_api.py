"""Quick debug to see exact API response"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("OPENAQ_API_KEY")
BASE = "https://api.openaq.org/v3"
headers = {"Accept": "application/json", "X-API-Key": API_KEY}

# Step 1: Find a location
print("Finding locations near Delhi...")
resp = requests.get(f"{BASE}/locations", headers=headers, params={
    "coordinates": "28.6139,77.2090",
    "radius": 25000,
    "limit": 3
})
print(f"Status: {resp.status_code}")
locations = resp.json().get("results", [])
print(f"Found {len(locations)} locations\n")

if locations:
    loc = locations[0]
    loc_id = loc.get("id")
    print(f"First location: {loc.get('name')} (ID: {loc_id})")
    print(f"Full location object keys: {loc.keys()}")
    
    # Check if sensors are embedded in location
    sensors = loc.get("sensors", [])
    print(f"\nSensors in location object: {len(sensors)}")
    if sensors:
        print("Sample sensor:")
        print(f"  {sensors[0]}")
    
    # Step 2: Try /latest endpoint
    print(f"\n--- Trying /locations/{loc_id}/latest ---")
    resp = requests.get(f"{BASE}/locations/{loc_id}/latest", headers=headers)
    print(f"Status: {resp.status_code}")
    data = resp.json()
    print(f"Response keys: {data.keys()}")
    print(f"Results: {data.get('results', [])}")
    
    # Step 3: Try getting the location details
    print(f"\n--- Trying /locations/{loc_id} ---")
    resp = requests.get(f"{BASE}/locations/{loc_id}", headers=headers)
    print(f"Status: {resp.status_code}")
    data = resp.json()
    results = data.get("results", [])
    if results:
        loc_detail = results[0] if isinstance(results, list) else results
        print(f"Location keys: {loc_detail.keys()}")
        
        # Check for latest values in sensors
        sensors = loc_detail.get("sensors", [])
        print(f"\nSensors: {len(sensors)}")
        for s in sensors[:3]:
            print(f"  Sensor {s.get('id')}: {s.get('parameter')} = {s.get('latest')}")
    
    # Step 4: Try /sensors endpoint
    if sensors:
        sensor_id = sensors[0].get("id")
        print(f"\n--- Trying /sensors/{sensor_id} ---")
        resp = requests.get(f"{BASE}/sensors/{sensor_id}", headers=headers)
        print(f"Status: {resp.status_code}")
        data = resp.json()
        print(f"Response: {data}")