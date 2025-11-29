"""
SHAP Explainer - Human-readable explanations for predictions
"""

import numpy as np
import joblib
from typing import Dict, List, Optional

# Display names for features
DISPLAY_NAMES = {
    "pm25": "PM2.5 (Fine Particles)",
    "pm10": "PM10 (Coarse Particles)",
    "o3": "Ozone (O₃)",
    "no2": "Nitrogen Dioxide (NO₂)",
    "so2": "Sulfur Dioxide (SO₂)",
    "co": "Carbon Monoxide (CO)",
    "pm_ratio": "PM2.5/PM10 Ratio",
    "hour_sin": "Time of Day (sin)",
    "hour_cos": "Time of Day (cos)",
    "is_rush_hour": "Rush Hour"
}

# Pollutant thresholds for explanations
THRESHOLDS = {
    "pm25": {"low": 12, "moderate": 35, "high": 55},
    "pm10": {"low": 54, "moderate": 154, "high": 254},
    "o3": {"low": 0.054, "moderate": 0.070, "high": 0.085},
    "no2": {"low": 53, "moderate": 100, "high": 360},
    "so2": {"low": 35, "moderate": 75, "high": 185},
    "co": {"low": 4.4, "moderate": 9.4, "high": 12.4},
}

# Units for each pollutant
UNITS = {
    "pm25": "µg/m³",
    "pm10": "µg/m³",
    "o3": "ppm",
    "no2": "ppb",
    "so2": "ppb",
    "co": "ppm",
}


class AirQualityExplainer:
    """Generates human-readable explanations using SHAP values."""
    
    def __init__(self, model=None, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or list(DISPLAY_NAMES.keys())
        self.shap_explainer = None
    
    def fit(self, X_train: np.ndarray):
        """Fit SHAP explainer on training data."""
        if self.model is None:
            return
        
        import shap
        self.shap_explainer = shap.TreeExplainer(self.model)
        # Warm up the explainer
        _ = self.shap_explainer.shap_values(X_train[:10])
    
    def get_shap_values(self, features: np.ndarray) -> np.ndarray:
        """Get SHAP values for features."""
        if self.shap_explainer is None:
            return None
        return self.shap_explainer.shap_values(features)
    
    def get_contributions(self, features: np.ndarray, predicted_class: int) -> Dict[str, float]:
        """Get feature contributions for a prediction."""
        if self.shap_explainer is None:
            return {}
        
        shap_values = self.get_shap_values(features)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Old format: list per class
            class_shap = shap_values[predicted_class][0]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                # New format: (samples, features, classes)
                class_shap = shap_values[0, :, predicted_class]
            else:
                class_shap = shap_values[0]
        else:
            return {}
        
        contributions = {}
        for i, name in enumerate(self.feature_names):
            val = class_shap[i]
            contributions[name] = float(val.item() if hasattr(val, 'item') else val)
        
        return contributions
    
    def generate_explanation(self, features: Dict, tier: int, 
                            contributions: Dict = None) -> str:
        """Generate human-readable explanation."""
        explanations = []
        
        # Main pollutants to explain
        main_pollutants = ["pm25", "pm10", "o3", "no2", "so2", "co"]
        
        # Sort by contribution if available
        if contributions:
            sorted_pollutants = sorted(
                main_pollutants,
                key=lambda x: abs(contributions.get(x, 0)),
                reverse=True
            )
        else:
            sorted_pollutants = main_pollutants
        
        # Generate explanation for top contributors
        for pollutant in sorted_pollutants[:3]:
            value = features.get(pollutant, 0)
            if value <= 0:
                continue
            
            thresholds = THRESHOLDS.get(pollutant, {})
            unit = UNITS.get(pollutant, "")
            name = DISPLAY_NAMES.get(pollutant, pollutant)
            
            if value > thresholds.get("high", float('inf')):
                level = "very high"
            elif value > thresholds.get("moderate", float('inf')):
                level = "elevated"
            elif value > thresholds.get("low", float('inf')):
                level = "moderate"
            else:
                level = "low"
            
            explanations.append(f"{name}: {value:.1f} {unit} ({level})")
        
        if not explanations:
            return "Insufficient data for detailed analysis."
        
        return " | ".join(explanations)
    
    def save(self, path: str):
        """Save explainer to file."""
        joblib.dump({
            "feature_names": self.feature_names,
            "shap_explainer": self.shap_explainer
        }, path)
    
    def load(self, path: str):
        """Load explainer from file."""
        data = joblib.load(path)
        self.feature_names = data.get("feature_names", self.feature_names)
        self.shap_explainer = data.get("shap_explainer")