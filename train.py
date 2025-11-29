"""
Training Script - Train XGBoost classifier on air quality data
Supports both real data (from CSV) and synthetic data
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

from src.feature_engine import FEATURE_NAMES, calculate_aqi, aqi_to_tier, prepare_features
from src.explainer import AirQualityExplainer


def load_real_data(csv_path: str = "data/air_quality_training.csv") -> pd.DataFrame:
    """Load real data from CSV and prepare for training."""
    print(f"Loading data from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records")
    
    prepared_data = []
    
    for _, row in df.iterrows():
        measurements = {
            "pm25": row.get("pm25", 0) or 0,
            "pm10": row.get("pm10", 0) or 0,
            "o3": row.get("o3", 0) or 0,
            "no2": row.get("no2", 0) or 0,
            "so2": row.get("so2", 0) or 0,
            "co": row.get("co", 0) or 0,
        }
        
        # Skip if no PM2.5 (primary pollutant)
        if measurements["pm25"] <= 0:
            continue
        
        # Calculate AQI and tier
        aqi, _ = calculate_aqi(measurements)
        tier = aqi_to_tier(aqi)
        
        # Random hour for time features
        hour = np.random.randint(0, 24)
        
        # Prepare features
        features, _ = prepare_features(measurements, hour)
        
        # Add to data
        prepared_row = list(features[0]) + [tier, aqi]
        prepared_data.append(prepared_row)
    
    columns = FEATURE_NAMES + ["tier", "aqi"]
    result_df = pd.DataFrame(prepared_data, columns=columns)
    
    return result_df


def generate_synthetic_data(n_samples: int = 12000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic air quality data."""
    np.random.seed(seed)
    
    data = []
    
    # Define pollution scenarios with realistic distributions
    scenarios = [
        # (name, pm25_range, pm10_range, o3_range, probability)
        ("clean", (2, 12), (5, 50), (0.01, 0.05), 0.25),
        ("moderate", (12, 35), (30, 100), (0.04, 0.07), 0.30),
        ("usg", (35, 55), (80, 180), (0.06, 0.085), 0.20),
        ("unhealthy", (55, 150), (150, 300), (0.07, 0.10), 0.15),
        ("very_unhealthy", (150, 250), (250, 400), (0.08, 0.15), 0.07),
        ("hazardous", (250, 500), (350, 600), (0.10, 0.20), 0.03),
    ]
    
    for _ in range(n_samples):
        # Pick scenario based on probability
        probs = [s[4] for s in scenarios]
        scenario_idx = np.random.choice(len(scenarios), p=probs)
        scenario = scenarios[scenario_idx]
        
        # Generate pollutant values
        pm25 = np.random.uniform(*scenario[1])
        pm10 = max(pm25 * np.random.uniform(1.2, 2.5), np.random.uniform(*scenario[2]))
        o3 = np.random.uniform(*scenario[3])
        
        # Secondary pollutants (correlated with primary)
        no2 = np.random.uniform(5, 50) + pm25 * np.random.uniform(0.5, 1.5)
        so2 = np.random.uniform(2, 30) + pm25 * np.random.uniform(0.2, 0.8)
        co = np.random.uniform(0.1, 2) + pm25 * np.random.uniform(0.01, 0.05)
        
        # Random hour
        hour = np.random.randint(0, 24)
        
        # Create measurements dict
        measurements = {
            "pm25": pm25, "pm10": pm10, "o3": o3,
            "no2": no2, "so2": so2, "co": co
        }
        
        # Calculate AQI and tier
        aqi, _ = calculate_aqi(measurements)
        tier = aqi_to_tier(aqi)
        
        # Prepare features
        features, _ = prepare_features(measurements, hour)
        
        # Add to data
        row = list(features[0]) + [tier, aqi]
        data.append(row)
    
    # Create DataFrame
    columns = FEATURE_NAMES + ["tier", "aqi"]
    df = pd.DataFrame(data, columns=columns)
    
    return df


def train_model(use_real_data: bool = True):
    """Train the XGBoost classifier."""
    print("=" * 60)
    print("AIR QUALITY CLASSIFIER - TRAINING")
    print("=" * 60)
    
    # Load data
    csv_path = Path("data/air_quality_training.csv")
    
    if use_real_data and csv_path.exists():
        print("\n[1/4] Loading real-world training data...")
        df = load_real_data(str(csv_path))
        
        # If not enough real data, supplement with synthetic
        if len(df) < 1000:
            print(f"  Only {len(df)} real samples, adding synthetic data...")
            synthetic_df = generate_synthetic_data(n_samples=10000 - len(df))
            df = pd.concat([df, synthetic_df], ignore_index=True)
    else:
        print("\n[1/4] Generating synthetic training data...")
        print("  (Run collect_data.py first to use real data)")
        df = generate_synthetic_data(n_samples=12000)
    
    print(f"  Total samples: {len(df)}")
    print(f"  Tier distribution:\n{df['tier'].value_counts().sort_index()}")
    
    # Prepare features and labels
    X = df[FEATURE_NAMES].values
    y = df["tier"].values
    
    # Split data
    print("\n[2/4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Calculate class weights
    class_counts = np.bincount(y_train, minlength=6)
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    sample_weights = class_weights[y_train]
    
    # Train XGBoost
    print("\n[3/4] Training XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        early_stopping_rounds=15
    )
    
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    tier_names = ["Good", "Moderate", "USG", "Unhealthy", "V.Unhealthy", "Hazardous"]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=tier_names, zero_division=0))
    
    # Feature importance
    print("\nFeature Importance:")
    importance = model.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")
    
    # Create SHAP explainer
    print("\n[4/4] Creating SHAP explainer...")
    explainer = AirQualityExplainer(model=model, feature_names=FEATURE_NAMES)
    explainer.fit(X_train)
    
    # Test explainer
    sample_idx = np.random.randint(len(X_test))
    sample = X_test[sample_idx:sample_idx+1]
    contributions = explainer.get_contributions(sample, int(y_pred[sample_idx]))
    print(f"  Sample SHAP contributions: {contributions}")
    
    # Save model and explainer
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    joblib.dump(model, model_dir / "xgboost_model.pkl")
    explainer.save(model_dir / "shap_explainer.pkl")
    
    # Save training info
    training_info = {
        "samples": len(df),
        "accuracy": accuracy,
        "data_source": "real" if use_real_data and csv_path.exists() else "synthetic",
        "trained_at": pd.Timestamp.now().isoformat()
    }
    joblib.dump(training_info, model_dir / "training_info.pkl")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model saved to: model/xgboost_model.pkl")
    print(f"  Explainer saved to: model/shap_explainer.pkl")
    print(f"  Data source: {training_info['data_source']}")
    print(f"  Final accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    train_model(use_real_data=True)