
# 🌬️ Air Quality Health Risk Classifier

Predict health risk levels from air quality data using XGBoost with SHAP explainability.

## Features

* **Real-time Data** : Fetches air quality from 30,000+ OpenAQ monitoring stations worldwide
* **Location Search** : Enter any city, address, or landmark
* **Health Risk Tiers** : 6 EPA-standard health categories (Good → Hazardous)
* **SHAP Explanations** : Understand which pollutants contribute most to the risk
* **Interactive Visualizations** : AQI gauge, pollutant contribution charts, and maps

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get OpenAQ API Key

1. Register at: https://explore.openaq.org/register
2. Get your API key from: https://explore.openaq.org/account
3. Create `.env` file:
   ```
   OPENAQ_API_KEY=your_api_key_here
   ```

### 3. Collect Real-World Training Data (Optional but Recommended)

```bash
python collect_data.py
```

This fetches data from 27 major cities worldwide and saves to `data/air_quality_training.csv`.

### 4. Train the Model

```bash
python train.py
```

* If real data exists in `data/air_quality_training.csv`, it uses that
* Otherwise, it generates synthetic training data
* Either way, the model will work correctly

### 5. Run the App

```bash
python app.py
```

Open http://localhost:7860 in your browser.

## Project Structure

```
air-quality-classifier/
├── app.py                 # Gradio web interface
├── train.py               # Model training script
├── collect_data.py        # Real-world data collection
├── requirements.txt       # Dependencies
├── .env.example           # Environment variable template
├── src/
│   ├── data_fetcher.py    # OpenAQ API client
│   ├── feature_engine.py  # Feature engineering & AQI calculations
│   └── explainer.py       # SHAP explainability
├── data/
│   └── air_quality_training.csv  # Training data (after collection)
└── model/
    ├── xgboost_model.pkl  # Trained model (after training)
    └── shap_explainer.pkl # SHAP explainer (after training)
```

## Health Tiers

| Tier               | AQI Range | Health Impact                    |
| ------------------ | --------- | -------------------------------- |
| 0 - Good           | 0-50      | Satisfactory, little to no risk  |
| 1 - Moderate       | 51-100    | Acceptable for most              |
| 2 - USG            | 101-150   | Sensitive groups may be affected |
| 3 - Unhealthy      | 151-200   | Everyone may experience effects  |
| 4 - Very Unhealthy | 201-300   | Health alert for everyone        |
| 5 - Hazardous      | 301-500   | Emergency conditions             |

## Tech Stack

* **ML** : XGBoost, SHAP, scikit-learn
* **Data** : OpenAQ API, Nominatim geocoding
* **UI** : Gradio, Plotly
* **Python 3.10+**

## License

MIT
