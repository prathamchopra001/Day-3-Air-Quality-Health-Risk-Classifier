"""
Air Quality Health Risk Classifier - Gradio Web Interface
UI based on wireframe design
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import numpy as np
import joblib
import plotly.graph_objects as go

from src.data_fetcher import OpenAQClient, geocode
from src.feature_engine import (
    FEATURE_NAMES, HEALTH_TIERS,
    calculate_aqi, aqi_to_tier, prepare_features, get_recommendations
)
from src.explainer import AirQualityExplainer, DISPLAY_NAMES, UNITS

# Paths
MODEL_PATH = Path("model/xgboost_model.pkl")
EXPLAINER_PATH = Path("model/shap_explainer.pkl")

# Global model and explainer
model = None
explainer = None


def load_model():
    """Load trained model and explainer."""
    global model, explainer
    
    if os.environ.get("OPENAQ_API_KEY"):
        print("✓ OpenAQ API key found")
    else:
        print("⚠ OPENAQ_API_KEY not set")
        print("  Get your key at: https://explore.openaq.org/register")
    
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print("✓ Model loaded")
    else:
        print("✗ Model not found - run train.py first")
    
    if EXPLAINER_PATH.exists():
        explainer = AirQualityExplainer(model=model, feature_names=FEATURE_NAMES)
        explainer.load(EXPLAINER_PATH)
        print("✓ SHAP explainer loaded")
    else:
        explainer = AirQualityExplainer(model=model, feature_names=FEATURE_NAMES)
        print("⚠ SHAP explainer not found")


def create_gauge(aqi: float, tier: int) -> go.Figure:
    """Create AQI gauge chart."""
    tier_info = HEALTH_TIERS.get(tier, HEALTH_TIERS[0])
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        number={"font": {"size": 60, "color": tier_info["color"]}, "suffix": ""},
        title={"text": f"<b>{tier_info['name']}</b>", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 2, "tickcolor": "#666"},
            "bar": {"color": tier_info["color"], "thickness": 0.75},
            "bgcolor": "#f0f0f0",
            "borderwidth": 2,
            "bordercolor": "#ccc",
            "steps": [
                {"range": [0, 50], "color": "#00E400"},
                {"range": [50, 100], "color": "#FFFF00"},
                {"range": [100, 150], "color": "#FF7E00"},
                {"range": [150, 200], "color": "#FF0000"},
                {"range": [200, 300], "color": "#8F3F97"},
                {"range": [300, 500], "color": "#7E0023"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.8,
                "value": aqi
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial"}
    )
    return fig


def create_shap_chart(contributions: dict) -> go.Figure:
    """Create SHAP contribution bar chart."""
    main = ["pm25", "pm10", "o3", "no2", "so2", "co"]
    filtered = {k: v for k, v in contributions.items() if k in main and abs(v) > 0.001}
    
    if not filtered:
        fig = go.Figure()
        fig.add_annotation(text="No significant pollutant impacts", 
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font={"size": 14, "color": "#666"})
        fig.update_layout(height=180, margin=dict(l=20, r=20, t=20, b=20))
        return fig
    
    sorted_items = sorted(filtered.items(), key=lambda x: abs(x[1]), reverse=True)
    
    names = [DISPLAY_NAMES.get(k, k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = ["#e74c3c" if v > 0 else "#27ae60" for v in values]
    
    fig = go.Figure(go.Bar(
        y=names, x=values, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont={"size": 11}
    ))
    
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=70, t=10, b=20),
        xaxis_title="Impact on Risk (+ increases, - decreases)",
        xaxis={"zeroline": True, "zerolinecolor": "#ccc", "zerolinewidth": 2},
        yaxis={"automargin": True},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_map(lat: float = None, lon: float = None, name: str = None, 
               color: str = "#3498db") -> go.Figure:
    """Create location map."""
    fig = go.Figure()
    
    center_lat = lat if lat else 20
    center_lon = lon if lon else 0
    zoom = 11 if lat else 1
    
    if lat and lon:
        fig.add_trace(go.Scattermap(
            lat=[lat], lon=[lon],
            mode="markers",
            marker=dict(size=18, color=color),
            hoverinfo="text",
            hovertext=name or "Selected Location"
        ))
    
    fig.update_layout(
        map=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        height=280,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    return fig


def format_readings(measurements: dict, station: str = "") -> str:
    """Format readings as markdown."""
    if not measurements:
        return "*No readings available*"
    
    lines = []
    if station:
        lines.append(f"**📍 Station:** {station}\n")
    
    for poll, val in measurements.items():
        unit = UNITS.get(poll, "")
        lines.append(f"**{poll.upper()}:** {val:.1f} {unit}")
    
    return "\n\n".join(lines)


def format_analysis(tier: int, aqi: float, confidence: float, explanation: str, location: str = "") -> str:
    """Format analysis as markdown."""
    tier_info = HEALTH_TIERS.get(tier, HEALTH_TIERS[0])
    
    loc_line = f"**📍 {location}**\n\n" if location else ""
    
    return f"""{loc_line}### 🎯 {tier_info['name']}

**AQI:** {aqi:.0f} &nbsp;|&nbsp; **Confidence:** {confidence:.1f}%

---

{explanation}"""


def format_recommendations(recs: list) -> str:
    """Format recommendations as markdown."""
    if not recs:
        return "*No recommendations available*"
    
    lines = ["### 💡 Health Recommendations\n"]
    for rec in recs:
        lines.append(f"✓ {rec}")
    
    return "\n\n".join(lines)


def predict_location(location_query: str):
    """Get prediction for a location."""
    empty_outputs = (
        "*Enter a location to search*",
        "", "", create_map(), None, None
    )
    
    if not location_query or len(location_query.strip()) < 2:
        return empty_outputs
    
    if model is None:
        return (
            "⚠️ **Model not loaded.** Run `python train.py` first.",
            "", "", create_map(), None, None
        )
    
    # Geocode location
    geo = geocode(location_query)
    if "error" in geo:
        return (
            f"⚠️ **Location Error:** {geo['error']}",
            "", "", create_map(), None, None
        )
    
    lat, lon = geo["lat"], geo["lon"]
    location_name = geo["name"]
    short_name = ", ".join(location_name.split(",")[:2])
    
    # Fetch air quality data
    client = OpenAQClient()
    result = client.get_measurements(lat, lon)
    client.close()
    
    if "error" in result:
        return (
            f"⚠️ **{result['error']}**\n\nTry Manual Input tab instead.",
            "",
            "",
            create_map(lat, lon, short_name, "#e74c3c"),
            None,
            None
        )
    
    measurements = result["measurements"]
    station = result["station_name"]
    
    # Prepare features and predict
    features, feature_dict = prepare_features(measurements)
    aqi, dominant = calculate_aqi(measurements)
    
    pred_tier = int(model.predict(features)[0])
    pred_proba = model.predict_proba(features)[0]
    confidence = pred_proba[pred_tier] * 100
    
    tier_info = HEALTH_TIERS[pred_tier]
    
    # Get SHAP contributions
    contributions = {}
    if explainer and explainer.shap_explainer:
        contributions = explainer.get_contributions(features, pred_tier)
    
    # Generate explanation
    explanation = ""
    if explainer:
        explanation = explainer.generate_explanation(feature_dict, pred_tier, contributions)
    
    # Format outputs
    analysis = format_analysis(pred_tier, aqi, confidence, explanation, short_name)
    readings = format_readings(measurements, station)
    recommendations = format_recommendations(get_recommendations(pred_tier))
    map_fig = create_map(lat, lon, short_name, tier_info["color"])
    gauge_fig = create_gauge(aqi, pred_tier)
    shap_fig = create_shap_chart(contributions)
    
    return analysis, readings, recommendations, map_fig, gauge_fig, shap_fig


def predict_manual(pm25, pm10, o3, no2, so2, co):
    """Predict from manual input."""
    if model is None:
        return ("⚠️ **Model not loaded.** Run `python train.py` first.", "", "", None, None)
    
    # Build measurements dict
    measurements = {}
    if pm25 > 0: measurements["pm25"] = pm25
    if pm10 > 0: measurements["pm10"] = pm10
    if o3 > 0: measurements["o3"] = o3
    if no2 > 0: measurements["no2"] = no2
    if so2 > 0: measurements["so2"] = so2
    if co > 0: measurements["co"] = co
    
    if not measurements:
        return ("⚠️ *Enter at least one pollutant value*", "", "", None, None)
    
    # Prepare and predict
    features, feature_dict = prepare_features(measurements)
    aqi, _ = calculate_aqi(measurements)
    
    pred_tier = int(model.predict(features)[0])
    pred_proba = model.predict_proba(features)[0]
    confidence = pred_proba[pred_tier] * 100
    
    # SHAP contributions
    contributions = {}
    if explainer and explainer.shap_explainer:
        contributions = explainer.get_contributions(features, pred_tier)
    
    # Generate explanation
    explanation = ""
    if explainer:
        explanation = explainer.generate_explanation(feature_dict, pred_tier, contributions)
    
    # Format outputs
    analysis = format_analysis(pred_tier, aqi, confidence, explanation)
    readings = format_readings(measurements)
    recommendations = format_recommendations(get_recommendations(pred_tier))
    gauge_fig = create_gauge(aqi, pred_tier)
    shap_fig = create_shap_chart(contributions)
    
    return analysis, readings, recommendations, gauge_fig, shap_fig


def create_app():
    """Create Gradio interface matching the wireframe."""
    
    with gr.Blocks(title="Air Quality Health Risk Classifier") as app:
        
        # Title
        gr.Markdown("""
        <h1 style="text-align: center; margin-bottom: 5px;">
            🌬️ AIR QUALITY HEALTH RISK CLASSIFIER
        </h1>
        """)
        
        with gr.Tabs() as tabs:
            
            # ==================== BY LOCATION TAB ====================
            with gr.TabItem("📍 By Location", id=0):
                
                # Search input
                with gr.Row():
                    location_input = gr.Textbox(
                        label="",
                        placeholder="🔍 Enter city, address, or landmark...",
                        scale=4,
                        container=False
                    )
                    search_btn = gr.Button("Search", variant="primary", scale=1)
                
                gr.Markdown("---")
                
                # Row 1: Analysis + Readings | Health Recommendations
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        gr.Markdown("**Analysis:**")
                        loc_analysis = gr.Markdown(value="*Enter a location to search*")
                        gr.Markdown("**Readings:**")
                        loc_readings = gr.Markdown(value="")
                    
                    with gr.Column(scale=1):
                        loc_recommendations = gr.Markdown(value="")
                
                # Row 2: Map | AQI Gauge
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        gr.Markdown("**MAP**")
                        loc_map = gr.Plot(value=create_map(), show_label=False)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("**AQI**")
                        loc_gauge = gr.Plot(show_label=False)
                
                # Row 3: Pollutant Impact (full width)
                gr.Markdown("**Pollutant Impact**")
                loc_shap = gr.Plot(show_label=False)
                
                # Event handlers
                search_btn.click(
                    predict_location,
                    inputs=[location_input],
                    outputs=[loc_analysis, loc_readings, loc_recommendations, 
                            loc_map, loc_gauge, loc_shap]
                )
                location_input.submit(
                    predict_location,
                    inputs=[location_input],
                    outputs=[loc_analysis, loc_readings, loc_recommendations,
                            loc_map, loc_gauge, loc_shap]
                )
            
            # ==================== MANUAL INPUT TAB ====================
            with gr.TabItem("📝 Manual Input", id=1):
                
                # Pollutant inputs
                gr.Markdown("**Enter pollutant values** (leave 0 for unknown):")
                
                with gr.Row():
                    pm25_in = gr.Number(label="PM2.5 (µg/m³)", value=0, minimum=0)
                    pm10_in = gr.Number(label="PM10 (µg/m³)", value=0, minimum=0)
                    o3_in = gr.Number(label="O₃ (ppm)", value=0, minimum=0, step=0.001)
                
                with gr.Row():
                    no2_in = gr.Number(label="NO₂ (ppb)", value=0, minimum=0)
                    so2_in = gr.Number(label="SO₂ (ppb)", value=0, minimum=0)
                    co_in = gr.Number(label="CO (ppm)", value=0, minimum=0, step=0.1)
                
                analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                
                gr.Markdown("---")
                
                # Row 1: Analysis + Readings | Health Recommendations
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        gr.Markdown("**Analysis:**")
                        man_analysis = gr.Markdown(value="*Enter pollutant values above*")
                        gr.Markdown("**Readings:**")
                        man_readings = gr.Markdown(value="")
                    
                    with gr.Column(scale=1):
                        man_recommendations = gr.Markdown(value="")
                
                # Row 2: AQI Gauge (centered)
                gr.Markdown("**AQI**")
                man_gauge = gr.Plot(show_label=False)
                
                # Row 3: Pollutant Impact
                gr.Markdown("**Pollutant Impact**")
                man_shap = gr.Plot(show_label=False)
                
                # Event handler
                analyze_btn.click(
                    predict_manual,
                    inputs=[pm25_in, pm10_in, o3_in, no2_in, so2_in, co_in],
                    outputs=[man_analysis, man_readings, man_recommendations,
                            man_gauge, man_shap]
                )
        
        # Footer
        gr.Markdown("""
        ---
        <center>
        <small>
        Powered by XGBoost + SHAP | Data from <a href="https://openaq.org" target="_blank">OpenAQ</a> | 
        ⚠️ For informational purposes only
        </small>
        </center>
        """)
    
    return app


# Initialize
load_model()

# Create app
app = create_app()

if __name__ == "__main__":
    app.launch()