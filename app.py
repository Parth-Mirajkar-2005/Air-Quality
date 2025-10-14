import streamlit as st
import pandas as pd
import joblib
import os
from feature_engineering import run_feature_engineering
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Air Quality Forecast Engine",
    page_icon="🌍",
    layout="wide",
)

# --- Color Scheme ---
COLORS = {"O3": "#1f77b4", "NO2": "#ff7f0e"}

# --- Functions ---
@st.cache_resource
def load_artifacts(site_id):
    """Load models, feature list, and validation metrics for a specific site."""
    model_dir = f'models/site_{site_id}'
    if not os.path.exists(model_dir):
        return None, None, None
    
    models = {
        'O3': joblib.load(f'{model_dir}/model_o3.pkl'),
        'NO2': joblib.load(f'{model_dir}/model_no2.pkl')
    }
    feature_cols = joblib.load(f'{model_dir}/feature_cols.pkl')
    
    try:
        metrics = joblib.load(f'{model_dir}/validation_metrics.pkl')
    except FileNotFoundError:
        metrics = None
        
    return models, feature_cols, metrics

def make_predictions(input_df, models, feature_cols):
    """Process data and generate the 3-column prediction DataFrame."""
    processed_df = run_feature_engineering(input_df, is_train=False)
    
    for col in feature_cols:
        if col not in processed_df.columns:
            processed_df[col] = 0
    X_unseen = processed_df[feature_cols]

    predictions = pd.DataFrame({
        'datetime': processed_df['datetime'],
        'O3_target': models['O3'].predict(X_unseen),
        'NO2_target': models['NO2'].predict(X_unseen)
    })
    
    predictions['O3_target'] = predictions['O3_target'].clip(lower=0)
    predictions['NO2_target'] = predictions['NO2_target'].clip(lower=0)
    
    return predictions

def create_prediction_plot(df, pollutant):
    """Creates a more attractive and INTERACTIVE Plotly chart."""
    pollutant_col = f'{pollutant}_target'
    fig = go.Figure()
    
    # Main forecast line
    fig.add_trace(go.Scatter(
        x=df['datetime'], 
        y=df[pollutant_col], 
        mode='lines', 
        line=dict(color=COLORS[pollutant], width=3), 
        name='Forecast'
    ))
    
    # Highlight the peak value on the chart
    if not df.empty:
        peak_value = df[pollutant_col].max()
        peak_time = df.loc[df[pollutant_col].idxmax(), 'datetime']
        fig.add_annotation(
            x=peak_time, y=peak_value,
            text=f"Peak: {peak_value:.2f}",
            font=dict(color="white", size=12),
            showarrow=True, arrowhead=2, arrowcolor=COLORS[pollutant],
            bordercolor=COLORS[pollutant], borderwidth=2, bgcolor="rgba(0,0,0,0.6)",
            ax=0, ay=-50
        )

    # --- INTERACTIVITY ENHANCEMENTS ---
    fig.update_layout(
        title=f'{pollutant} 24-Hour Forecast', 
        xaxis_title=None, # Title is now implicit with the interactive controls
        yaxis_title='Concentration (µg/m³)',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            # Add a rangeslider for intuitive zooming and panning
            rangeslider=dict(
                visible=True
            ),
            # Add quick-select buttons for common time ranges
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="Last 6h", step="hour", stepmode="backward"),
                    dict(count=12, label="Last 12h", step="hour", stepmode="backward"),
                    dict(step="all", label="24h")
                ]),
                y=1.15 # Position buttons above the plot
            )
        )
    )
    return fig

# --- UI ---
st.title("🌍 Air Quality Forecast Engine")
# st.markdown("An interactive dashboard to predict O₃ and NO₂ concentrations at various monitoring sites.")

# --- Sidebar ---
st.sidebar.title("⚙ Controls & Information")
st.sidebar.markdown("---")
try:
    # --- FIX 1: Changed d.split('') to d.split('_') to correctly find sites ---
    available_sites = sorted([int(d.split('_')[1]) for d in os.listdir('models') if d.startswith('site_')])
except FileNotFoundError:
    available_sites = []

if not available_sites:
    st.error("Model files not found! Please run `python train_and_save_models.py` from your terminal first.")
    st.stop()
    
selected_site = st.sidebar.selectbox("1. Select a Monitoring Site", options=available_sites, format_func=lambda x: f"Site {x}")

st.sidebar.markdown("---")
# --- FIX 2 & 3: Added missing underscores for clarity and consistency ---
uploaded_file = st.sidebar.file_uploader(f"2. Upload  data like 'site_{selected_site}_unseen_input_data.csv'", type="csv", key=f"uploader_{selected_site}")
st.sidebar.markdown("---")

# --- Main Content ---
st.header(f"📍 Forecast Dashboard for Site {selected_site}")
models, feature_cols, metrics = load_artifacts(selected_site)

if models is None:
    st.error(f"Models for Site {selected_site} could not be loaded. Please check the 'models' directory.")
    st.stop()

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    
    with st.spinner(f'Forecasting for Site {selected_site}...'):
        predictions_df = make_predictions(input_df, models, feature_cols)
        
        # --- VISUAL FORECASTS ---
        st.subheader("📈 Visual Forecasts")
        st.info("NOTE : You can pan, zoom, and use the buttons or the rangeslider at the bottom to explore the forecast.")
        tab1, tab2 = st.tabs(["O₃ (Ozone) Forecast", "NO₂ (Nitrogen Dioxide) Forecast"])
        with tab1:
            st.plotly_chart(create_prediction_plot(predictions_df, 'O3'), use_container_width=True)
        with tab2:
            st.plotly_chart(create_prediction_plot(predictions_df, 'NO2'), use_container_width=True)
        
        # --- EXPANDABLE SECTIONS FOR DETAILS ---
        with st.expander("📄 View Detailed Prediction Data and Download", expanded= True):
            st.dataframe(predictions_df)
            csv = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full 24-Hour Forecast (CSV)",
                data=csv,
                file_name=f'site_{selected_site}_predictions.csv',
                mime='text/csv'
            )

        if metrics:
            with st.expander("📌 View Model Performance Metrics"):
                metrics_df = pd.DataFrame({
                    'Pollutant': ['O₃', 'NO₂'],
                    'Accurecy': [f"{metrics['O3']['R2'] * 100.0 :.2f} %", f"{metrics['NO2']['R2'] * 100.0 :.2f} %"],
                    'RMSE (µg/m³)': [f"{metrics['O3']['RMSE']:.2f}", f"{metrics['NO2']['RMSE']:.2f}"],
                    'RIA': [f"{metrics['O3']['RIA']:.3f}", f"{metrics['NO2']['RIA']:.3f}"]
                }).set_index('Pollutant')
                st.table(metrics_df)
                st.caption("Metrics are calculated on the internal validation set during model training.")

else:
    st.info(f"Please select a site and upload its corresponding data file to begin the forecast.")