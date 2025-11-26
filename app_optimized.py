import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DC Crime Insight",
    page_icon="🛡️",
    layout="wide"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load("optimized_model.pkl")
        options = joblib.load("app_options.pkl")
        return model, options
    except FileNotFoundError:
        st.error("Please run 'train_optimized.py' first to generate model files.")
        st.stop()

model, options = load_resources()

# --- HEADER ---
st.title("🛡️ DC Crime Risk Predictor")
st.markdown("### Intelligent Analysis Based on Location & Time")
st.markdown("---")

# --- SIDEBAR (INPUTS) ---
with st.sidebar:
    st.header("📍 Context Settings")
    
    # Neighborhood Dropdown
    selected_cluster = st.selectbox(
        "Select Neighborhood", 
        options["clusters"], 
        index=0,
        help="Choose the specific neighborhood cluster."
    )
    
    st.header("🕒 Time Settings")
    
    # Time Sliders and Dropdowns
    selected_day = st.selectbox("Day of Week", options["days"])
    selected_month = st.selectbox("Month", options["months"])
    selected_hour = st.slider("Hour of Day", 0, 23, 12, format="%d:00")

    predict_btn = st.button("Analyze Risk 🔍")

# --- MAIN DASHBOARD ---
if predict_btn:
    # 1. Prepare Input
    input_data = pd.DataFrame([{
        "NEIGHBORHOOD_CLUSTER": selected_cluster,
        "HOUR_OF_DAY": selected_hour,
        "DAY_OF_WEEK": selected_day,
        "MONTH_NAME": selected_month
    }])

    # 2. Get Prediction
    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    classes = model.classes_

    # 3. Visualize Results
    
    # Layout: Top Prediction on Left, Chart on Right
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ Primary Risk</h3>
            <h1 style="color: #ff4b4b;">{prediction}</h1>
            <p>Most likely incident for this context.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate highest probability
        max_prob = np.max(probs) * 100
        st.info(f"**Confidence Level:** {max_prob:.1f}%")

    with col2:
        # Create Dataframe for Plotly
        risk_df = pd.DataFrame({
            "Crime Type": classes,
            "Probability": probs * 100
        }).sort_values(by="Probability", ascending=True)

        # Plotly Bar Chart
        fig = px.bar(
            risk_df, 
            x="Probability", 
            y="Crime Type", 
            orientation='h',
            title="Risk Probability Distribution",
            text_auto='.1f',
            color="Probability",
            color_continuous_scale="Reds"
        )
        fig.update_layout(xaxis_title="Probability (%)", yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

else:
    # Default State
    st.info("👈 Please select a Neighborhood and Time on the sidebar, then click 'Analyze Risk'.")
    
    # Just a placeholder visual
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Map_of_Washington_DC_Neighborhood_Clusters.svg/1200px-Map_of_Washington_DC_Neighborhood_Clusters.svg.png", width=400, caption="DC Neighborhood Clusters")