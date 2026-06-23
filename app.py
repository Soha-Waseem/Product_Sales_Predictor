import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Sales Predictor Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM CUSTOM CSS STYLES ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

/* Apply font to the entire Streamlit App */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Custom Glassmorphism Metric Cards */
.metric-card {
    background: rgba(30, 41, 59, 0.45);
    border-radius: 14px;
    padding: 22px;
    border: 1px solid rgba(99, 102, 241, 0.2);
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.15);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    text-align: center;
    margin-bottom: 10px;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99, 102, 241, 0.6);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.2);
}
.metric-title {
    font-size: 0.95rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 10px;
}
.metric-value {
    font-size: 2.3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #c7d2fe 0%, #818cf8 50%, #4f46e5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.metric-value-green {
    font-size: 2.3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a7f3d0 0%, #34d399 50%, #059669 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.metric-value-gold {
    font-size: 2.3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #fef08a 0%, #fbbf24 50%, #d97706 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.metric-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 8px;
    font-weight: 400;
}

/* Custom badge styling */
.badge {
    background: rgba(99, 102, 241, 0.15);
    color: #c7d2fe;
    border: 1px solid rgba(99, 102, 241, 0.3);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
    margin-top: 5px;
}

/* Subtitle dashboard title custom spacing */
.dashboard-header {
    margin-bottom: 25px;
    padding-bottom: 15px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)


# --- DATA & MODEL CACHING ---
@st.cache_resource
def get_trained_model():
    # Load dataset
    df = pd.read_csv("advertising.csv")
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    # Model parameters matching Polynomial_Regression.ipynb
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Collect data stats
    stats = {
        'TV': {'min': float(df['TV'].min()), 'max': float(df['TV'].max()), 'mean': float(df['TV'].mean())},
        'Radio': {'min': float(df['Radio'].min()), 'max': float(df['Radio'].max()), 'mean': float(df['Radio'].mean())},
        'Newspaper': {'min': float(df['Newspaper'].min()), 'max': float(df['Newspaper'].max()), 'mean': float(df['Newspaper'].mean())},
    }
    
    return model, poly, r2, rmse, stats, df

# Load the trained components
model, poly, r2, rmse, stats, df = get_trained_model()


# --- HEADER ---
st.markdown("""
<div class="dashboard-header">
    <h1 style="margin: 0; font-weight: 700; color: #f8fafc; font-size: 2.6rem;">📈 Sales Predictor AI Dashboard</h1>
    <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 1.1rem; font-weight: 300;">
        Interactive Budget Simulation & Performance Modeling using Quadratic Polynomial Regression (Degree 2)
    </p>
</div>
""", unsafe_allow_html=True)


# --- SESSION STATE FOR SYNCED INPUTS ---
if 'tv' not in st.session_state:
    st.session_state.tv = 150.0
if 'radio' not in st.session_state:
    st.session_state.radio = 25.0
if 'newspaper' not in st.session_state:
    st.session_state.newspaper = 30.0

def update_inputs(tv_val, radio_val, newspaper_val):
    st.session_state.tv = tv_val
    st.session_state.radio = radio_val
    st.session_state.newspaper = newspaper_val


# --- SIDEBAR INPUTS & PRESETS ---
with st.sidebar:
    st.markdown("### 🛠️ Simulation Controls")
    st.write("Adjust marketing channels below to predict sales volume.")
    
    # Preset Mix Buttons
    st.markdown("#### Preset Strategies")
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        if st.button("⚖️ Balanced Mix"):
            update_inputs(150.0, 25.0, 30.0)
        if st.button("📻 Digital/Radio Focus"):
            update_inputs(40.0, 45.0, 15.0)
    with col_pre2:
        if st.button("📺 TV Heavy"):
            update_inputs(280.0, 10.0, 10.0)
        if st.button("🌱 Bootstrapped"):
            update_inputs(15.0, 5.0, 5.0)

    st.markdown("---")
    
    # Channel Sliders
    st.markdown("#### Custom Allocation ($K)")
    
    # TV Slider
    st.slider(
        "TV Advertising Budget",
        min_value=0.0,
        max_value=350.0,
        step=0.5,
        key="tv",
        help="Investment in TV commercials and placement."
    )
    
    # Radio Slider
    st.slider(
        "Radio Advertising Budget",
        min_value=0.0,
        max_value=100.0,
        step=0.5,
        key="radio",
        help="Investment in Radio and Audio Streaming ads."
    )
    
    # Newspaper Slider
    st.slider(
        "Newspaper Advertising Budget",
        min_value=0.0,
        max_value=150.0,
        step=0.5,
        key="newspaper",
        help="Investment in print, local publications, and newsletter ads."
    )
    
    st.markdown("---")
    # Quick Summary inside Sidebar
    total_budget = st.session_state.tv + st.session_state.radio + st.session_state.newspaper
    st.metric("Total Budget Allocated", f"${total_budget:,.1f} K")


# --- CALCULATE INSTANT PREDICTION ---
input_data = pd.DataFrame({
    'TV': [st.session_state.tv],
    'Radio': [st.session_state.radio],
    'Newspaper': [st.session_state.newspaper]
})
input_poly = poly.transform(input_data)
predicted_sales = model.predict(input_poly)[0]

# Clip prediction at 0 just in case polynomial curve goes negative on edge cases
predicted_sales = max(0.0, predicted_sales)


# --- KPI METRICS DISPLAY ---
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

# KPI 1: Predicted Sales Volume
with kpi_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Predicted Sales Volume</div>
        <div class="metric-value-green">{predicted_sales:,.2f} K</div>
        <div class="metric-subtitle">Thousands of Units predicted</div>
        <div class="badge">Quadratic Fit</div>
    </div>
    """, unsafe_allow_html=True)

# KPI 2: Total Marketing Spend
with kpi_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Campaign Investment</div>
        <div class="metric-value">${total_budget:,.1f} K</div>
        <div class="metric-subtitle">TV + Radio + Newspaper spend</div>
        <div class="badge">Simulated Budget</div>
    </div>
    """, unsafe_allow_html=True)

# KPI 3: Efficiency Index (Sales generated per $1K spend)
efficiency = predicted_sales / (total_budget if total_budget > 0 else 1.0)
eff_label = "Optimal Mix" if efficiency > 0.08 else "Moderate Mix" if efficiency > 0.04 else "Low Efficiency Mix"
with kpi_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Efficiency Index</div>
        <div class="metric-value-gold">{efficiency:.3f}</div>
        <div class="metric-subtitle">Sales Units per $1K spent</div>
        <div class="badge">{eff_label}</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# --- TABS FOR VISUALIZATIONS ---
tab1, tab2, tab3 = st.tabs([
    "📈 Channel Response & Trajectories", 
    "🌌 3D Budget Landscape", 
    "📊 Data Exploration & Correlation"
])

# TAB 1: 2D Channel Curves (Holding other variables static)
with tab1:
    st.markdown("### Budget Sensitivity Curves")
    st.write("Understand the sales impact of changing one channel's budget while keeping others at your current allocation.")
    
    col_tab1_left, col_tab1_right = st.columns([1, 3])
    
    with col_tab1_left:
        selected_channel = st.radio(
            "Select Channel to Vary",
            ["TV", "Radio", "Newspaper"],
            help="This channel will span across its full historical range while other channels stay at your slider settings."
        )
        st.info(f"The yellow star on the plot indicates your current simulated mix.")
        
    with col_tab1_right:
        chan_min = stats[selected_channel]['min']
        chan_max = stats[selected_channel]['max']
        chan_range = np.linspace(0.0, chan_max * 1.2, 100)
        
        # Build temp dataframe to predict values
        temp_dict = {
            'TV': st.session_state.tv,
            'Radio': st.session_state.radio,
            'Newspaper': st.session_state.newspaper
        }
        
        temp_dfs = []
        for v in chan_range:
            d = temp_dict.copy()
            d[selected_channel] = v
            temp_dfs.append(d)
        
        curve_df = pd.DataFrame(temp_dfs)
        curve_poly = poly.transform(curve_df)
        curve_preds = model.predict(curve_poly)
        curve_preds = np.clip(curve_preds, 0.0, None) # clip negative predictions
        
        # Render Plotly Line Plot
        fig_curve = go.Figure()
        
        # Historical scatter points for visual baseline
        fig_curve.add_trace(go.Scatter(
            x=df[selected_channel],
            y=df['Sales'],
            mode='markers',
            name='Historical Data Points',
            marker=dict(color='rgba(148, 163, 184, 0.25)', size=5)
        ))
        
        # Polynomial Curve
        fig_curve.add_trace(go.Scatter(
            x=chan_range,
            y=curve_preds,
            mode='lines',
            name='Regression Trajectory',
            line=dict(color='#6366f1', width=3.5)
        ))
        
        # Current Scenario Point
        fig_curve.add_trace(go.Scatter(
            x=[st.session_state[selected_channel.lower()]],
            y=[predicted_sales],
            mode='markers',
            name='Your Selected Budget',
            marker=dict(
                color='#fbbf24',
                size=15,
                symbol='star',
                line=dict(color='#ffffff', width=1.5)
            )
        ))
        
        fig_curve.update_layout(
            xaxis_title=f"{selected_channel} Advertising Budget ($K)",
            yaxis_title="Sales Volume (K Units)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Outfit",
            font_color="#f8fafc",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=0, r=0, b=0, t=20),
            height=400
        )
        
        st.plotly_chart(fig_curve, use_container_width=True)

# TAB 2: Plotly 3D Scatter Visualizer
with tab2:
    st.markdown("### 3D Scatter Distribution vs. Simulation Scenario")
    st.write("Examine where your current mix sits compared to the general shape of our historical records.")
    
    fig_3d = go.Figure()
    
    # 3D Historical Cases
    fig_3d.add_trace(go.Scatter3d(
        x=df['TV'],
        y=df['Radio'],
        z=df['Sales'],
        mode='markers',
        name='Historical Campaign Records',
        marker=dict(
            size=4.5,
            color=df['Sales'],
            colorscale='Viridis',
            opacity=0.55
        )
    ))
    
    # 3D Current Selection marker
    fig_3d.add_trace(go.Scatter3d(
        x=[st.session_state.tv],
        y=[st.session_state.radio],
        z=[predicted_sales],
        mode='markers',
        name='Simulated Target Mix',
        marker=dict(
            size=11,
            color='#d97706',
            symbol='diamond',
            line=dict(color='#ffffff', width=2)
        )
    ))
    
    fig_3d.update_layout(
        scene=dict(
            xaxis_title="TV ($K)",
            yaxis_title="Radio ($K)",
            zaxis_title="Sales Volume"
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Outfit",
        font_color="#f8fafc",
        height=500
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)

# TAB 3: Heatmap and General Correlation Stats
with tab3:
    st.markdown("### Feature Correlation Analysis")
    st.write("Understand which variables have the highest linear relationships in the training dataset.")
    
    col_corr1, col_corr2 = st.columns([2, 1])
    
    with col_corr1:
        corr_matrix = df.corr()
        fig_heat = px.imshow(
            corr_matrix,
            text_auto=".3f",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            labels=dict(color="Correlation Score")
        )
        fig_heat.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Outfit",
            font_color="#f8fafc",
            margin=dict(l=0, r=0, b=0, t=20),
            height=380
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with col_corr2:
        st.markdown("#### Highlights & Key Insights")
        st.write("""
        - **TV Dominance**: TV spend displays a high linear correlation with sales volume.
        - **Radio Synergy**: Radio spends are also strongly positive, often proving highly synergistic when combined with TV.
        - **Newspaper Decay**: Newspaper displays the weakest correlation with Sales, indicating that it should be budgeted carefully.
        - **Non-Linear Interactions**: The model applies polynomial regression ($TV \\times Radio$, $TV^2$, etc.) to capture synergistic effects between platforms.
        """)


st.markdown("<br><hr>", unsafe_allow_html=True)


# --- MODEL METRICS & DATA WORKSPACE EXPANDER ---
with st.expander("🔬 Model Configuration & Historical Data Reference"):
    st.markdown("#### Regression Model Statistics (Test Dataset)")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="R² Score (Coefficient of Determination)", value=f"{r2:.5f}")
    with m_col2:
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.4f}")
    with m_col3:
        st.metric(label="Polynomial Degree", value="2 (Quadratic Interactions)")
        
    st.markdown("#### Sample Historical Campaign Data (advertising.csv)")
    st.dataframe(df, height=180)
    
    st.markdown("#### Descriptive Summary of Data")
    st.dataframe(df.describe(), height=150)
