import streamlit as st
import numpy as np
from pythermalcomfort.models import utci

st.set_page_config(page_title="Heat Stress Index Calculator", layout="wide")

# ---------- CSS STYLE ----------
st.markdown("""
<style>

.main{
background: linear-gradient(120deg,#f6f9fc,#e9f2ff);
}

.metric-card{
background:white;
padding:25px;
border-radius:15px;
box-shadow:0 4px 12px rgba(0,0,0,0.08);
text-align:center;

min-height:200px;     /* allow flexible height */
width:100%;           /* full column width */
display:flex;
flex-direction:column;
justify-content:center;
align-items:center;
gap:10px;

transition:0.3s;
}

.metric-card:hover{
transform:scale(1.03);
box-shadow:0 8px 20px rgba(0,0,0,0.15);
}

.metric-title{
font-size:20px;
font-weight:600;
color:#2c3e50;
}

.metric-value{
font-size:34px;
font-weight:700;
color:#34495e;
}

.metric-desc{
font-size:14px;
color:#7f8c8d;
}

.metric-label{
padding:6px 16px;
border-radius:12px;
color:white;
font-size:14px;
font-weight:500;
display:inline-block;
}

</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("""
<div style='text-align:center;padding:20px'>
<h1>🌡️ Heat Stress Index Calculator</h1>
<p>Human Thermal Comfort Assessment Using Environmental Parameters</p>
</div>
""", unsafe_allow_html=True)


# ---------- SIDEBAR ----------
st.sidebar.header("Select Index")

index_choice = st.sidebar.selectbox(
"Choose Index",
["Heat Index (HI)", "Apparent Temperature", "WBGT", "UTCI", "All Indices"]
)

st.sidebar.header("Environmental Inputs")

T = st.sidebar.number_input("Air Temperature (°C)", value=30.0)

RH = None
wind = None
solar = None

if index_choice in ["Heat Index (HI)", "Apparent Temperature", "WBGT", "UTCI", "All Indices"]:
    RH = st.sidebar.number_input("Relative Humidity (%)", value=60.0)

if index_choice in ["Apparent Temperature", "WBGT", "UTCI", "All Indices"]:
    wind = st.sidebar.number_input("Wind Speed (m/s)", value=2.0)

if index_choice in ["WBGT", "UTCI", "All Indices"]:
    solar = st.sidebar.number_input("Solar Radiation (W/m²)", value=500.0)

calculate = st.sidebar.button("Calculate")


# ---------- SAFETY CLASSIFICATION ----------
def classify(val, index):

    if index == "HI":
        if val < 27: return "Safe", "#2ecc71"
        elif val < 32: return "Caution", "#f1c40f"
        elif val < 41: return "Extreme Caution", "#e67e22"
        else: return "Danger", "#e74c3c"

    if index == "AT":
        if val < 28: return "Comfortable", "#2ecc71"
        elif val < 32: return "Warm", "#f1c40f"
        elif val < 38: return "Hot", "#e67e22"
        else: return "Very Hot", "#e74c3c"

    if index == "WBGT":
        if val < 27: return "Safe", "#2ecc71"
        elif val < 30: return "Moderate Risk", "#f1c40f"
        elif val < 32: return "High Risk", "#e67e22"
        else: return "Extreme Risk", "#e74c3c"

    if index == "UTCI":
        if val < 26: return "No heat stress", "#2ecc71"
        elif val < 32: return "Moderate stress", "#f1c40f"
        elif val < 38: return "Strong stress", "#e67e22"
        else: return "Very strong stress", "#e74c3c"


# ---------- CARD FUNCTION ----------
def show_card(title, value, unit, desc, index):

    label, color = classify(value, index)

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-title">{title}</div>

    <div class="metric-value">{value:.2f} {unit}</div>

    <div class="metric-desc">{desc}</div>

    <div>
        <span class="metric-label" style="background:{color}">
        {label}
        </span>
    </div>

    </div>
    """, unsafe_allow_html=True)


# ---------- CALCULATION ----------
if calculate:

    results = []
    selected = []

    if index_choice in ["Heat Index (HI)", "All Indices"]:
        selected.append("HI")

    if index_choice in ["Apparent Temperature", "All Indices"]:
        selected.append("AT")

    if index_choice in ["WBGT", "All Indices"]:
        selected.append("WBGT")

    if index_choice in ["UTCI", "All Indices"]:
        selected.append("UTCI")

    n = len(selected)

    if n == 1:
        cols = st.columns([1,2,1])
        display_cols = [cols[1]]
    else:
        cols = st.columns(n)
        display_cols = cols

    i = 0


    # ---------- HEAT INDEX ----------
    if "HI" in selected:

        HI = -8.784695 + 1.61139411*T + 2.338549*RH - 0.14611605*T*RH \
        -0.012308094*T*T -0.016424828*RH*RH +0.002211732*T*T*RH \
        +0.00072546*T*RH*RH -0.000003582*T*T*RH*RH

        results.append(HI)

        with display_cols[i]:
            show_card(
                "Heat Index",
                HI,
                "°C",
                "Perceived temperature considering humidity.",
                "HI"
            )
        i += 1


    # ---------- APPARENT TEMPERATURE ----------
    if "AT" in selected:

        AT = T + 0.33*(RH/100*6.105*np.exp((17.27*T)/(237.7+T))) -0.7*wind -4

        results.append(AT)

        with display_cols[i]:
            show_card(
                "Apparent Temperature",
                AT,
                "°C",
                "Perceived temperature including humidity and wind.",
                "AT"
            )
        i += 1


    # ---------- WBGT ----------
    if "WBGT" in selected:

        WBGT = 0.567*T + 0.393*(RH/100*6.105*np.exp((17.27*T)/(237.7+T))) + 0.002*solar + 3.94

        results.append(WBGT)

        with display_cols[i]:
            show_card(
                "WBGT",
                WBGT,
                "°C",
                "Index widely used for occupational heat stress.",
                "WBGT"
            )
        i += 1


    # ---------- UTCI ----------
    if "UTCI" in selected:

        tr = T + solar/100

        UTCI_val = utci(
            tdb=T,
            tr=tr,
            v=wind,
            rh=RH
        ).utci

        results.append(UTCI_val)

        with display_cols[i]:
            show_card(
                "UTCI",
                UTCI_val,
                "°C",
                "Universal Thermal Climate Index for human heat stress.",
                "UTCI"
            )


    # ---------- SUMMARY ----------
    st.markdown("---")

    avg = np.mean(results)

    st.markdown(f"""
    ### 📊 Summary

    Average thermal stress index value: **{avg:.2f} °C**

    Higher values indicate stronger **heat stress risk**.
    """)