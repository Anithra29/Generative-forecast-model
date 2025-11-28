import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import numpy as np
import re
import base64
import time

# ==========================================================
# SET PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Generative Forecast Explanation System", layout="wide")

# ==========================================================
# APPLY FULL-PAGE BACKGROUND IMAGE
# ==========================================================
def add_bg_from_local(image_path):
    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Global white text */
        h1, h2, h3, h4, h5, h6, p, label, span, div, .stMarkdown, .stMarkdown p, .stMarkdown li {{
            color: white !important;
            font-weight: 600 !important;
        }}

        /* TABLE text */
        .stDataFrame table tbody tr td {{
            color: white !important;
        }}

        .stDataFrame table thead tr th {{
            color: white !important;
            font-weight: 800 !important;
        }}

        /* FIX: Chatbot input box BLACK text + WHITE background */
        .stTextInput > div > div > input {{
            background-color: white !important;
            color: black !important;
            border-radius: 8px;
            padding: 8px;
        }}

        /* FIX: Slider labels stay white */
        .stSlider label {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("background/background.png")

# ==========================================================
# PAGE TITLE
# ==========================================================
st.title("📈 Generative Forecast Explanation System")

# ==========================================================
# UPLOAD CSV
# ==========================================================
uploaded = st.file_uploader("Upload your CSV (columns: ds, y, Customers, Promo, DayOfWeek)", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    df["ds"] = pd.to_datetime(df["ds"])

    # PREVIEW
    pretty_df = df.rename(columns={"ds": "Date", "y": "Sales"})
    st.subheader("📄 Data Preview")
    st.dataframe(pretty_df.head())

    # ==========================================================
    # PROPHET FORECASTING + EXECUTION TIME
    # ==========================================================
    st.header("Forecast")

    periods = st.slider("Days to Forecast", 30, 365, 90)

    start_time = time.time()

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(df)

    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)

    end_time = time.time()
    execution_time = end_time - start_time

    st.success(f"Model Execution Time: {execution_time:.2f} seconds")

    # Forecast Plots
    st.subheader("📈 Forecast Plot")
    st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)

    st.subheader("📉 Forecast Components")
    st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

    merged = pd.merge(forecast, df, on="ds", how="left")

    # ==========================================================
    # SMART WHY CHATBOT
    # ==========================================================
    st.header("🤖 Smart WHY Chatbot")

    st.markdown("""
    **Ask questions like:**
    - Why were sales low on **2015-07-10**?
    - Explain the spike on **2014-12-20**
    - Why high on this date?
    """)

    question = st.text_input("Ask your question:")

    def extract_date(q):
        m = re.search(r"\d{4}-\d{2}-\d{2}", q)
        if m:
            return pd.to_datetime(m.group(0))
        return None

    def explain_date(date):
        row = merged[merged["ds"] == date]

        if row.empty:
            return "⚠️ No data for this date."

        row = row.iloc[0]

        actual = row["y"]
        predicted = row["yhat"]
        trend = row["trend"]
        weekly = row["weekly"]
        yearly = row["yearly"]

        customers = row.get("Customers", None)
        promo = row.get("Promo", None)

        diff = actual - predicted

        header = f"""
📅 **Date Analyzed:** {date.date()}

📊 **Sales Performance Summary**
- **Actual Sales:** {actual}
- **Forecasted Sales:** {round(predicted)}
- **Difference:** {round(diff)} ({'significant drop' if diff < 0 else 'higher than expected'})
"""

        interp = []
        if diff < -0.2 * predicted:
            interp.append("1️⃣ **Sales vs Forecast:** Actual sales were far below expectations.")
        elif diff > 0.2 * predicted:
            interp.append("1️⃣ **Sales vs Forecast:** Sales exceeded forecast considerably.")
        else:
            interp.append("1️⃣ **Sales vs Forecast:** Sales stayed close to expectations.")

        causes = ["\n2️⃣ **Possible Causes:**"]
        if promo == 0:
            causes.append("- No active promotion on this date.")
        if customers is not None and customers < df["Customers"].mean() * 0.7:
            causes.append("- Customer traffic was much lower than average.")
        if weekly < 0:
            causes.append("- Historically weak weekday.")
        if yearly < 0:
            causes.append("- Seasonal downturn.")
        causes.append("- External factors like weather, events, or supply issues.")

        summary = """
**Summary:**  
The deviation suggests external or operational factors influenced sales beyond natural trend or seasonality.
"""

        return header + "\n\n" + "\n".join(interp) + "\n\n" + "\n".join(causes) + "\n\n" + summary

    if question:
        date = extract_date(question)

        if not date:
            st.error("❌ Please include a date in the format YYYY-MM-DD")
        else:
            st.markdown(explain_date(date))

else:
    st.info("⬆️ Upload a CSV to get started.")
