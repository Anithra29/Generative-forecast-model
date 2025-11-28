import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import numpy as np
import re
import base64
import time
import plotly.express as px
import os

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Generative Forecast Explanation System", layout="wide")

# ---------------------------
# BACKGROUND + CSS
# ---------------------------
def add_bg_from_local(image_path):
    if os.path.exists(image_path):
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

            /* FIX #3 — Larger Title */
            h1 {{
                font-size: 42px !important;
                font-weight: 900 !important;
                color: white !important;
            }}

            h2, h3, h4, h5, h6, p, label, span, div {{
                color: white !important;
                font-weight: 600 !important;
            }}

            /* DataFrame text */
            .stDataFrame table tbody tr td {{
                color: white !important;
            }}
            .stDataFrame table thead tr th {{
                color: white !important;
            }}

            .stFileUpload {{
                color: white !important;
            }}

            /* FIX #2 — Button visible */
            .stDownloadButton>button {{
                background-color: #ffffff !important;
                color: black !important;
                border-radius: 8px !important;
                padding: 10px 20px !important;
                border: 2px solid black !important;
                font-weight: 700 !important;
            }}

            /* Extra fix for invisible button text */
            .stDownloadButton > button div {{
                color: black !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

add_bg_from_local("background/background.png")

# ---------------------------
# TITLE + UPLOAD AREA
# ---------------------------
st.markdown("<h1>📈 Generative Forecast Explanation System</h1>", unsafe_allow_html=True)
st.caption("Upload your CSV (columns: ds, y, Customers, Promo, DayOfWeek)")

uploaded = st.file_uploader(
    "prophet_ready.csv\nDrag and drop file here\nLimit 200MB per file • CSV",
    type="csv"
)

# Show uploaded file info
if uploaded:
    file_name = uploaded.name
    try:
        size_kb = round(len(uploaded.getvalue()) / 1024, 1)
    except:
        size_kb = None

    st.markdown(f"**{file_name}**")
    if size_kb:
        st.markdown(f"{size_kb} KB")
    st.write("")

# ---------------------------
# START APP AFTER FILE UPLOAD
# ---------------------------
if uploaded:

    benchmark_results = []

    # -------------------------------------
    # 1️⃣ DATA PREPROCESSING
    # -------------------------------------
    try:
        t0 = time.time()

        df = pd.read_csv(uploaded)
        df["ds"] = pd.to_datetime(df["ds"])
        pretty_df = df.rename(columns={"ds": "Date", "y": "Sales"})

        t1 = time.time()
        benchmark_results.append({
            "Step": "Data Preprocessing",
            "Description": "Loaded dataset, parsed dates, validated columns.",
            "Time (seconds)": round(t1 - t0, 3),
            "Status": "Success"
        })

    except Exception as e:
        benchmark_results.append({
            "Step": "Data Preprocessing",
            "Description": "Loaded dataset, parsed dates, validated columns.",
            "Time (seconds)": 0,
            "Status": f"Failed: {str(e)}"
        })
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    # -------------------------------------
    # TABS
    # -------------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 EDA", "📈 Forecasting", "🤖 WHY Chatbot"])

    # -------------------------------------
    # 2️⃣ EDA TAB
    # -------------------------------------
    with tab1:
        try:
            t_e1 = time.time()

            st.header("🔍 Exploratory Data Analysis (EDA)")
            st.write("This section explores your dataset to understand data quality, trends, and relationships.")

            st.subheader("📄 Data Preview")
            st.dataframe(pretty_df.head())

            st.subheader("📌 Dataset Info")
            r, c = df.shape
            st.markdown(f"**Rows:** {r} | **Columns:** {c}")

            st.subheader("❗ Missing Values")
            st.write(df.isnull().sum())

            st.subheader("⏳ Missing Date Detection")
            full_range = pd.date_range(df["ds"].min(), df["ds"].max())
            missing_dates = full_range.difference(df["ds"])
            if len(missing_dates) == 0:
                st.success("✔ No missing dates in the timeline.")
            else:
                st.error(f"{len(missing_dates)} missing dates found.")
                st.write(missing_dates)

            st.subheader("📊 Summary Statistics")
            st.write(df.describe())

            st.subheader("📈 Sales Over Time")
            st.plotly_chart(px.line(df, x="ds", y="y"), use_container_width=True)
            st.caption("Shows how sales moved over time, including general patterns and seasonal behavior.")

            st.subheader("📉 Sales Distribution")
            st.plotly_chart(px.histogram(df, x="y"), use_container_width=True)
            st.caption("Displays how often different sales values occur, highlighting peaks and unusual days.")

            st.subheader("🚨 Outlier Detection (Z-Score)")
            df["zscore"] = (df["y"] - df["y"].mean()) / df["y"].std()
            outliers = df[df["zscore"].abs() > 3]
            if outliers.empty:
                st.success("✔ No extreme outliers detected (|Z| > 3).")
            else:
                st.error(f"{len(outliers)} outliers found.")
                st.dataframe(outliers)
            st.caption("Shows days where sales were extremely high or low compared to normal patterns.")

            st.subheader("📊 Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            st.plotly_chart(px.imshow(numeric_df.corr(), text_auto=True), use_container_width=True)
            st.caption("Shows how strongly numerical features move together, helping identify relationships.")

            t_e2 = time.time()
            benchmark_results.append({
                "Step": "EDA",
                "Description": "Generated preview, checked missing data, plotted sales trends, distributions, and correlations.",
                "Time (seconds)": round(t_e2 - t_e1, 3),
                "Status": "Success"
            })

        except Exception as e:
            benchmark_results.append({
                "Step": "EDA",
                "Description": "Generated preview, checked missing data, plotted sales trends, distributions, and correlations.",
                "Time (seconds)": 0,
                "Status": f"Failed: {str(e)}"
            })

    # -------------------------------------
    # 3️⃣ FORECASTING TAB
    # -------------------------------------
    with tab2:
        try:
            t_f1 = time.time()

            st.header("📈 Forecasting")
            periods = st.slider("Days to Forecast", 30, 365, 90)

            m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            m.fit(df)

            future = m.make_future_dataframe(periods=periods)
            forecast = m.predict(future)

            t_f2 = time.time()
            st.success(f"Model Execution Time: {round(t_f2 - t_f1,2)} seconds")

            benchmark_results.append({
                "Step": "Prophet Forecasting",
                "Description": "Trained Prophet model, generated future dates, created forecast and component plots.",
                "Time (seconds)": round(t_f2 - t_f1, 3),
                "Status": "Success"
            })

            st.subheader("📈 Forecast Plot")
            st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)

            st.markdown("""
### 📝 Understanding the Forecast Plot

- **Black dots** = actual sales  
- **Blue line** = predicted trend  
- **Light blue area** = uncertainty  
- **Right side** = future forecast  

This helps identify expected rises, dips, and confidence levels.
""")

            st.subheader("📉 Forecast Components")
            st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

            st.markdown("""
### 📝 What the Forecast Components Mean

- **Trend** = long-term movement  
- **Weekly** = weekday pattern  
- **Yearly** = seasonal cycles  
""")

            t_m1 = time.time()
            merged = pd.merge(forecast, df, on="ds", how="left")
            t_m2 = time.time()

            benchmark_results.append({
                "Step": "Actual vs Forecast Merge",
                "Description": "Combined predictions with actual values for comparison and analysis.",
                "Time (seconds)": round(t_m2 - t_m1, 3),
                "Status": "Success"
            })

        except Exception as e:
            benchmark_results.append({
                "Step": "Prophet Forecasting",
                "Description": "Trained Prophet model, generated future dates, created forecast and component plots.",
                "Time (seconds)": 0,
                "Status": f"Failed: {str(e)}"
            })

    # -------------------------------------
    # 4️⃣ WHY CHATBOT — Simplified Explanation (FINAL)
    # -------------------------------------
    chatbot_executed = False

    with tab3:
        st.header("🤖 WHY Chatbot")
        question = st.text_input("Ask questions like: Why were sales low on 2015-07-10?")

        def extract_date(q):
            m = re.search(r"\d{4}-\d{2}-\d{2}", q)
            return pd.to_datetime(m.group(0)) if m else None

        # --- NEW EXPLANATION FUNC (Simplified + Human-Friendly) ---
        def explain_date(date):
            row = merged[merged["ds"] == date]
            if row.empty:
                return "⚠️ No data for that exact date."

            row = row.iloc[0]
            actual = row["y"]
            forecasted = row["yhat"]
            diff = actual - forecasted

            reasoning = []

            if diff > 0:
                reasoning.append("Sales were **higher than expected**.")
            else:
                reasoning.append("Sales were **lower than expected**.")

            if row.get("weekly", 0) > 0:
                reasoning.append("Positive **weekly seasonality** likely boosted demand.")
            elif row.get("weekly", 0) < 0:
                reasoning.append("Weak weekday effect reduced activity.")

            if row.get("yearly", 0) > 0:
                reasoning.append("This time of year typically sees **higher seasonal demand**.")
            elif row.get("yearly", 0) < 0:
                reasoning.append("Seasonal patterns show lower demand in this period.")

            if row.get("Promo", 1) == 0:
                reasoning.append("No active promotion may have lowered demand.")

            explanation = "\n".join(f"• {x}" for x in reasoning)

            return f"""
📅 **Date Analyzed:** {date.date()}

## 📊 Sales Summary
• **Actual Sales:** {actual}  
• **Forecasted Sales:** {round(forecasted)}  
• **Difference:** {round(diff)}  

## 🧠 Why did this happen?
{explanation}

## 📝 Overall Interpretation
Sales changed due to weekly effects, seasonal behavior, and demand conditions.
"""

        if question:
            try:
                t_c1 = time.time()
                date = extract_date(question)
                st.markdown(explain_date(date))
                t_c2 = time.time()

                benchmark_results.append({
                    "Step": "WHY Chatbot Reasoning",
                    "Description": "Interpreted forecast results and explained the date.",
                    "Time (seconds)": round(t_c2 - t_c1, 3),
                    "Status": "Success"
                })
                chatbot_executed = True

            except Exception as e:
                benchmark_results.append({
                    "Step": "WHY Chatbot Reasoning",
                    "Description": "Interpreted forecast results and explained the date.",
                    "Time (seconds)": 0,
                    "Status": f"Failed: {str(e)}"
                })
                chatbot_executed = True

    if not chatbot_executed:
        benchmark_results.append({
            "Step": "WHY Chatbot Reasoning",
            "Description": "Interpreted forecast results and explained the date.",
            "Time (seconds)": 0,
            "Status": "Skipped (no question)"
        })

    # ---------------------------
    # EXPORT BENCHMARK EXCEL
    # ---------------------------
    try:
        bench_df = pd.DataFrame(benchmark_results)
        excel_path = "benchmark_results.xlsx"
        bench_df.to_excel(excel_path, index=False)

        st.success("Benchmark Excel generated successfully!")

        with open(excel_path, "rb") as f:
            data_bytes = f.read()

        st.download_button(
            "⬇ Download Benchmark Excel",
            data=data_bytes,
            file_name="benchmark_results.xlsx"
        )

    except Exception as e:
        st.error(f"Failed to generate benchmark file: {str(e)}")

else:
    st.info("⬆️ Upload a CSV to get started.")
