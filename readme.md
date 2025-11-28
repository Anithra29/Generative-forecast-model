📌 Project Title:

Generative Forecast Explanation System using Prophet & Explainable AI

📘 Summary

I built an intelligent forecasting system that not only predicts future sales using Facebook Prophet, but also explains why sales increased or decreased on any given date.
The system includes EDA, forecasting, a WHY chatbot, full benchmarking, and performance optimization, all deployed in an interactive Streamlit application.

🎯 Problem Statement

Most forecasting tools only show predictions, but business users (managers, analysts) need to understand:

Why did sales drop on a specific day?

What seasonal or promotional factors influenced the trend?

Can the model justify its forecast?

My goal was to build a system that answers both:
✔ What will happen
✔ Why it will happen

🔧 Key Features
1. Data Preprocessing

Automatic date parsing

Missing date detection

Outlier detection using Z-Score

Clean data preview

2. Exploratory Data Analysis

Sales trend visualization

Sales distribution

Correlation heatmap

Outlier inspection

Clear captions under each visualization

3. Forecasting with Prophet

Weekly & yearly seasonality

Configurable forecast window

Forecast plot + component plots

Easy interpretation with textual explanations

4. WHY Chatbot (Explainable AI)

A rule-based reasoning engine that explains:

Why sales were higher or lower than forecast

Weekly seasonality effect

Yearly seasonal influence

Promotion impact

Trend behavior

Produces a natural language explanation for any date.

5. Benchmarking System

Measured execution time for:

Preprocessing

EDA

Prophet model training

Forecast generation

Data merge

Chatbot reasoning

Automatically generates a Benchmark Excel Report.

6. Optimization

Improved performance across all steps:

Single-load preprocessing

Vectorized EDA operations

Prophet efficiency (disabled daily seasonality)

One-time merging

Lightweight chatbot logic

Reduced redundant operations

Result: Faster UI, quicker forecasts, instant chatbot.

🚀 Technology Stack

Python

Streamlit

Prophet

Pandas / NumPy

Plotly

Regex

Benchmarking using time module`

Excel report generation (to_excel)

📈 Impact

✔ Built a fully explainable forecasting system
✔ Improved Prophet performance significantly
✔ Enabled instant “why” answer generation
✔ Delivered an interactive, intuitive dashboard
✔ Added benchmarking + optimization for accuracy & speed

This project demonstrates my skills in:

Time series forecasting

Explainable AI

Data visualization

Streamlit development

Performance tuning

Real-world analytical problem-solving
