# 🌍 Air Quality Forecast Engine

> An end-to-end machine learning pipeline for training site-specific models and deploying an interactive dashboard to forecast $O_3$ and $NO_2$ concentrations.

-----

## 📋 Table of Contents

A Table of Contents (TOC) is a clickable menu that helps you navigate the document. It provides a quick overview of the project's structure and allows you to jump directly to any section of interest.

  * [📸 Live Demo](#-live-demo)
  * [✨ Key Features](#-key-features)
  * [🛠️ Technical Deep Dive](#️-technical-deep-dive)
  * [🚀 Getting Started](#-getting-started)
  * [▶️ Usage](#️-usage)
  * [📁 Project Structure](#-project-structure)
  * [📦 Technology Stack](#-technology-stack)

## 📸 Live Demo

Below is a placeholder for a GIF showcasing the interactive dashboard. A live demo highlights the fluid user experience, from uploading data to visualizing the 24-hour forecast with interactive Plotly charts.

Link<>

## ✨ Key Features

  * **🛰️ Multi-Site Modeling**: The training pipeline automatically processes data and trains dedicated models for multiple distinct monitoring sites.
  * **🧠 Advanced Feature Engineering**: Implements a sophisticated pipeline that generates over 50 features, capturing temporal, meteorological, and chemical interaction patterns to boost model accuracy.
  * **⚡ High-Performance Models**: Utilizes LightGBM, a high-performance gradient boosting framework, for fast training and accurate predictions.
  * **📊 Interactive Dashboard**: A polished and user-friendly Streamlit application for generating, visualizing, and exploring forecasts.
  * **📈 Rich Visualizations**: Employs Plotly to create fully interactive charts with zooming, panning, and annotations, allowing for in-depth exploration of forecast data.
  * **⭐ transparency**: The dashboard displays key performance metrics ($R^2$, RMSE, RIA) for each model, calculated on a validation set during training, ensuring transparency about model reliability.

## 🛠️ Technical Deep Dive

This section details the architectural choices and methodology.

**1. Feature Engineering Rationale**

The core of this project's accuracy lies in its feature engineering pipeline. We don't just feed raw data to the model; we enrich it.

  * **Cyclical Time Features**: Time is cyclical (e.g., hours of the day, months of the year). Representing `hour` as a number from 0-23 is problematic because 23 and 0 are far apart numerically but close in time. By converting them into `sin` and `cos` components, we represent time on a circle, which helps the model understand these patterns.
  * **Satellite Data Handling**: Satellite readings can have gaps due to cloud cover. The pipeline creates a specific `_missing` flag for each satellite column before filling the NaNs. This tells the model not just the value, but also whether that value was original or imputed, which can be a powerful signal.
  * **Physics-Informed Proxies**: We create features that represent known atmospheric relationships, such as the ratio of HCHO to $NO_2$ from satellite data (`HCHO_NO2_sat_ratio`). These "proxy" features guide the model with domain-specific knowledge.
  * **Time-Series Features**: Lags and rolling window statistics (mean, std) are created for key variables. This gives the model a memory of recent trends, which is crucial for time-series forecasting.

**2. Modeling and Evaluation Strategy**

  * **Why LightGBM?**: LightGBM (`lgb.LGBMRegressor`) was chosen for its exceptional performance, speed, and lower memory usage compared to other gradient boosting methods. It's well-suited for the tabular data format of this project.
  * **Preventing Overfitting**: We use a chronological 75/25 split for training and validation. Crucially, the model uses **early stopping** (`lgb.early_stopping`). This monitors the model's performance on the validation set and stops training if it doesn't improve for 50 consecutive rounds, preventing it from memorizing the training data.
  * **Evaluation Metrics**:
      * **$R^2$**: Tells us the percentage of variance in the real-world data that our model can explain.
      * **RMSE**: Gives us the typical error magnitude in the original units ($\mu g/m^3$), making it highly interpretable.
      * **RIA (Refined Index of Agreement)**: A custom metric used to provide a more robust measure of model prediction accuracy.


## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

  * Python 3.9 or higher
  * `pip` and `venv` for package management

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://your-repository-url/air-quality-forecast.git
    cd air-quality-forecast
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies from `requirements.txt`:**

    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

The project has two main workflows: training the models and running the forecast app.

### 1\. 🚂 Train the Models

  * **Prerequisite**: Place your `site_{id}_train_data.csv` files into a `data/` directory at the project root.
  * **Run the training script:**
    ```bash
    python train_and_save_models.py
    ```
    This will generate all necessary model artifacts inside the `models/` directory.

### 2\. 💡 Launch the Forecast Dashboard

  * **Prerequisite**: Models must be trained and available in the `models/` directory.
  * **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    Navigate to the local URL provided in your terminal to use the application.

## 📁 Project Structure

The project expects the following directory layout:

```
air-quality-forecast/
│
├── data/
│   ├── site_1_train_data.csv
│   └── site_1_unseen_input_data.csv
│
├── models/
│   └── site_1/
│       ├── model_o3.pkl
│       ├── model_no2.pkl
│       ├── feature_cols.pkl
│       └── validation_metrics.pkl
│
├── app.py
├── train_and_save_models.py
├── feature_engineering.py
└── requirements.txt
```

## 📦 Technology Stack

This project is built with the following core libraries:

  * **`Streamlit`**: For building the interactive web application.
  * **`Pandas` & `NumPy`**: For data manipulation and numerical operations.
  * **`Scikit-learn`**: For data processing and evaluation metrics.
  * **`LightGBM`**: For the core gradient boosting models.
  * **`Plotly`**: For creating rich, interactive data visualizations.
  * **`Joblib`**: For serializing and deserializing Python objects (models, etc.).