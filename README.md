Air Quality Forecast Engine
An end-to-end machine learning pipeline for training air quality models and deploying an interactive forecasting dashboard with Streamlit. This project predicts 24-hour concentrations of Ozone (O3) and Nitrogen Dioxide (NO 2) for multiple monitoring sites.

📖 Overview
This project provides a complete solution for air quality forecasting. It consists of two main components:

A training pipeline (train_and_save_models.py): This script automates the process of loading data for multiple sites, performing extensive feature engineering, training separate LightGBM models for O3 and NO2​, evaluating their performance, and saving the resulting artifacts (models, feature lists, and metrics).

An interactive web application (app.py): A Streamlit dashboard that allows users to select a monitoring site, upload unseen input data, and receive a 24-hour forecast. The forecasts are presented in interactive Plotly charts, along with detailed data tables and model performance metrics.

✨ Key Features
Multi-Site Modeling: The training pipeline is designed to build and manage models for numerous distinct monitoring sites automatically.


Advanced Feature Engineering: Implements a sophisticated feature engineering pipeline to capture temporal patterns, satellite data characteristics, wind dynamics, and pollutant interactions.

High-Performance Models: Utilizes the LightGBM library for fast, efficient, and accurate gradient boosting models.

Interactive Dashboard: A user-friendly Streamlit application for generating and visualizing forecasts.

Rich Visualizations: Employs Plotly to create interactive, pannable, and zoomable forecast charts with annotations for peak values.

Model Transparency: The dashboard displays key performance metrics (R2, RMSE, RIA) for the selected site's models, calculated on an internal validation set.

Data Export: Allows users to download the full 24-hour prediction data as a CSV file.

🛠️ Methodology
The forecasting approach is built on a robust machine learning methodology.

Feature Engineering
The feature_engineering.py script creates a rich set of features to improve model accuracy:


Time-Based Features: Cyclical features (sine/cosine) are generated for the hour of the day and month of the year to capture daily and seasonal patterns.


Satellite Data Processing: Creates flags for missing satellite data and then applies forward/backward fill to handle gaps.


Wind Features: Calculates wind speed and direction from u and v wind components.


Physics-Informed Proxies: Creates interaction terms and ratios, such as the ratio of HCHO to NO2 from satellite data, to model chemical relationships.


Time-Series Features: Generates lagged features (e.g., pollutant concentration from 24 hours prior) and rolling window statistics (mean, std dev) to capture temporal dependencies.

Modeling
Algorithm: LightGBM (LGBMRegressor), a gradient boosting framework, is used as the core learning algorithm.

Targets: Separate models are trained independently for the two target pollutants: O3 and NO2.

Data Splitting: Data for each site is split chronologically into a 75% training set and a 25% validation set. Early stopping is used during training to prevent overfitting.

Evaluation
Model performance is assessed on the held-out validation set using three metrics:

R-squared (R2): The proportion of the variance in the dependent variable that is predictable from the independent variables.

Root Mean Squared Error (RMSE): The standard deviation of the prediction errors, giving a sense of the average error magnitude in the target units (μg/m3).


Refined Index of Agreement (RIA): A robust measure of model prediction accuracy, with the function defined in feature_engineering.py and implemented in train_and_save_models.py.

📁 Project Structure
The project expects the following directory structure, which is inferred from the file paths in the scripts:

⚙️ Setup and Installation
Clone the repository:

Create and activate a virtual environment (recommended):

Install the required dependencies from requirements.txt:

🚀 Usage
The project workflow is divided into two main steps: training and forecasting.

1. Training the Models
Before running the dashboard, you must train the models by running the training script.

Prerequisite: Place your training data files (e.g., site_1_train_data.csv) into a data/ directory at the project root.

Run the training script from your terminal:

Output: The script will create a models/ directory. Inside, it will generate a subdirectory for each site containing the trained models and other necessary artifacts.

2. Running the Forecast Dashboard
Once the models are trained and saved, you can launch the interactive application.

Run the Streamlit app from your terminal:

How to use the dashboard:

Your web browser will open with the application running.

In the sidebar, use the dropdown menu to select the desired Monitoring Site.

Click the "Browse files" button to upload the corresponding unseen input data CSV for that site (e.g., site_1_unseen_input_data.csv).

The application will automatically process the data, generate the 24-hour forecast, and display the results.

📦 Dependencies
This project relies on the libraries listed in requirements.txt. The key dependencies are:

streamlit

pandas

numpy

scikit-learn

lightgbm

plotly

joblib
