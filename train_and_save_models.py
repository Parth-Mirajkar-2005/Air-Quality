import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
# --- NEW IMPORTS ---
from sklearn.metrics import mean_squared_error, r2_score
from feature_engineering import run_feature_engineering, ria # Import the pipeline and ria metric

# --- Configuration ---
NUM_SITES = 7

def train_lgbm(X_tr, y_tr, X_v, y_v, objective='regression', alpha=0.5):
    """A helper function to train a LightGBM model with early stopping."""
    params = {'objective': objective, 'metric': 'rmse', 'n_estimators': 2000, 'n_jobs': -1, 'verbose': -1}
    if objective == 'quantile':
        params['alpha'] = alpha
        params['n_estimators'] = 1000
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return model

# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================
print("--- Starting Multi-Site Model Training and Export Process ---")

for i in range(1, NUM_SITES + 1):
    site_id = i
    print(f"\n{'='*20} Processing Site {site_id} {'='*20}")

    # 1. LOAD DATA
    train_file = f"data/site_{site_id}_train_data.csv"
    unseen_file = f"data/site_{site_id}_unseen_input_data.csv"
    print(f"Step 1/6: Loading data...")
    try:
        df_train = pd.read_csv(train_file)
        df_unseen = pd.read_csv(unseen_file)
    except FileNotFoundError:
        print(f"Warning: Data for Site {site_id} not found. Skipping.")
        continue

    # 2. PROCESS DATA
    print("Step 2/6: Applying feature engineering...")
    df_train['source'] = 'train'
    df_unseen['source'] = 'unseen'
    df_combined = pd.concat([df_train, df_unseen], ignore_index=True)
    df_processed = run_feature_engineering(df_combined, is_train=True)
    
    train_fe = df_processed[df_processed['source'] == 'train'].drop(columns='source')

    # 3. DEFINE FEATURES AND TARGETS
    exclude_cols = ['datetime', 'year', 'month', 'day', 'hour', 'O3_target', 'NO2_target']
    feature_cols = [c for c in train_fe.columns if c not in exclude_cols and train_fe[c].dtype != 'object']
    X = train_fe[feature_cols]
    y_o3 = train_fe['O3_target']
    y_no2 = train_fe['NO2_target']

    valid_rows_mask = y_o3.notna() & y_no2.notna()
    X = X[valid_rows_mask].reset_index(drop=True)
    y_o3 = y_o3[valid_rows_mask].reset_index(drop=True)
    y_no2 = y_no2[valid_rows_mask].reset_index(drop=True)
    
    split_point = int(0.75 * len(X))
    X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
    y_o3_train, y_o3_val = y_o3.iloc[:split_point], y_o3.iloc[split_point:]
    y_no2_train, y_no2_val = y_no2.iloc[:split_point], y_no2.iloc[split_point:]
    
    # 4. TRAIN MODELS
    print(f"Step 3/6: Training models for Site {site_id}...")
    model_o3 = train_lgbm(X_train, y_o3_train, X_val, y_o3_val)
    model_no2 = train_lgbm(X_train, y_no2_train, X_val, y_no2_val)
    # NOTE: Quantile models are not needed for the 3-column output, but we can keep them for other uses.
    
    # --- NEW: STEP 5 - CALCULATE VALIDATION METRICS ---
    print(f"Step 4/6: Calculating performance metrics on validation set...")
    pred_o3_val = model_o3.predict(X_val)
    pred_no2_val = model_no2.predict(X_val)
    
    metrics = {
        "O3": {
            "RMSE": np.sqrt(mean_squared_error(y_o3_val, pred_o3_val)),
            "R2": r2_score(y_o3_val, pred_o3_val),
            "RIA": ria(y_o3_val, pred_o3_val)
        },
        "NO2": {
            "RMSE": np.sqrt(mean_squared_error(y_no2_val, pred_no2_val)),
            "R2": r2_score(y_no2_val, pred_no2_val),
            "RIA": ria(y_no2_val, pred_no2_val)
        }
    }
    print("Metrics calculated successfully.")
    
    # 6. SAVE ARTIFACTS
    output_dir = f'models/site_{site_id}'
    print(f"Step 5/6: Saving model artifacts to '{output_dir}/'...")
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(model_o3, f'{output_dir}/model_o3.pkl')
    joblib.dump(model_no2, f'{output_dir}/model_no2.pkl')
    joblib.dump(feature_cols, f'{output_dir}/feature_cols.pkl')
    
    # --- NEW: Save the metrics dictionary ---
    joblib.dump(metrics, f'{output_dir}/validation_metrics.pkl')
    print(f"Step 6/6: --- Site {site_id} processing complete! ---")

print("\n--- All sites processed. Multi-site training complete. ---")