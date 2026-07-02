# GridCastAI: Household Electricity Load Forecasting

## Project Overview
GridCastAI is an end-to-end machine learning pipeline designed for highly accurate forecasting of household electricity demand. The project utilizes a Direct Multi-Horizon Forecasting methodology to predict electricity loads 24 hours into the future, empowering stakeholders to manage energy distribution more efficiently and reduce grid instability.

## Dataset
This project utilizes the **Individual Household Electric Power Consumption** dataset. It contains 2,075,259 measurements gathered over nearly four years, detailing active and reactive power, voltage, and various sub-metering metrics at a minute-level resolution. The primary target for forecasting in this project is `Global_active_power`.

## Objectives
1. Perform robust data cleaning and handling of missing values using time-based interpolation.
2. Construct advanced temporal, cyclical, and rolling statistical features to capture underlying electricity consumption patterns.
3. Implement a Direct Multi-Horizon Forecasting strategy, utilizing 24 distinct LightGBM regressors to predict the next 24 hours sequentially.
4. Outperform a naive 24-hour persistence baseline significantly in Mean Absolute Error (MAE).

## Project Structure
```text
GridCastAI/
├── data/
│   └── household_power_consumption.txt   # Raw dataset
├── models/                               # Serialized LightGBM model files
├── outputs/
│   ├── feature_importance.csv            # Average feature importances
│   ├── forecast_results.csv              # Next 24-hour forecast from last timestamp
│   └── metrics.csv                       # Evaluation metrics for all 24 horizons
├── main.py                               # Core executable ML pipeline
├── forecast_results.png                  # Test set actual vs predicted visualization
├── requirements.txt                      # Project dependencies
└── README.md                             # Project documentation
```

## Feature Engineering
The pipeline engineers dozens of features to optimize predictive performance:
- **Lag Features**: Lags from 1 hour to 1 week (168 hours) to capture recent history.
- **Rolling Statistics**: Mean, median, max, min, std, skewness, and kurtosis computed over varying windows (e.g., 24 hours).
- **Cyclical Features**: Sine and cosine transformations of hours and weekdays to capture the periodic nature of human activity.
- **Trend Features**: Hourly and daily differences to gauge rapid consumption shifts.

## Direct Multi-Horizon Forecasting
Unlike recursive forecasting (which uses its own predictions as inputs for future steps, causing error accumulation), this project employs **Direct Forecasting**. We train 24 completely separate LightGBM models, one specifically optimized for each hourly horizon ($h=1, 2, ..., 24$).

## LightGBM
We selected LightGBM due to its industry-leading execution speed, minimal memory footprint, and native handling of high-dimensional temporal datasets.
**Hyperparameters**:
- `n_estimators`: 500
- `learning_rate`: 0.05
- `num_leaves`: 31
- `subsample`: 0.8
- `colsample_bytree`: 0.8

## Evaluation Metrics
The primary metric used is **Mean Absolute Error (MAE)**. We compare our LightGBM predictions against a Naive Baseline (predicting that demand will simply be exactly what it was 24 hours prior).

## Results
The Direct Multi-Horizon approach drastically outperforms the naive baseline across every single horizon. The **Overall Average Model MAE** across all 24 horizons rests impressively at roughly `0.4295 kW`, reflecting baseline improvements stretching from ~21% for day-ahead predictions to ~45% for short-term predictions.

## How to Run

1. Ensure Python 3.9+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place `household_power_consumption.txt` inside the `data/` directory.
4. Execute the pipeline:
   ```bash
   python main.py
   ```
*The script will autonomously read data, train models, evaluate, and output all CSVs, the visualization, and serialized models without requiring human intervention.*

## Requirements
See `requirements.txt` for exact versions, consisting of:
- `pandas`
- `numpy`
- `scikit-learn`
- `lightgbm`
- `joblib`
- `matplotlib`

## Business Question Answers

### 1. What would you change if you had to forecast for hundreds of thousands of meters at once instead of one?
Scaling from a single meter to hundreds of thousands requires a shift from local processing to distributed MLOps:
- **Data Infrastructure**: I would replace in-memory Pandas with a distributed framework like Apache Spark (PySpark) to handle the out-of-core memory requirements. The raw data would be ingested into a scalable cloud storage solution or time-series database (e.g., Snowflake, TimescaleDB).
- **Modeling Approach**: Managing 100,000 individual "local" models is an operational bottleneck. I would pivot to a Global Forecasting Model (training a single, highly complex model like LightGBM or a Temporal Fusion Transformer on all meters simultaneously). To retain individual accuracy, I would inject static metadata features (e.g., meter_id, household_size, geography) so the model can learn cross-meter patterns while respecting individual baselines.
- **Pipeline Automation**: The pipeline would be containerized (Docker) and orchestrated using a tool like Apache Airflow to automate nightly data ingestion, feature engineering, and batch inference.

### 2. Do you think a model like this is used in practice by utilities, or would something simpler win?
In practice, the choice between a complex ML model and a simple heuristic depends entirely on the deployment environment and the ROI of accuracy:
- **Grid/Substation Level (Complex Models Win)**: For aggregated forecasting, complex models like XGBoost or LSTMs are the industry standard. At the grid level, a 1% reduction in forecasting error translates to millions of dollars saved in load balancing and peak-generation costs. The infrastructure cost of running the model is negligible compared to the financial savings.
- **Individual/Edge Level (Simpler Models Win)**: If the goal is to forecast at the individual smart meter level (especially if deployed on the 'Edge' device itself), compute, memory, and inference costs become strict bottlenecks. For millions of individual homes, a lightweight heuristic—such as a dynamic rolling average or a SARIMA model—often provides a 'good enough' forecast at a fraction of the computational cloud cost, resulting in a better overall business ROI.
