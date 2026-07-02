from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'gridcast_forecast_pipeline',
    default_args=default_args,
    description='Daily ML Pipeline for Electricity Load Forecasting',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    # 1. Load Data
    load_data = BashOperator(
        task_id='load_data',
        bash_command='echo "Loading data from database/lake..."'
    )

    # 2. Clean Data
    clean_data = BashOperator(
        task_id='clean_data',
        bash_command='echo "Cleaning and interpolating missing values..."'
    )

    # 3. Feature Engineering
    feature_engineering = BashOperator(
        task_id='feature_engineering',
        bash_command='echo "Generating temporal and rolling features..."'
    )

    # 4. Train Model & Log to MLflow
    train_model = BashOperator(
        task_id='train_model',
        # In a real environment, this might call python src/train.py or trigger a Databricks job
        bash_command='python /app/main.py' 
    )

    # 5. Model Evaluation (Embedded in training here, but logically separate)
    evaluate_model = BashOperator(
        task_id='evaluate_model',
        bash_command='echo "Evaluating model against naive baseline..."'
    )

    # 6. Register Model
    register_model = BashOperator(
        task_id='register_model',
        bash_command='echo "Model automatically registered in MLflow registry..."'
    )

    # 7. Generate Forecast
    generate_forecast = BashOperator(
        task_id='generate_forecast',
        bash_command='echo "Generating forecast for the next 24 hours..."'
    )

    # 8. Save Outputs
    save_outputs = BashOperator(
        task_id='save_outputs',
        bash_command='echo "Saving outputs to Data Lake..."'
    )

    # Define DAG Flow
    load_data >> clean_data >> feature_engineering >> train_model >> evaluate_model >> register_model >> generate_forecast >> save_outputs
