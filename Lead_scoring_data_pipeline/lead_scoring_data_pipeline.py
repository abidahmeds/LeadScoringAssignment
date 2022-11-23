##############################################################################
# Import necessary modules
# #############################################################################


from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta
from Lead_scoring_data_pipeline.utils import *
from Lead_scoring_data_pipeline.data_validation_checks import *


###############################################################################
# Define default arguments and DAG
# ##############################################################################

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022,7,30),
    'retries' : 1, 
    'retry_delay' : timedelta(seconds=5)
}


ML_data_cleaning_dag = DAG(
                dag_id = 'Lead_Scoring_Data_Engineering_Pipeline',
                default_args = default_args,
                description = 'DAG to run data pipeline for lead scoring',
                schedule_interval = '@daily',
                catchup = False
)
# tasks needed to be performed -  build_dbs -> raw_data_schema_check -> load_data_into_db -> map_city_tier -> map_categorical_vars -> interactions_mapping -> model_input_schema_check


###############################################################################
# Create a task for build_dbs() function with task_id 'building_db'
# ##############################################################################
building_db_task = PythonOperator(
                        task_id = 'building_db',
                        python_callable = build_dbs,
                        dag = ML_data_cleaning_dag)
###############################################################################
# Create a task for raw_data_schema_check() function with task_id 'checking_raw_data_schema'
# ##############################################################################
check_raw_data_schema_task = PythonOperator(
                                task_id = 'check_raw_data_schema',
                                python_callable = raw_data_schema_check,
                                dag = ML_data_cleaning_dag)

###############################################################################
# Create a task for load_data_into_db() function with task_id 'loading_data'
# #############################################################################
loading_data_task = PythonOperator(
                        task_id = 'loading_data',
                        python_callable = load_data_into_db,
                        dag = ML_data_cleaning_dag)

###############################################################################
# Create a task for map_city_tier() function with task_id 'mapping_city_tier'
# ##############################################################################
mapping_city_tier_dask =  PythonOperator(
                               task_id = 'mapping_city_tier',
                               python_callable = map_city_tier,
                               dag = ML_data_cleaning_dag)

###############################################################################
# Create a task for map_categorical_vars() function with task_id 'mapping_categorical_vars'
# ##############################################################################
mapping_categorical_vars_task = PythonOperator(
                                      task_id = 'mapping_categorical_vars',
                                      python_callable = map_categorical_vars,
                                      dag = ML_data_cleaning_dag)

###############################################################################
# Create a task for interactions_mapping() function with task_id 'mapping_interactions'
# ##############################################################################
mapping_interactions_task = PythonOperator(
                                  task_id = 'mapping_interactions',
                                  python_callable = interactions_mapping,
                                  dag = ML_data_cleaning_dag)

###############################################################################
# Create a task for model_input_schema_check() function with task_id 'checking_model_inputs_schema'
# ##############################################################################
check_model_inputs_schema_task = PythonOperator(
                                task_id = 'checking_model_inputs_schema',
                                python_callable = model_input_schema_check,
                                dag = ML_data_cleaning_dag)

###############################################################################
# Define the relation between the tasks
# ##############################################################################

building_db_task >> check_raw_data_schema_task >> loading_data_task >> mapping_city_tier_dask >> mapping_categorical_vars_task >> mapping_interactions_task >> check_model_inputs_schema_task

