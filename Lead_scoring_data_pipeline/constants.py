# You can create more variables according to your project. The following are the basic variables that have been provided to you
DB_PATH = '/home/airflow/dags/Lead_scoring_data_pipeline'
DB_FILE_NAME = 'lead_scoring_data_cleaning.db'
#UNIT_TEST_DB_FILE_NAME = 
DATA_DIRECTORY = '/home/airflow/dags/Lead_scoring_data_pipeline/data/'
INTERACTION_MAPPING_INFO = '/home/airflow/dags/Lead_scoring_data_pipeline/mapping/interaction_mapping.csv'
INDEX_OF_COLUMNS_NEEDED =  ['created_date', 'city_tier', 'first_platform_c', 'first_utm_medium_c',
       'first_utm_source_c', 'total_leads_droppped', 'referred_lead',
       'app_complete_flag']
# to be set when training is used to fetch data 
#LEADSCORING_CSV_DATA_PATH = f"{DATA_DIRECTORY}leadscoring.csv"
## to be set when inference is used
LEADSCORING_CSV_DATA_PATH = f"{DATA_DIRECTORY}leadscoring_inference.csv"  

