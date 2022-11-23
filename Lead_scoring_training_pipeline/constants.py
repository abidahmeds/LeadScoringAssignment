PROD_DB_PATH ='/home/airflow/dags/Lead_scoring_data_pipeline/'
PROD_DB_FILE_NAME = 'lead_scoring_data_cleaning.db'
MLFLOW_PATH = '/home/Assignment/mlruns/'
DB_FOR_MLFLOW_FILES = 'Lead_scoring_mlflow_production.db'

ML_TRACKING_URI = 'http://0.0.0.0:6007'
EXPERIMENT = 'Lead_scoring_mlflow_production'


# model config imported from pycaret experimentation
model_config ={'boosting_type' :'gbdt', 'class_weight' : None, 'colsample_bytree':1.0,
               'device':'gpu', 'importance_type':'split', 'learning_rate' : 0.1,
               'max_depth' :-1, 'min_child_samples': 20, 'min_child_weight': 0.001,
               'min_split_gain': 0.0, 'n_estimators': 100, 'n_jobs':-1, 'num_leaves' : 31,
               'objective': None, 'random_state' : 42, 'reg_alpha' : 0.0, 'reg_lambda' : 0.0,
               'silent':'warn', 'subsample':1.0, 'subsample_for_bin' : 200000,
               'subsample_freq' : 0} 

# list of the features that needs to be there in the final encoded dataframe
#ONE_HOT_ENCODED_FEATURES =
ONE_HOT_ENCODED_FEATURES = ['total_leads_droppped', 'city_tier', 'referred_lead', 'first_platform_c_Level0', 'first_platform_c_Level1',
                           'first_platform_c_Level2', 'first_platform_c_Level3',
                           'first_platform_c_Level7', 'first_platform_c_Level8',
                           'first_platform_c_others', 'first_utm_medium_c_Level0',
                           'first_utm_medium_c_Level10', 'first_utm_medium_c_Level11',
                           'first_utm_medium_c_Level13', 'first_utm_medium_c_Level15',
                           'first_utm_medium_c_Level16', 'first_utm_medium_c_Level2',
                           'first_utm_medium_c_Level20', 'first_utm_medium_c_Level26',
                           'first_utm_medium_c_Level3', 'first_utm_medium_c_Level30',
                           'first_utm_medium_c_Level33', 'first_utm_medium_c_Level4',
                           'first_utm_medium_c_Level43', 'first_utm_medium_c_Level5',
                           'first_utm_medium_c_Level6', 'first_utm_medium_c_Level8',
                           'first_utm_medium_c_Level9', 'first_utm_medium_c_others',
                           'first_utm_source_c_Level0', 'first_utm_source_c_Level14',
                           'first_utm_source_c_Level16', 'first_utm_source_c_Level2',
                           'first_utm_source_c_Level4', 'first_utm_source_c_Level5',
                           'first_utm_source_c_Level6', 'first_utm_source_c_Level7',
                   'first_utm_source_c_others']
FEATURES_TO_ENCODE = ['first_platform_c','first_utm_medium_c','first_utm_source_c']
# list of features that need to be one-hot encoded
