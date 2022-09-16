import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import os
os.getcwd()

# data ingestion
app_df = pd.read_csv('export_files/cc_raw_data/application_record.csv')
credit_df = pd.read_csv('export_files/cc_raw_data/credit_record.csv')

####### VARIABLE CREATION #######

# TARGET: probability of default
cc_default_df = pd.DataFrame()

cc_default_df['Total'] = credit_df[['ID', 'STATUS']].groupby('ID').count()
credit_df['STATUS_DEFAULT'] = np.select([credit_df['STATUS'].isin(['2', '3', '4', '5'])], [1], default=0)
cc_default_df['Default'] = credit_df[['ID', 'STATUS_DEFAULT']].groupby('ID').sum()

cc_default_df['PROB_DEFAULT'] = cc_default_df['Default']/cc_default_df['Total']

# TARGET: binary indicator
cc_default_df['TRGT_BADRISK_IND'] = np.select([cc_default_df['PROB_DEFAULT'] > 0], [1], default=0)

# join dataset
app_df = app_df.set_index('ID')
df = app_df.join(cc_default_df[['TRGT_BADRISK_IND']], how='inner')

df['TRGT_BADRISK_IND'].value_counts()

####### DATA CLEANING #######

#== OCCUPATION TYPES ==#
conditions = [
    ((df['OCCUPATION_TYPE'].isna()==True) & (df['DAYS_EMPLOYED']==365243) & (df['NAME_INCOME_TYPE']=='Pensioner')), # create Pensioner Occupation Type for missing
    (df['OCCUPATION_TYPE'].isna()==True) # code the rest of the missing Occupation Type as unknown
]

values = [
    'Pensioner',
    'Unknown'
]

df['CODE_OCCUPATION_TYPE'] = np.select(conditions, values, default=df['OCCUPATION_TYPE'])

#== CNT DAYS EMPLOYED ==#

# convert the pensioners days employed to 0 because they don't work
df['CNT_DAYS_EMPLOYED'] = np.select([(df['DAYS_EMPLOYED']==365243) & (df['NAME_INCOME_TYPE']=='Pensioner')], [0], default=df['DAYS_EMPLOYED'])

#== BINARY CATEGORICAL ==#

# convert Y/N to 1/0
df['FLAG_OWN_REALTY_OHE'] = np.select([df['FLAG_OWN_REALTY'] == 'Y'], [1], default=0)
df['FLAG_OWN_CAR_OHE'] = np.select([df['FLAG_OWN_CAR'] == 'Y'], [1], default=0)

# conver F/M to 1/0
df['CODE_GENDER_F'] = np.select([df['CODE_GENDER'] == 'F'], [1], default=0)

#== CNT FAM MEMBERS ==#

# fix dtype
df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].astype('int')

mean_df_gender = df.groupby('CODE_GENDER_F').sum().reset_index()
mean_df_gender['TRGT_BADRISK_IND']

#== CONVERT TO POSITIVE ==#

df['CNT_DAYS_EMPLOYED'] = -1*df['CNT_DAYS_EMPLOYED']
df['DAYS_BIRTH'] = -1*df['DAYS_BIRTH']

#== DROP VARIABLES ==#

df.drop('OCCUPATION_TYPE', axis=1, inplace=True)
df.drop('FLAG_MOBIL', axis=1, inplace=True)
df.drop('DAYS_EMPLOYED', axis=1, inplace=True)
df.drop('FLAG_OWN_REALTY', axis=1, inplace=True)
df.drop('FLAG_OWN_CAR', axis=1, inplace=True)
df.drop('CODE_GENDER', axis=1, inplace=True)

####### SAVE DATA #######

import datetime as dt
today = dt.datetime.today().strftime('%Y%m%d')
df.to_csv(f"export_files/cc_clean_data_{today}.csv")