import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#################################################### DEFINE SETUP VARIABLES ###############################################################
split = True
outliers_rmv = 'ZS'
year = 2017
out_file = 'EnergyClean/Energy_%s_%s_%i.csv' % (str(split), outliers_rmv, year)
#################################################### LOAD DATA FROM EXCEL FILES ###########################################################
if year == 2017:
    df1 = pd.read_excel('EnergySignature/EnSig_2017_Jan_Apr.xls', sheet_name='Sheet0', header=[0, 1], index_col=None)
    df2 = pd.read_excel('EnergySignature/EnSig_2017_May_Sep.xls', sheet_name='Sheet0', header=[0, 1], index_col=None)
    df3 = pd.read_excel('EnergySignature/EnSig_2017_Sep_Dec.xls', sheet_name='Sheet0', header=[0, 1], index_col=None)
    if split == True:
        df1_split = pd.read_excel('EnergySignature/En_HVAC_Light_2017_Jan_Apr.xls', sheet_name='Sheet0', header=[0, 1],
                                  index_col=0)
        df2_split = pd.read_excel('EnergySignature/En_HVAC_Light_2017_May_Aug.xls', sheet_name='Sheet0', header=[0, 1],
                                  index_col=0)
        df3_split = pd.read_excel('EnergySignature/En_HVAC_Light_2017_Sep_Dec.xls', sheet_name='Sheet0', header=[0, 1],
                                  index_col=0) 
elif year == 2018:
    df1 = pd.read_excel('EnergySignature/EnSig_2018_Genn_Apr.xls', sheet_name='Sheet0', header=[0, 1], index_col=None)
    df2 = pd.read_excel('EnergySignature/EnSig_2018_Magg_Ago.xls', sheet_name='Sheet0', header=[0, 1], index_col=None)
    df3 = pd.read_excel('EnergySignature/EnSig_2018_Sett_Dic.xls', sheet_name='Sheet0', header=[0, 1], index_col=None)
    if split == True:
        df1_split = pd.read_excel('EnergySignature/En_HVAC_Light_2018_Jan_Apr.xls', sheet_name='Sheet0', header=[0, 1],
                                  index_col=0)
        df2_split = pd.read_excel('EnergySignature/En_HVAC_Light_2018_May_Aug.xls', sheet_name='Sheet0', header=[0, 1],
                                  index_col=0)
        df3_split = pd.read_excel('EnergySignature/En_HVAC_Light_2018_Sep_Dec.xls', sheet_name='Sheet0', header=[0, 1],
                                  index_col=0)
else:
    sys.exit("year value not recognised, choose 2017 or 2018")

################################################ PREPARE DATAFRAME ########################################################################
# Concatenate the data
df = pd.concat([df1, df2, df3])

if split:
    df_split = pd.concat([df1_split, df2_split, df3_split])

# Define column names as join of both levels of column names
df.columns = ['_'.join(col).strip() for col in df.columns.values]
if split:
    df_split.columns = ['_'.join(col).strip() for col in df_split.columns.values]

# Rename column names to be easier to read (df)
cols = df.columns
for col in cols:
    if 'Irradiance' in col:
        col2 = col.split('_')[-1]
        df.columns = ['Irradiance_' + col2 if (x == col) else x for x in df.columns]
    elif 'Energia Totale' in col:
        col2 = col.split('_')[-1]
        df.columns = ['ConsEnergy_' + col2 if (x == col) else x for x in df.columns]
    elif 'Esterna' in col:
        col2 = col.split('_')[-1]
        df.columns = ['ExternalTemp_' + col2 if (x == col) else x for x in df.columns]
    else:
        col2 = col.split('_')[-1]
        df.columns = ['InternalTemp_' + col2 if (x == col) else x for x in df.columns]
# Rename column names to be easier to read (df_split)
if split:
    cols = df_split.columns
    for col in cols:
        if 'HVAC' in col:
            col2 = col.split('_')[-1]
            df_split.columns = ['HVAC_' + col2 if (x == col) else x for x in df_split.columns]
        elif 'Illuminazione' in col:
            col2 = col.split('_')[-1]
            df_split.columns = ['Lightning_' + col2 if (x == col) else x for x in df_split.columns]
#%%
################################# REPLACE THE SECONDS WITH 00 #############################################################################    
# Create array with date columns
date_columns = ['Irradiance_Date',
                'ExternalTemp_Date',
                'InternalTemp_Date',
                'ConsEnergy_Date']
# Iterate through date_columns array to replace the seconds with 00 (1th dataframe)
for column in date_columns:
    df[column] = df[column].map(lambda t: t.replace(second=0))
if split:
    df_split.index = df_split.index.map(lambda x: x.replace(second=0))
################################## CLEAN DATA BASED ON THE QUALITY ########################################################################
# Create array with quality columns
df_quality_columns = ['Irradiance_Quality',
                      'ExternalTemp_Quality',
                      'InternalTemp_Quality',
                      'ConsEnergy_Quality',
                      ]
if split:
    df1_quality_columns = ['HVAC_Quality',
                           'Lightning_Quality']
# Iterate through the quality_columns array to filter QUALITY == REAL
for column in df_quality_columns:
    df = df[df.loc[:, column] == 'Real']

if split:
    for column in df1_quality_columns:
        df_split = df_split[df_split.loc[:, column] == 'Real']

df = df[['Irradiance_Date', 'Irradiance_Value', 'ConsEnergy_Date', 'ConsEnergy_Value', 'ExternalTemp_Date',
         'ExternalTemp_Value', 'InternalTemp_Date', 'InternalTemp_Value']]

# Remove columns we do not need - Quality and Status were just for cleaning the data
if split:
    df_split = df_split[['HVAC_Value', 'Lightning_Value']]

#################### SPLITTING THE DATA IN DIFFERENT DATAFRAMES AND RENAMING DATE COLUMNS #################################################
# Irradiance
df_irrad = df[['Irradiance_Date','Irradiance_Value']]
df_irrad.columns = ['Date', 'Irradiance_Value']

# Consumption
df_cons = df[['ConsEnergy_Date', 'ConsEnergy_Value']]
df_cons.columns = ['Date', 'ConsEnergy_Value']
# External Temp
df_extTemp = df[['ExternalTemp_Date', 'ExternalTemp_Value']]
df_extTemp.columns = ['Date', 'ExternalTemp_Value']
# Internal Temp
df_intTemp = df[['InternalTemp_Date', 'InternalTemp_Value']]
df_intTemp.columns = ['Date', 'InternalTemp_Value']

if split:
    # Change from index back to column before merging
    df_split['Date'] = df_split.index
# Join - this way we do not deal with inconsistency of datetime by rows (some measurements were shifted)
df_merge = pd.merge(df_irrad, df_cons, how='inner', on=['Date'])
df_merge = pd.merge(df_merge, df_extTemp, how='inner', on=['Date'])
df_merge = pd.merge(df_merge, df_intTemp, how='inner', on=['Date'])
if split:
    df_split['Date'] = df_split.index
    df_merge = pd.merge(df_merge, df_split, how='inner', on=['Date'])


######################################### DEFINE OUTLIERS REMOVAL METHODS #################################################################
# REMOVE OUTLIERS USING Z-SCORE
def zs_clean(df):
    columns_to_clean = ['ConsEnergy_Value', 'ExternalTemp_Value', 'InternalTemp_Value']
    df = df[(np.abs(stats.zscore(df[columns_to_clean])) < 2).all(axis=1)]
    return df


# REMOVE OUTLIERS USING IQR
def iqr_clean(df):
    columns_to_clean = ['ConsEnergy_Value', 'ExternalTemp_Value', 'InternalTemp_Value']
    # First quartile
    Q1 = df.quantile(0.25)
    # Third quartile
    Q3 = df.quantile(0.75)
    # Calculate columns IQR values and print the table
    IQR = Q3 - Q1
    print('''
          
    --------------------------------
                  IQR              
    --------------------------------
          ''')
    print(IQR)
    # Calculate lower and upper bounds
    lower_bound = (Q1 - 1.5 * IQR)
    upper_bound = (Q3 + 1.5 * IQR)
    # Actual dataframe modify
    df = df[~((df[columns_to_clean] < lower_bound) |
              (df[columns_to_clean] > upper_bound)
              ).any(axis=1)]
    return df


###################################### ACTUAL OUTLIERS REMOVAL BASED ON SETUP VARIABLE ####################################################
if outliers_rmv == 'ZS':
    df_clean = zs_clean(df_merge)
elif outliers_rmv == 'IQR':
    df_clean = iqr_clean(df_merge)
elif outliers_rmv == 'NO':
    df_clean = df_merge  # No outliers removal is done
else:
    sys.exit("Outliers_rmv value not recognised")

######################### REMOVING NEGATIVE VALUES FOR ENERGY #############################################################################
df_clean = df_clean[df_clean.loc[:, 'ConsEnergy_Value'] >= 0]
if split:
    df_clean = df_clean[df_clean.loc[:, 'HVAC_Value'] >= 0]
    df_clean = df_clean[df_clean.loc[:, 'Lightning_Value'] >= 0]
################################ ADD FEATURES #############################################################################################
# Temp Ext - Temp Int
df_clean['Temp_diff'] = df_clean['ExternalTemp_Value'] - df_clean['InternalTemp_Value']
# Put date-related features in the DF
df_clean['Weekday'] = df_clean['Date'].dt.dayofweek
df_clean['Hour'] = df_clean['Date'].dt.hour
df_clean['Month'] = df_clean['Date'].dt.month


# define work time
def work(x):
    if x['Weekday'] < 5 and (8 <= x['Hour'] <= 18):
        return 1
    else:
        return 0


df_clean['worktime'] = df_clean.apply(work, axis=1)
################################# PLOT FINAL DATA #########################################################################################
# %%
fig = plt.figure(figsize=(30, 10))
plt.scatter(df_clean.index, df_clean['ConsEnergy_Value'])
plt.show()
# %%
################################### SAVE RESULTS TO CSV FILE ##############################################################################
df_clean.to_csv(out_file, index=False)
