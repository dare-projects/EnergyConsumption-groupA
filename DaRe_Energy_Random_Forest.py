import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import matplotlib.pyplot as plt
import numpy as np

# this is the best dataset
split = True
outliers_rmv = 'ZS'
    
file_2017 = 'EnergyClean\Energy_%s_%s_2017.csv' % (str(split), outliers_rmv)
file_2018 = 'EnergyClean\Energy_%s_%s_2018.csv' % (str(split), outliers_rmv)

df_2017 = pd.read_csv(file_2017, index_col=False)
df_2018 = pd.read_csv(file_2018, index_col=False)
    
df = df_2017.append(df_2018)

############################## DATA SPLITTING ####################################
######################### x = features, y = targets ##############################
#features array
features = ['Irradiance_Value'
            ,'ExternalTemp_Value'
            #,'InternalTemp_Value'
            #,'Temp_diff'
            ,'Weekday'
            ,'Month'
            ,'Hour'
            #,'worktime'
            #,'HVAC_Value'
            #,'Lightning_Value'
            ]
n_features = len(features)

#independent variables
x = df.loc[:, features].values
#extract dependent variable
y = df.loc[:, 'ConsEnergy_Value'].values

#split train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 21)

#scaling features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#%%#
########################### RANDOM FOREST REGRESSOR ########################################

#random forest regressor
regressor = RandomForestRegressor(n_jobs=-1, verbose=2,
                                  criterion='mse',              #default value
                                  min_impurity_decrease=0.,     #default value
                                  bootstrap=True,               #default value
                                  warm_start=False,             #default value
                                  oob_score=False,              #default value
                                  random_state=None,            #default value
                                  max_depth=400,                
                                  min_weight_fraction_leaf=0.,  #default value
                                  max_leaf_nodes=None,          #default value
                                  max_features=4,
                                  min_samples_leaf=1,
                                  min_samples_split=3,
                                  n_estimators = 200)

# Build a forest of trees from the training set (x, y)
regressor.fit(x_train, y_train)
# Predict regression target for X.
y_pred = regressor.predict(x_test)
# Return the feature importances (the higher, the more important the feature)
feature_importances = pd.DataFrame(regressor.feature_importances_, index = features, columns=['importance'])
# Returns the coefficient of determination R^2 of the prediction.
r2_score = regressor.score(x_test, y_test)
print('\nRandom Forest Regressor R2 Score: ', r2_score)

# calculate error margin between prediction and actual values
data_difference = (y_test - y_pred)
data_difference = pd.DataFrame(data=data_difference, columns=['difference'])
# create arrays for plotting
index_np = np.array(data_difference.index)
values_np = np.array(data_difference['difference'])
#%%#
# calculate mean error on predictions
mean_error = np.mean(np.abs(data_difference))[0]
m2_error = np.mean(data_difference**2)[0]

fig = plt.figure(figsize =(15 , 10))

min_val = int(np.min(values_np))
max_val = int(np.max(values_np))

label = ('''mean error: %0.2f
mean squared error: %0.2f 
r2 score: %0.2f ''')

scores = label % (mean_error, m2_error, r2_score)
plt.hist(values_np,
         width=1.9,
         bins=(np.arange(min_val, max_val, step=2)))
plt.xticks(np.arange(np.min(values_np),np.max(values_np), step=2, dtype=int))
fig.suptitle('Histogram of error margin between predicted and actual data', fontsize=20)
plt.title(scores)
plt.xlabel('Actual value - Predicted value')
plt.savefig('Plots/testplot.png')
plt.show()
#%%#

########################### SAVE MODEL TO FILE ###################################

# save random forest regressor model to file
filename = 'Random_Forest_model.sav'
pickle.dump(regressor, open(filename, 'wb'))

# save test dataframe to csv
df_x_test = pd.DataFrame(data=x_test, columns=features)
df_y_test = pd.DataFrame(data=y_test, columns=['ConsEnergy_Value'])
df_test = df_x_test.join(df_y_test, how='outer')
out_file = 'test_data.csv'
df_test.to_csv(out_file, index = False)
