
EnergySignature = Input Data

DaRe_Energy_Main =  Script for data preparation.

                    Input -> Input Data from Energy Signature
                    Output -> .csv file containing elaborated data
                                            'EnergyClean/Energy_%s_%s_%i.csv' % (str(split), outliers_rmv, year)
                                            split = True || False
                                            outliers_rmv = 'ZS' || 'IQR' || 'NO'
                                            year = 2017 || 2018

DaRe_Energy_Random_Forest = Script for random forest model creation.

                            Input -> .csv file containing elaborated data
                            Output -> 1. 'Random_Forest_model.sav' containing the Random Forest model
                                      2. 'test_data.csv' containing the data used to test the model
                                      
DaRe_Energy_Predictor = Autonomous script for data prediction.

                        Input -> 1. Random Forest Model File
                                 2. Input Data to predict
                                 3. Output file
                                 
                                 usage: DaRe_Energy_Predictor.py <modelFile> <inFile> <outFile>
                                 for more instructions: DaRe_Energy_Predictor.py -h || --help
                                 
                        Output -> <outFile> containing the input data with added column for predicted values
                        
====================================================================================================
