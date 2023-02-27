# Predict_Match_Winner
This is a machine learning model that predicts the winner of a tennis match based on the data from the first set. For each match the statistics like the number of aces, first serve percentage and which slam the match is at are calculated. Then using the XGBoost package a random forrest classifier model is built and evaluated against training data. The model predicts the winner of the match correctly 80.42% of the time.

List of important files:
ATP_pred.py - The model is trained and tested
config.py - Stores all the variables e.g. model name
Winner_based_on_first_set_2023-02-27 - Folder containing the model and the data it was trained and tested on
Accuracy.JPG - Picture of the result when the model is tested and maximum memory usage of the script

This model is far from perfect since it was only built over a weekend, it has lots of scope for improvement.

Steps that have been taken to improve the model:
1. Remove matches that ended in retirement or a walkover
2. Carry out deep feature sysnthesis to creat many different features to explore 
3. Added custom features like first and second serve percentage

Possible improvements:
1. Restict the model to just men or just women
2. Restrict the model to just one of the grand slams
3. Carry out feature selection by looking at the correlation between features and the target feature
4. Carry out hyperparamter optimisation 
5. Add more data 
6. Use a future year as a validation group to get a better result for how accurate the model is

The deep deature systhesis step is done in parallel so speed it up however, if the memory usage is too great this can be removed and the script will run slower but with a lower maximum memory usage.
The maximum memory usage is tracked using the tracemaloc package
