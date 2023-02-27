# Predict_Match_Winner
A machine learning model that predicts the winner of a tennis match based on the first set.

This model is far from perfect since it was only built over a weekend, it has lots of scope for improvement.
The model is a random forrest classifier model made using the XGBoost package.
The model predicts the winner of the match correctly 80.42% of the time.

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
