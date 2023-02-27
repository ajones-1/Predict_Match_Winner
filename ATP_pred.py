# Predict the winner of a tennis match based on the first set  

# import packages
import os
from datetime import date
import numpy as np
import pandas as pd
import xgboost as xgb
import featuretools as ft
from dask.distributed import Client
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tracemalloc

# Import variables from another file
from config import MODEL_NAME, SAVE_CSV_TO, MAX_SET_NO, LIST_OF_FEATURES

print("Starting Script")

# Start tracking memory usage
tracemalloc.start()

# Define the model"s name
model_name = MODEL_NAME

# Get the current time
current_day = date.today()

# Define folder paths
folder_name = f"{model_name}_{current_day}"

save_csv_to = SAVE_CSV_TO

###############################################################################
# Start defining functions
###############################################################################

print("Functions defined")
###############################################################################
# End of defining functions
###############################################################################

# Defines an entry point
if __name__ == "__main__":
    
    # Loading data and drop extra columns
    pointsW2011 = pd.read_csv("./2011-wimbledon-points.csv") 
    pointsW2011 = pointsW2011.drop(["Serve_Direction", "Winner_FH", "Winner_BH"], axis=1)
    matchesW2011 = pd.read_csv("./2011-wimbledon-matches.csv")
    
    pointsF2011 = pd.read_csv("./2011-frenchopen-points.csv") 
    pointsF2011 = pointsF2011.drop(["Serve_Direction", "Winner_FH", "Winner_BH"], axis=1)
    matchesF2011 = pd.read_csv("./2011-frenchopen-matches.csv")
    
    pointsU2011 = pd.read_csv("./2011-usopen-points.csv") 
    pointsU2011 = pointsU2011.drop(["Serve_Direction", "Winner_FH", "Winner_BH", "ServingTo"], axis=1)
    matchesU2011 = pd.read_csv("./2011-usopen-matches.csv")
    
    pointsA2011 = pd.read_csv("./2011-ausopen-points.csv") 
    matchesA2011 = pd.read_csv("./2011-ausopen-matches.csv")
    
    points =  pd.concat([pointsW2011, pointsF2011, pointsU2011, pointsA2011])
    matches = pd.concat([matchesW2011, matchesF2011, matchesU2011, matchesA2011])
    
    # Remove retirement and walkover matches
    matches = matches[matches["status"]=="Complete"] 
    
    # Assign point_id
    points['point_id'] = range(1, len(points) + 1)
    points.set_index('point_id', inplace=False)
    
    # Restrict to first set 
    points = points[points["SetNo"] < MAX_SET_NO]
    
    # Carry out Deep Feature Synthesis 
    entities = {
    "matches" : (matches, "match_id"),
    "points" : (points,"point_id")}

    relationships = [("matches", "match_id", "points", "match_id")]

    entityset = ft.EntitySet("matches_points", entities, relationships)
    
    # Opening client required for running DFS in parallel
    client = Client(processes= True)
    cluster = client.cluster
    
    feature_matrix_matches, feature_defs = ft.dfs(entityset = entityset, 
                                                  target_entity="matches",
                                                  verbose = True, 
                                                  n_jobs=-1)
    
    # Closing client required for DFS
    client.close()
    
    feature_matrix_target = feature_matrix_matches[LIST_OF_FEATURES]
    
    # Add features manually
    feature_matrix_target["P1_1st_serve_percentage"] = \
    points.groupby("match_id").sum()["P1FirstSrvWon"] / \
    points.groupby("match_id").sum()["P1FirstSrvIn"]

    feature_matrix_target["P1_2nd_serve_percentage"] = \
    points.groupby("match_id").sum()["P1SecondSrvWon"] / \
    points.groupby("match_id").sum()["P1SecondSrvIn"]

    feature_matrix_target["P2_1st_serve_percentage"] = \
    points.groupby("match_id").sum()["P2FirstSrvWon"] / \
    points.groupby("match_id").sum()["P2FirstSrvIn"]

    feature_matrix_target["P2_2nd_serve_percentage"] = \
    points.groupby("match_id").sum()["P2SecondSrvWon"] / \
    points.groupby("match_id").sum()["P2SecondSrvIn"]
    
    # Make categorical variables for each slam
    feature_matrix_target['Wimbledon'] = np.where(feature_matrix_target["slam"] == 'wimbledon', 1, 0)
    feature_matrix_target["French_Open"] = np.where(feature_matrix_target["slam"] == 'frenchopen', 1, 0)
    feature_matrix_target["US_Open"] = np.where(feature_matrix_target["slam"] == 'usopen', 1, 0)
    feature_matrix_target["Aus_Open"] = np.where(feature_matrix_target["slam"] == 'ausopen', 1, 0)
    
    # Remove slam and winner features
    feature_matrix_target = feature_matrix_target.drop("slam", axis=1)
    feature_matrix = feature_matrix_target.drop("winner", axis=1)
    
    target = matches[["match_id", "winner"]]
    target = target.set_index("match_id")
    
    x_train, x_test, y_train, y_test = train_test_split(feature_matrix, 
                                                        target, 
                                                        test_size=0.3)
    
    # Setting up folder to a save data
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
        print("Folder set up")
    
    # Saving how the data has been split
    x_train.to_csv(save_csv_to + "x_train.csv")	
    y_train.to_csv(save_csv_to + "y_train.csv")	
    x_test.to_csv(save_csv_to + "x_test.csv")	
    y_test.to_csv(save_csv_to + "y_test.csv")	
    
    # Define the parameters for the XGBoost Random Forest Classifier model
    params = {"max_depth": 5,
              "learning_rate": 0.1,
              "n_estimators": 100,
              "objective": "binary:logistic",
              'eval_metric': 'error',
              "random_state": 42}
    
    # Train the model using the training data
    model = xgb.XGBRFClassifier(**params)
    
    # Turn y_train into a 1D-array
    y_train = y_train.squeeze().values
    model.fit(x_train, y_train)
    
    # Use the model to predict on the testing data
    y_pred = model.predict(x_test)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %.2f%%' % (accuracy * 100.0))
    
    print("The maximum memory usage of the script was " + 
          f"{tracemalloc.get_traced_memory()[1]}" + " bytes.")
    
    print("Starting Finished")