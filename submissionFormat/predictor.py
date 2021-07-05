import pandas as pd
import numpy as np
import joblib
def predictRuns(x):
    try:
        with open('cantdobetter.joblib','rb') as f:
            regressor =joblib.load(f)
        with open('venue.joblib','rb') as f:
            venue =joblib.load(f)
        with open('team.joblib','rb') as f:
            team =joblib.load(f)
        test=pd.read_csv(x)
        test["venue"]=venue.transform(test["venue"])
        test["batting_team"]=team.transform(test["batting_team"])
        test["bowling_team"]=team.transform(test["bowling_team"])
        test["date"]=2021
        test=test[["venue","innings","date","batting_team","bowling_team"]]
        testt=test.to_numpy()
        predicted=regressor.predict(testt)
            
    except:
        predicted=[45]
    
    return predicted[0]
