
# coding: utf-8

# In[1]:


import numpy as np
import scipy
import os
import sklearn as skl
import pandas as pd
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing


# In[ ]:


def prediction(genre,latitude, longitude, bit_rate, duration, acousticness, danceability,energy,instrumentalness,liveness,
             speechiness,tempo, valence,artist_hotttnesss,senti neg, senti pos):
    if genre == 'Rock':
        df = pd.read_csv('Rock_10.csv', header=[0],dtype=float)
        df.drop(['track_ID','track_listens', 'artist_discovery','artist_familiarity'],axis=1,inplace=True)
    elif genre == 'Hiphop':
        df = pd.read_csv('Hiphop_10.csv', header=[0],dtype=float)
        df.drop(['track_ID','track_listens','artist_discovery','artist_familiarity'],axis=1,inplace=True)
    elif genre == 'Elec':
        df = pd.read_csv('Elec_10.csv', header=[0],dtype=float)
        df.drop(['track_ID','track_listens','artist_discovery','artist_familiarity'],axis=1,inplace=True)
    
    
    X=df.drop(['popular'],axis=1).values
    y=df.[['popular']].values.ravel()
    X.append({'latitude':latitude, 'longitude':longitude, 'bit_rate':bit_rate, 'duration':duration, 'acousticness':acousticness,
            'danceability':danceability, 'energy':energy, 'instrumentalness':instrumentalness,'liveness':liveness,
             'speechiness':speechiness,'tempo':tempo, 'valence':valence,'artist_hotttnesss':artist_hotttnesss,
             'senti neg':senti neg,'senti pos':senti pos}, ignore_index=True)
   
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train = X_scaled[:-1]
    X_test = X_scaled.iloc[-1]


    rfc=RandomForestClassifier()
    param_grid = { 'max_depth' : np.arange(1,10), 'max_features' : np.arange(1,10)}
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    CV_rfc.fit(X_train, y)
    CV_rfc.best_params_
    
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y)

    Y_prediction = random_forest.predict(X_test)
    return(Y_prediction)

