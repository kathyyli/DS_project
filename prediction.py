
# coding: utf-8

# In[6]:


import flask
from flask import Flask, request
import json
import numpy as np
import scipy
import os
import sklearn as skl
import pandas as pd
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


# In[7]:


app = Flask(__name__)
def api():
    print('api processing')
    if request.is_json:
        input_data = request.get_json()
    else:
        # if not json post, return.
        return ''
    
    # debug input json data.
        print(input_data)

        Y_prediction = None
        try:
            Y_prediction = prediction(input_data['song_name'],
                        input_data['genre'],
                        float(input_data['latitude']),
                        float(input_data['longtitude']),
                        float(input_data['bit_rate']),
                        float(input_data['duration']),
                        float(input_data['acousticness']),
                        float(input_data['danceability']),
                        float(input_data['energy']),
                        float(input_data['instrumentalness']),
                        float(input_data['liveness']),
                        float(input_data['speechiness']),
                        float(input_data['tempo']),
                        float(input_data['valence']),
                        float(input_data['artist_h']))
        except Exception as ex:
            Y_prediction = 'Exception raised'

        if isinstance(Y_prediction, str):
            _output_data = {'result': Y_prediction}
        else:
            print(type(Y_prediction))
            _output_data = {'result': str(int(Y_prediction[0]))}

        # 0 to 'Not popular' and 1 to 'Popular'.
        output_data = _output_data
        if _output_data['result'] == '0':
            output_data['result'] = 'Not popular'
        elif _output_data['result'] == '1':
            output_data['result'] = 'Popular'
        else:
            pass
         # This is the REST response.
        response = flask.make_response(json.JSONEncoder().encode(output_data))
        response.headers['content-type'] = 'application/json'
        return response

def prediction(genre,latitude, longitude, bit_rate, duration, acousticness, danceability,energy,instrumentalness,liveness, speechiness,tempo, valence,artist_hotttnesss,song_name):
    

    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(song_name)
    senti_neg=ss['neg']
    senti_pos=ss['pos']
    newline=pd.DataFrame({'latitude':latitude, 'longitude':longitude, 'bit_rate':bit_rate, 'duration':duration, 'acousticness':acousticness,
            'danceability':danceability, 'energy':energy, 'instrumentalness':instrumentalness,'liveness':liveness,
             'speechiness':speechiness,'tempo':tempo, 'valence':valence,'artist_hotttnesss':artist_hotttnesss,
             'senti neg':senti_neg,'senti pos':senti_pos}, index=[372])
    
    
    if genre == 'Rock':
        df = pd.read_csv('Rock_10.csv', dtype=float)
        df.drop(['track_ID','track_listens', 'artist_discovery','artist_familiarity'],axis=1,inplace=True)
        
        y=df[['popular']].values.ravel()
        df1 = pd.concat([df,newline])

        X=df1.drop(['popular'],axis=1)

        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train = X_scaled[:-1,:]
        X_test = X_scaled[-1,:].reshape(1, -1)
        

        
        rfc=RandomForestClassifier()
        x_train, x_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2, random_state=7)
        param_grid = { 'max_depth' : np.arange(1,10), 'max_features' : np.arange(1,10)}
        CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
        CV_rfc.fit(x_train, y_train)
        CV_rfc.best_params_

        random_forest = RandomForestClassifier(n_estimators=100,max_depth=CV_rfc.best_params_['max_depth'],max_features=CV_rfc.best_params_['max_features'])
        random_forest.fit(X_train, y)


        Y_prediction = random_forest.predict(X_test)
    if genre == 'Hiphop':
        df = pd.read_csv('Hiphop_10.csv', dtype=float)
        df.drop(['track_ID','track_listens', 'artist_discovery','artist_familiarity'],axis=1,inplace=True)
        
        y=df[['popular']].values.ravel()
        df1 = pd.concat([df,newline])

        X=df1.drop(['popular'],axis=1)

        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train = X_scaled[:-1,:]
        X_test = X_scaled[-1,:].reshape(1, -1)
        

        x_train, x_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2, random_state=7)
        # Regularize over L1 penalty 
        C_vals = np.logspace(-4,0,100)
        scores = []
        for C_val in C_vals:

            #change penalty to l1
            regr = LogisticRegression(penalty='l1', C = C_val)
            regr.fit(x_train, y_train)

            probas_ = regr.fit(x_train, y_train).predict_proba(x_test)

            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            roc_auc = auc(fpr, tpr)

            scores.append(roc_auc)

        C_best_L1 = C_vals[scores.index(max(scores))]
  
        regr = LogisticRegression(penalty='l1',C=C_best_L1)
        regr.fit(X_train, y)


        Y_prediction = regr.predict(X_test)
    if genre == 'Electronic':
        df = pd.read_csv('Elec_10.csv', dtype=float)
        df.drop(['track_ID','track_listens', 'artist_discovery','artist_familiarity'],axis=1,inplace=True)
        
        y=df[['popular']].values.ravel()
        df1 = pd.concat([df,newline])

        X=df1.drop(['popular'],axis=1)

        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train = X_scaled[:-1,:]
        X_test = X_scaled[-1,:].reshape(1, -1)
        

        x_train, x_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2, random_state=7)
        # Regularize over L1 penalty 
        C_vals = np.logspace(-4,0,100)
        scores = []
        for C_val in C_vals:

            #change penalty to l1
            regr = LogisticRegression(penalty='l1', C = C_val)
            regr.fit(x_train, y_train)

            probas_ = regr.fit(x_train, y_train).predict_proba(x_test)

            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            roc_auc = auc(fpr, tpr)

            scores.append(roc_auc)

        C_best_L1 = C_vals[scores.index(max(scores))]
  
        regr = LogisticRegression(penalty='l1',C=C_best_L1)
        regr.fit(X_train, y)


        Y_prediction = regr.predict(X_test)
    return(Y_prediction)

