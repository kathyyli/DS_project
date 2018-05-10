
# A very simple Flask Hello World app for you to get started with...

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
from sklearn import preprocessing
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


app = Flask(__name__)

# @app.route('/hello_world')
# def hello_world():
#     return "Hello World"

@app.route('/api', methods=['POST'])
def api():
    print('api processing')
    if request.is_json:
        input_data = request.get_json()
    else:
        # if not json post, return.
        return ''

    # fake input.
    # input_data = {'genre': 'Rock', 'song_name': 'I WANNA SLEEP', 'artist_h': '0.414522', 'latitude': '41.87194', 'longtitude': '12.56738', 'duration': '209', 'bit_rate': '320000', 'tempo': '124.293', 'acousticness': '0.403708029', 'danceability': '0.679438214', 'speechiness': '0.033102901', 'liveness': '0.347895294', 'energy': '0.732271232', 'instrumentalness': '0.751735226', 'valence': '0.826063122'}
    # fake input end.

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


def senti(title):
    title='die'

    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(title)
    score=[]
    for k in ss:
        a=ss[k]
        score.append(a)
    senti_neg=score[0]
    senti_pos=score[2]
    return senti_neg, senti_pos

def prediction(title,genre,latitude, longitude, bit_rate, duration, acousticness, danceability,energy,instrumentalness,liveness, speechiness,tempo, valence,artist_hotttnesss):
    senti_analy=senti(title)
    senti_neg=senti_analy[0]
    senti_pos=senti_analy[1]

    if genre == 'Rock':
        # df = pd.read_csv('mysite/Rock_10.csv', header=[0],dtype=float)
        df = pd.read_csv('mysite/Rock_10.csv', dtype=float)
        df.drop(['track_ID','track_listens', 'artist_discovery','artist_familiarity','Z'],axis=1,inplace=True)
    elif genre == 'Hip-Pop':
        # df = pd.read_csv('mysite/Hiphop_10.csv', header=[0],dtype=float)
        df = pd.read_csv('mysite/Hiphop_10.csv', dtype=float)
        df.drop(['track_ID','track_listens','artist_discovery','artist_familiarity','Z'],axis=1,inplace=True)
    elif genre == 'Electronic':
        # df = pd.read_csv('mysite/Elec_10.csv', header=[0],dtype=float)
        df = pd.read_csv('mysite/Elec_10.csv', dtype=float)
        df.drop(['track_ID','track_listens','artist_discovery','artist_familiarity','Z'],axis=1,inplace=True)
    else:
        return 'Do not find genre'

    y=df[['popular']].values.ravel()

    newline=pd.DataFrame({'latitude':latitude, 'longitude':longitude, 'bit_rate':bit_rate, 'duration':duration, 'acousticness':acousticness, 'danceability':danceability, 'energy':energy, 'instrumentalness':instrumentalness,'liveness':liveness, 'speechiness':speechiness,'tempo':tempo, 'valence':valence,'artist_hotttnesss':artist_hotttnesss, 'senti neg':senti_neg,'senti pos':senti_pos}, index=[372])

    df1 = pd.concat([df,newline])

    X=df1.drop(['popular'],axis=1)

    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train = X_scaled[:-1,:]
    X_test = X_scaled[-1,:].reshape(1, -1)

    rfc=RandomForestClassifier()
    param_grid = { 'max_depth' : np.arange(1,10), 'max_features' : np.arange(1,10)}
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    CV_rfc.fit(X_train, y)
    CV_rfc.best_params_

    random_forest = RandomForestClassifier(n_estimators=100,max_depth=CV_rfc.best_params_['max_depth'],max_features=CV_rfc.best_params_['max_features'])
    random_forest.fit(X_train, y)

    return random_forest.predict(X_test)


