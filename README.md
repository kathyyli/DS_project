# DS_project

# Data info
Data for project can be found at: https://os.unil.cloud.switch.ch/fma/fma_metadata.zip - pulls sound features and social features from Echo Nest API (linked below), and track metadata.

Explanation of 'echonest.csv' features: http://docs.echonest.com.s3-website-us-east-1.amazonaws.com/artist.html#hotttnesss.

Songs from: https://freemusicarchive.org/

File key:
'DS project v4.ipynb' - updated notebook with data analysis and model training / selection

'DSP.py' - python file for prediction

'Web App Script.txt' - web app script

'sentimental_analysis_title.csv' - sentiment analysis of tracks

Our website is at http://rlx.pythonanywhere.com/. 

# Data cleaning and preparation
-Combine tracks and echonest databases

-Extract year released from the track release date

-Select year with most songs, and genres with most songs within that year 

-Change the 'track_listens' variable into popular or not, in order to do classification

-Check for correlation between variables and scale data

# Modeling
-First use a baseline model, including genre as a variable and finding that genre is an important feature

-Split datasets by genre (three separate datasets for rock, hip-hop, and electronic for 2010)

-Test logistic regression with penalty (L1, L2), decision tree, random forest, and knn, using GridSearch to search over parameter values

-Compare AUC ROC on random subsets of data and for each genre select optimal model
