# -*- coding: utf-8 -*-
"""
Created on Fri Nov 2 17:48:18 2018

@author: matthewottomano
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from surprise import BaselineOnly
from surprise import Reader
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
import os

# path to dataset file
file_path = os.path.expanduser('~/Desktop/memes_data.txt') 

# As we're loading a custom dataset, we need to define a reader.
# Each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format='item user rating', sep=',')

data = Dataset.load_from_file(file_path, reader=reader)
print(data) 

# Load the dataset  UserID::MemeID::Rating::Timestamp
#data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.2,shuffle=True)  


# Use user_based true/false to switch between user-based or item-based collaborative filtering
algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
algo.fit(trainset)

 #we can now query for specific predicions
allpred = {}
uid = 'helloworld'  # raw user id
iid = str(1)  # raw item id
for i in range(20):
    iid = str(i+1)
    pred = algo.predict(uid, iid, verbose=True)
    allpred[pred.iid] = pred.est 
    
    
    
    

 #get a prediction for specific users and items.
#pred = algo.predict(uid, iid, r_ui=4, verbose=True)


# run the trained model against the testset
#test_pred = algo.test(testset)
#print(test_pred) 

# get RMSE
#print("User-based Model : Test Set")
#accuracy.rmse(test_pred, verbose=True)

# if you wanted to evaluate on the trainset
#print("User-based Model : Training Set")
#train_pred = algo.test(trainset.build_testset())
#accuracy.rmse(train_pred)