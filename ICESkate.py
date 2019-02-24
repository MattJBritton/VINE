
# coding: utf-8

from __future__ import division

from collections import Counter, defaultdict
from itertools import product
import json
import sys
from math import sqrt
import time
import datetime

from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

#scikit-learn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn import datasets


#for reproducibility
np.random.seed(224)
#functions to parse included datasets
def load_bike_dataset():
    def _datestr_to_timestamp(s):
        return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d").timetuple())

    data = pd.read_csv('data/bike.csv')
    data['dteday'] = data['dteday'].apply(_datestr_to_timestamp)
    data = pd.get_dummies(data, prefix=["weathersit"], columns=["weathersit"], drop_first=False)

    features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',\
    'weathersit_2', 'weathersit_3', 'weathersit_4', 'temp', 'atemp', 'hum', 'windspeed']

    X = data[features]
    y = data['cnt']

    return X, y

def load_diabetes_dataset():
    diabetes_dataset = datasets.load_diabetes()

    return pd.DataFrame(diabetes_dataset.data, columns=diabetes_dataset.feature_names), diabetes_dataset.target  

def load_boston_dataset():
    boston_dataset = datasets.load_boston()

    return pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names), boston_dataset.target    

def load_california_dataset():
    california_dataset = datasets.fetch_california_housing()

    return pd.DataFrame(california_dataset.data, columns=california_dataset.feature_names),california_dataset.target

def load_dataset(name="bike"):
    
    if name =="bike":
        X,y = load_bike_dataset()
    elif name=="boston":
        X,y = load_boston_dataset()   
    elif name=="california":
        X,y = load_california_dataset()   
    elif name=="diabetes":
        X,y = load_diabetes_dataset()           
    gbm = GradientBoostingRegressor(min_samples_leaf=10)
    gbm.fit(X, y) 
    return X, y, gbm

#from original PyCEBox library
#get the x_values for a given granularity of curve
def _get_grid_points(x, num_grid_points):
    if sorted(list(x.unique())) == [0,1]:
        return [0.,1.]
    if num_grid_points is None:
        return x.unique()
    else:
        # unique is necessary, because if num_grid_points is too much larger
        # than x.shape[0], there will be duplicate quantiles (even with
        # interpolation)
        return x.quantile(np.linspace(0, 1, num_grid_points)).unique()

#from original PyCEBox library
#average the PDP lines (naive method seems to work fine)
def pdp(ice_data):
    return ice_data.mean(axis=0)

# from http://nbviewer.jupyter.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb
def _default_factory():
    return float("inf")

def _get_dtw_distance(s1,s2, w=4):
    
    w = max(w, abs(len(s1)-len(s2)))
    DTW = defaultdict(_default_factory)
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            DTW[(i, j)] = (s1[i]-s2[j])**2 + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            
    return sqrt(DTW[len(s1)-1, len(s2)-1])

#adapted from https://www.datagrapple.com/Tech/GNPR-tutorial-How-to-cluster-random-walks.html
#transform curves before distance measurement
def _differentiate(series):
        
    dS = np.diff(series)
    #if series.shape[1] == 2:
        #return dS
    return dS
    #return np.concatenate([dS, (series/np.mean(series))[:,::4]], axis=1)

#interpolate lines to num_grid_points when comparing features for feature-space statistics
def _interpolate_line(x, y, length):
    if len(y) == length:
        return y
    else:
        f = interp1d(x,y, kind="cubic")
        return list(f(np.linspace(x[0], x[-1], num=length, endpoint=True)))

#main function - run this to export JSON file for vis
def export(data, y, predict_func, num_clusters=5, num_grid_points=40,\
    ice_curves_to_export=100):
    
    #generate data for one ICE plot per column
    for column_of_interest in data.columns:

        ice_data = pd.DataFrame(np.ones(data.shape[0]), columns=["temp"])
    
        x_s = _get_grid_points(data[column_of_interest], num_grid_points)      
            
        #create dataframe with synthetic points (one for each x value returned by _get_grid_points)
        for x_val in x_s:
            kwargs = {column_of_interest : x_val}
            ice_data[x_val] = predict_func(data.assign(**kwargs))
        
        ice_data.drop("temp", axis=1, inplace=True)
        
        #center all curves at the min point of the feature's range
        ice_data = ice_data.sub(ice_data.mean(axis=1), axis='index')

        pdp_data = pdp(ice_data)
        hist_counts, hist_bins = np.histogram(a=np.array(data.loc[:,column_of_interest]), bins="auto")
        hist_zip = [{"x":x[0], "y":x[1]} for x in zip(hist_bins, hist_counts)]
        export_dict["distributions"][column_of_interest] = hist_zip
        export_dict["features"][column_of_interest] = {"feature_name": column_of_interest,
                                                       "x_values": list(x_s),
                                                       "pdp_line": list(pdp_data),
                                                       "importance": np.mean(pdp_data),
                                                       "clusters":[]
                                                      }
        #perform clustering
        ice_data["cluster_label"] = Birch(n_clusters = num_clusters, threshold=0.1).fit(_differentiate(ice_data.values)).labels_
        ice_data["points"] = ice_data[x_s].values.tolist() 

        #generate all the ICE curves per cluster
        all_curves_by_cluster = ice_data.groupby("cluster_label")["points"].apply(lambda x: np.array(x)) 
        #average the above to get the mean cluster line
        cluster_average_curves = [np.mean(np.array(list(value)), axis=0) for _,value in all_curves_by_cluster.iteritems()]
        
        for cluster_num in range(len(all_curves_by_cluster)):                          
            num_curves_in_cluster = len(all_curves_by_cluster[cluster_num])

            #build model to predict cluster membership
            rdwcY = ice_data["cluster_label"].apply(lambda x: 1 if x==cluster_num else 0)
            #1-node decision tree to get best split for each cluster
            model = DecisionTreeClassifier(criterion="entropy", max_depth=1, presort=False, class_weight="balanced")
            model.fit(data, rdwcY)
            predY = model.predict(data)              
            
            #get random curves if there are more than 100
            #no reason to make the visualization display 1000+ curves for large datasets
            if num_curves_in_cluster > ice_curves_to_export:
                individual_ice_samples = [list(x) for x in\
                list(all_curves_by_cluster[cluster_num]\
                    [np.random.choice(num_curves_in_cluster, size=ice_curves_to_export, replace=False)])
                                         ]
            else:
                individual_ice_samples = [list(x) for x in list(all_curves_by_cluster[cluster_num])]
            
            #add cluster-level metrics to the JSON file
            export_dict["features"][column_of_interest]["clusters"].append({
                'accuracy': int(round(100.*metrics.accuracy_score(rdwcY, predY))),
                'precision': int(round(100.*metrics.precision_score(rdwcY, predY))),
                'recall': int(round(100.*metrics.recall_score(rdwcY, predY))),
                'split_feature': data.columns[model.tree_.feature[0]]\
                if model.tree_.value.shape[0] > 1 else 'none',
                'split_val': round(model.tree_.threshold[0],2),
                'split_direction': "<=" if model.tree_.value.shape[0] == 1 or\
                model.classes_[np.argmax(model.tree_.value[1])] == 1 else ">",
                'cluster_size': num_curves_in_cluster,
                'line': list(cluster_average_curves[cluster_num]),
                'individual_ice_curves': individual_ice_samples
            })              
        
        #feature-level calculation for cluster distance to pdp
        feature_val = export_dict["features"][column_of_interest]
        feature_val["cluster_deviation"]        = np.mean([_get_dtw_distance(np.array(feature_val["pdp_line"]),                                                  np.array(x["line"]))                  for x in feature_val["clusters"]])/np.mean(np.array(feature_val["pdp_line"]))
        if np.isnan(feature_val["cluster_deviation"]) or feature_val["cluster_deviation"]==float("inf"):
            feature_val["cluster_deviation"] = 0
        #EOF feature loop
        
    with open('static/data.json', 'w') as outfile:
        json.dump(export_dict, outfile)        



if __name__== "__main__":
    dataset_name = sys.argv[1]
    num_clusters = int(sys.argv[2])
    num_grid_points = int(sys.argv[3])
    dfX, y, predictor = load_dataset(name=dataset_name)
    export_dict = {"features":{}, "distributions":{}}
    export(dfX, y, predictor.predict, num_clusters=num_clusters, num_grid_points=num_grid_points,\
        ice_curves_to_export=50)



