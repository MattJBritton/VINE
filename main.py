from __future__ import division

from collections import Counter, defaultdict
from itertools import product
import json
from math import sqrt
import sys
import time
import datetime

from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

#scikit-learn
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering, Birch

#for reproducibility
np.random.seed(21227)

#functions to parse included datasets
def load_abalone_dataset():
	data = pd.read_csv('data/abalone.csv')

	# encode sex str values
	le = LabelEncoder()
	le.fit(data['sex'].values)
	data['sex'] = data['sex'].apply(lambda x: le.transform([x])[0])	  
	
	X = data[list(data)[:-1]]
	y = data['rings']

	return X, y
def load_bike_dataset():
	def _datestr_to_timestamp(s):
		return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d").timetuple())

	data = pd.read_csv('data/bike.csv')
	data['dteday'] = data['dteday'].apply(_datestr_to_timestamp)

	features = ['dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',\
				'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

	X = data[features]
	y = data['cnt']

	return X, y

def load_dataset(name="bike"):
	
	if name =="bike":
		X,y = load_bike_dataset()
	elif name=="abalone":
		X,y = load_abalone_dataset()
	gbm = GradientBoostingRegressor()
	gbm.fit(X, y) 
	return X, gbm

#from original PyCEBox library
#get the x_values for a given granularity of curve
def _get_grid_points(x, num_grid_points):
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
	return dS

#interpolate lines to num_grid_points when comparing features for feature-space statistics
def _interpolate_line(x, y, length):
	if len(y) == length:
		return y
	else:
		f = interp1d(x,y, kind="cubic")
		return list(f(np.linspace(x[0], x[-1], num=length, endpoint=True)))


#main function - run this to export JSON file for vis
def export(data, predict_func, centered=False, num_clusters=5,\
	num_grid_points=20, ice_curves_to_export=100):

	export_dict = {"features":{}}

	#1-node decision tree to get best split for each cluster
	model = DecisionTreeClassifier(criterion="entropy", max_depth=1, presort=True,\
		class_weight="balanced")
	
	data["id"] = data.index #index must be set as field since pivot_table function cannot access index
	
	#generate data for one ICE plot per column
	for column_of_interest in data.columns:
	
		x_s = _get_grid_points(data[column_of_interest], num_grid_points)
		if len(x_s) <= 5 or column_of_interest == "id": 
			#do not generate ICE plots for categorical features with few values
			continue		
			
		#create dataframe with synthetic points (one for each x value returned by _get_grid_points)
		ice_data = pd.DataFrame(np.repeat(data.values, x_s.size, axis=0), columns=data.columns)
		ice_data = ice_data.reset_index(drop=True).set_index("id")
		ice_data[column_of_interest] = np.tile(x_s, data.shape[0])
		ice_data['ice_y'] = predict_func(ice_data.values)
		ice_data = ice_data.reset_index().pivot_table(values="ice_y", index="id",columns=column_of_interest)

		#center all curves at the min point of the feature's range
		if centered:
			ice_data = ice_data.sub(ice_data[ice_data.columns.values[0]], axis='index')

		export_dict["features"][column_of_interest] = {"feature_name": column_of_interest,
													   "x_values": list(x_s),
													   "pdp_line": list(pdp(ice_data)),
													   "importance": np.mean(pdp(ice_data)),
													   "clusters":[]
													  }
		#perform clustering
		ice_data["cluster_label"] = Birch(n_clusters = num_clusters, threshold=0.1)\
					.fit(_differentiate(ice_data.values)).labels_  

		ice_data["points"] = ice_data[x_s].values.tolist() 

		#generate all the ICE curves per cluster
		all_curves_by_cluster = ice_data.groupby("cluster_label")["points"].apply(lambda x: np.array(x)) 
		#average the above to get the mean cluster line
		cluster_average_curves = [np.mean(np.array(list(value)), axis=0)\
								  for _,value in all_curves_by_cluster.iteritems()]
		for cluster_num in range(num_clusters):
								  
			num_curves_in_cluster = len(all_curves_by_cluster[cluster_num])

			#build model to predict cluster membership
			rdwcY = ice_data["cluster_label"].apply(lambda x: 1 if x==cluster_num else 0)
			model.fit(data.drop("id", axis=1), rdwcY)
			predY = model.predict(data.drop("id", axis=1))
			
			#get random curves if there are more than 100
			#no reason to make the visualization display 1000+ curves for large datasets
			if num_curves_in_cluster > ice_curves_to_export:
				individual_ice_samples = [list(x) for x in\
										  list(all_curves_by_cluster[cluster_num]\
										  [np.random.choice(num_curves_in_cluster,\
												 size=ice_curves_to_export, replace=False)])
										 ]
			else:
				individual_ice_samples = [list(x) for x in\
										  list(all_curves_by_cluster[cluster_num])\
										 ]
			
			#add cluster-level metrics to the JSON file
			confusion_matrix = metrics.confusion_matrix(rdwcY, predY)
			export_dict["features"][column_of_interest]["clusters"].append({
				'accuracy': int(round(100.*metrics.accuracy_score(rdwcY, predY))),
				'precision': int(round(100.*metrics.precision_score(rdwcY, predY))),
				'recall': int(round(100.*metrics.recall_score(rdwcY, predY))),
				'confusion_matrix': {"true_negative":confusion_matrix[0,0],
									"false_positive":confusion_matrix[0,1],
									"false_negative":confusion_matrix[1,0],
									"true_positive":confusion_matrix[1,1]},
				'split_feature': data.columns[model.tree_.feature[0]],
				'split_val': round(model.tree_.threshold[0],2),
				'split_direction': "<=" if model.classes_[np.argmax(model.tree_.value[1])] == 1 else ">",
				'cluster_size': num_curves_in_cluster,
				'line': list(cluster_average_curves[cluster_num]),
				'individual_ice_curves': individual_ice_samples
			})
		#feature-level calculation for cluster distance to pdp
		feature_val = export_dict["features"][column_of_interest]
		feature_val["cluster_deviation"]\
		= np.mean([_get_dtw_distance(np.array(feature_val["pdp_line"]),\
												  np.array(x["line"]))\
				  for x in feature_val["clusters"]])/np.mean(np.array(feature_val["pdp_line"]))
		
	#dataset-level characteristics
	#these drive the position of the feature chart in the vis overview screen
	feature_list = [(key, value) for key,value in export_dict["features"].iteritems()]
	pdpLines = np.vstack([np.array(_interpolate_line(v[1]["x_values"], v[1]["pdp_line"], num_grid_points ))\
				for v in feature_list])
	featureDistanceArray = cdist(pdpLines, pdpLines, 'euclidean')
	featureDistanceObjects = []
	range_features = range(len(export_dict["features"].keys()))
	for pair in product(*[range_features, range_features]):
		if pair[0]>pair[1]:
			featureDistanceObjects.append({"source": feature_list[pair[0]][0], 
										   "target":feature_list[pair[1]][0], 
										   "value":featureDistanceArray[pair[0], pair[1]]})
	export_dict["feature_distance_links"] = featureDistanceObjects
	
	with open('vis/static/data.json', 'w') as outfile:
		json.dump(export_dict, outfile)		

if __name__== "__main__":
	dataset_name = sys.argv[1]
	num_clusters = int(sys.argv[2])
	num_grid_points = int(sys.argv[3])
	dfX, predictor = load_dataset(name=dataset_name)
	export(
		data=dfX, 
		predict_func = predictor.predict,
		centered=True, 
		num_clusters=num_clusters,
		num_grid_points=num_grid_points
	)

