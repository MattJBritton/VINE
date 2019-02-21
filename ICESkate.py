
# coding: utf-8

# In[2212]:

from __future__ import division

from collections import Counter, defaultdict
from itertools import product
import json
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


# In[2213]:

#for reproducibility
np.random.seed(2019)
#functions to parse included datasets
def load_bike_dataset():
    def _datestr_to_timestamp(s):
        return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d").timetuple())

    data = pd.read_csv('data/bike.csv')
    data['dteday'] = data['dteday'].apply(_datestr_to_timestamp)
    data = pd.get_dummies(data, prefix=["weathersit"], columns=["weathersit"], drop_first=False)

    features = ['season',                'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',                'weathersit_2', 'weathersit_3', 'weathersit_4',                'temp', 'atemp', 'hum', 'windspeed']

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


# In[2214]:

aX, aY, aPred = load_dataset("diabetes")
metrics.r2_score(aY, aPred.predict(aX))


# In[1093]:

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


# In[4]:

#from original PyCEBox library
#average the PDP lines (naive method seems to work fine)
def pdp(ice_data):
    return ice_data.mean(axis=0)


# In[5]:

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


# In[1765]:

#adapted from https://www.datagrapple.com/Tech/GNPR-tutorial-How-to-cluster-random-walks.html
#transform curves before distance measurement
def _differentiate(series):
        
    dS = np.diff(series)
    #if series.shape[1] == 2:
        #return dS
    return dS
    #return np.concatenate([dS, (series/np.mean(series))[:,::4]], axis=1)


# In[7]:

#method for testing random clusters to ensure that algorithm performance is superior
def _test_random_clusters(ice_data, num_clusters=5):
    temp = np.random.uniform(size=num_clusters)
    distribution = temp/temp.sum()
    cluster_labels = np.random.choice(a = range(num_clusters),                                      size=ice_data.shape[0],                                      replace=True, p=distribution)
    return cluster_labels


# In[8]:

#interpolate lines to num_grid_points when comparing features for feature-space statistics
def _interpolate_line(x, y, length):
    if len(y) == length:
        return y
    else:
        f = interp1d(x,y, kind="cubic")
        return list(f(np.linspace(x[0], x[-1], num=length, endpoint=True)))


# In[597]:

from sklearn.tree import _tree

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print "def tree({}):".format(", ".join(feature_names))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print "{}if {} <= {}:".format(indent, name, threshold)
            recurse(tree_.children_left[node], depth + 1)
            print "{}else:  # if {} > {}".format(indent, name, threshold)
            recurse(tree_.children_right[node], depth + 1)
        else:
            print "{}return {}".format(indent,np.argmax(tree_.value[node]))

    recurse(0, 1)


# In[2200]:

#main function - run this to export JSON file for vis
def export(data, y, predict_func, num_clusters=5, num_grid_points=40,           ice_curves_to_export=100, export_type="vis"):
    
    #generate data for one ICE plot per column
    for column_of_interest in data.columns:
        
        print column_of_interest

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
        if export_type == "vis":
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
        ice_data["cluster_label"] = Birch(n_clusters = num_clusters, threshold=0.1)                    .fit(_differentiate(ice_data.values)).labels_
        ice_data["points"] = ice_data[x_s].values.tolist() 

        #generate all the ICE curves per cluster
        all_curves_by_cluster = ice_data.groupby("cluster_label")["points"].apply(lambda x: np.array(x)) 
        #average the above to get the mean cluster line
        cluster_average_curves = [np.mean(np.array(list(value)), axis=0)                                  for _,value in all_curves_by_cluster.iteritems()]
        
        for cluster_num in range(len(all_curves_by_cluster)):                          
            num_curves_in_cluster = len(all_curves_by_cluster[cluster_num])

            #build model to predict cluster membership
            rdwcY = ice_data["cluster_label"].apply(lambda x: 1 if x==cluster_num else 0)
            #1-node decision tree to get best split for each cluster
            model = DecisionTreeClassifier(criterion="entropy", max_depth=1, presort=False,                                           class_weight="balanced")
            model.fit(data, rdwcY)
            predY = model.predict(data)              
            
            #get random curves if there are more than 100
            #no reason to make the visualization display 1000+ curves for large datasets
            if num_curves_in_cluster > ice_curves_to_export:
                individual_ice_samples = [list(x) for x in                                          list(all_curves_by_cluster[cluster_num]                                          [np.random.choice(num_curves_in_cluster,                                                 size=ice_curves_to_export, replace=False)])
                                         ]
            else:
                individual_ice_samples = [list(x) for x in                                          list(all_curves_by_cluster[cluster_num])                                         ]
            
            #add cluster-level metrics to the JSON file
            if export_type == "vis":
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
            else:
                export_dict["features"][column_of_interest]["clusters"].append({
                    'accuracy': int(round(100.*metrics.accuracy_score(rdwcY, predY))),
                    'precision': int(round(100.*metrics.precision_score(rdwcY, predY))),
                    'recall': int(round(100.*metrics.recall_score(rdwcY, predY))),
                    'split_feature': data.columns[model.tree_.feature[0]]\
                    if model.tree_.value.shape[0] > 1 else 'none',
                    'split_val': round(model.tree_.threshold[0],2),
                    'split_direction': "<=" if model.tree_.value.shape[0] == 1 or\
                    model.classes_[np.argmax(model.tree_.value[1])] == 1 else ">",                    
                    'predict_function': model.predict, 
                    'cluster_size': num_curves_in_cluster,
                    'line': list(cluster_average_curves[cluster_num])
                })                
        
        #feature-level calculation for cluster distance to pdp
        feature_val = export_dict["features"][column_of_interest]
        feature_val["cluster_deviation"]        = np.mean([_get_dtw_distance(np.array(feature_val["pdp_line"]),                                                  np.array(x["line"]))                  for x in feature_val["clusters"]])/np.mean(np.array(feature_val["pdp_line"]))
        if np.isnan(feature_val["cluster_deviation"]) or feature_val["cluster_deviation"]==float("inf"):
            feature_val["cluster_deviation"] = 0
        #EOF feature loop
        
    if export_type == "vis":
        with open('static/data.json', 'w') as outfile:
            json.dump(export_dict, outfile)        


# In[2201]:

get_ipython().run_cell_magic(u'time', u'', u'# REGULAR EXPORT\ndfX, y, predictor = load_dataset(name="boston")\nexport_dict = {"features":{}, "distributions":{}}\nexport(dfX, y, predictor.predict, num_clusters=4, num_grid_points=20,\\\n       ice_curves_to_export=20, export_type="analysis")')


# # COMPARATIVE ANALYSIS

# In[97]:

#altair
import altair as alt
alt.renderers.enable('notebook')

def add_keys_to_cluster(cluster, new_vals):
    z = cluster.copy()
    z.update(new_vals)
    return z


# In[776]:

feature_list = [l[0] for l in [[add_keys_to_cluster(c, {"feature_name":key, "cluster_num":i})  for i,c in enumerate([v for v in value["clusters"]])] for key, value in export_dict["features"].items()] for l[0] in l]

feature_df = pd.DataFrame(feature_list).loc[:,
                                            ["feature_name", "cluster_num", "accuracy", "cluster_size", 
                                             "precision", "recall", "confusion_matrix",\
                                            "split_feature", "split_val", "split_direction"]]
feature_df["id"] = feature_df.apply(lambda x: x["feature_name"]+": "                                                +x["split_feature"]+x["split_direction"]+str(x["split_val"]),axis=1)

#one split
print "mean accuracy: " + str(np.mean(feature_df["accuracy"]))
print "stdDev accuracy: " + str(np.std(feature_df["accuracy"]))
#print "min accuracy: " + str(np.min(feature_df["accuracy"]))
#print "max accuracy: " + str(np.max(feature_df["accuracy"]))
print "mean precision: " + str(np.mean(feature_df["precision"]))
print "stdDev precision: " + str(np.std(feature_df["precision"]))
#print "min precision: " + str(np.min(feature_df["precision"]))
#print "max precision: " + str(np.max(feature_df["precision"]))
print "mean recall: " + str(np.mean(feature_df["recall"]))
print "stdDev recall: " + str(np.std(feature_df["recall"]))
#print "min recall: " + str(np.min(feature_df["recall"]))
#print "max recall: " + str(np.max(feature_df["recall"]))


# # ALTAIR

# In[ ]:

alt.Chart(feature_df).mark_circle().encode( #size=300
    x='precision',
    y='accuracy',
    size='cluster_size',
    color=alt.Color('feature_name')
).properties(
    title="Bike Dataset Explanations: Precision vs. Accuracy",
    height=400, 
    width=700
)


# In[ ]:

alt.Chart(feature_df).mark_circle(size=100).encode(
    x='precision',
    y={
        "field": "cluster_size", 
        "type": "quantitative",
        "scale": {"type": "log"}
    },
    color=alt.Color('feature_name'),
).properties(
    title="Precision vs. Size",
    height=400, 
    width=700
)


# In[ ]:

alt.Chart(feature_df).mark_bar().encode(
    x='feature_name',
    y='mean(accuracy)'
).properties(
    title="Accuracy by Feature",
    height=400, 
    width=700
)


# In[ ]:

alt.Chart(feature_df).mark_circle(size=100).encode(
    x='feature_name',
    y={
        "field": "cluster_size", 
        "type": "quantitative",
        "scale": {"type": "log", "exponent":0.5}
    },
    color=alt.Color(field="id", type="nominal", legend=None, scale=alt.Scale(scheme = "blues-3")),
    order=alt.Order('cluster_size', sort='descending')
).properties(
    title="Cluster Sizes by Feature",
    height=400, 
    width=700
)


# # EVALUATION

# In[2197]:

get_ipython().run_cell_magic(u'time', u'', u'# REGULAR EXPORT\ndfX, y, predictor = load_dataset(name="diabetes")\nexport_dict= {"features":{}}\nexport(dfX, y, predictor.predict, num_clusters=7, num_grid_points=20,\\\n       ice_curves_to_export=2, export_type="analysis")')


# In[1797]:

def predict_pdp(evaluation_set, model_def, columns, offset, centered):
    
    def generate_interpolation_func(x, y):
        return interp1d(x,y, kind="linear")
    
    funcs = [generate_interpolation_func(model_def[feature]["x_values"], model_def[feature]["pdp_line"])                      for feature in columns]
    #from https://stackoverflow.com/questions/52167120/numpy-fastest-way-to-apply-array-of-functions-to-matrix-columns
    for i,f in enumerate(funcs):
        evaluation_set[:,i] = f(evaluation_set[:,i])
    
    #print columns
    #print evaluation_set.mean(axis=0)
    if centered:
        return evaluation_set.sum(axis=1) + offset
    else:
        return evaluation_set.mean(axis=1)


# In[2198]:

#offset = predictor.predict(np.array(dfX.min(axis=0)).reshape(1,-1))[0]
offset = y.mean()
eval_samples = np.array(dfX)
pdp_model_predictions = predict_pdp(eval_samples, export_dict["features"], dfX.columns, offset, centered)
print metrics.r2_score(y, pdp_model_predictions)
print metrics.mean_squared_error(y, pdp_model_predictions)**.5


# In[2199]:

#offset = predictor.predict(np.array(dfX.min(axis=0)).reshape(1,-1))[0]
offset = y.mean()
num_samples = dfX.shape[0]
eval_sample_ids = np.random.choice(a=dfX.shape[0], size=num_samples, replace=False)
eval_samples = np.array(dfX.iloc[eval_sample_ids,:])
iceskate_model_predictions = predict_iceskate(eval_samples, export_dict["features"], list(dfX.columns),                                              offset, centered)
print metrics.r2_score(y[eval_sample_ids], iceskate_model_predictions)
print metrics.mean_squared_error(y[eval_sample_ids], iceskate_model_predictions)**.5


# In[2195]:

def predict_iceskate(evaluation_set, model_def, columns, offset, centered):

    def generate_interpolation_func(x, y):
        if x == [0.,1.]:
            def categorical_func(arr, y=y):
                return np.where(arr>0,y[1],y[0])
            return categorical_func
        else:
            return interp1d(x,y, kind="linear")    
    
    def generate_predicate(cluster_def, x_vals, pdp_line, pdp_only=False):
        
        if pdp_only or cluster_def["split_feature"] == "none":
            pdp_interpol = generate_interpolation_func(x_vals, pdp_line)
            def pdp_func(x, pdp_interpol=pdp_interpol):
                return np.full(x.shape[0],pdp_interpol)
            return pdp_func
        
        interpol = generate_interpolation_func(x_vals, cluster_def["line"])
        
        def output_func(x, predict_func=cluster_def["predict_function"], interpol=interpol):
            return np.where(predict_func(x) == 1, interpol, 0) 
        
        return output_func
    
    funcs = [[generate_predicate(cluster_def, model_def[feature]["x_values"],                                          model_def[feature]["pdp_line"])                       for cluster_def in                       sorted(model_def[feature]["clusters"], reverse=True, key=lambda x: x["cluster_size"])]             for feature in columns]
    for i_func, possible_split_funcs in enumerate(funcs):
        feature = columns[i_func]
        possible_split_funcs.append(generate_predicate(None, model_def[feature]["x_values"],                                          model_def[feature]["pdp_line"], True))
    
    results = np.zeros(shape=(evaluation_set.shape[0], evaluation_set.shape[1]))
    
    #from https://stackoverflow.com/questions/52167120/numpy-fastest-way-to-apply-array-of-functions-to-matrix-columns
    for i_func,possible_split_funcs in enumerate(funcs):
        temp = np.zeros(shape=(evaluation_set.shape[0], len(possible_split_funcs)), dtype=object)
        for i_inner, func in enumerate(possible_split_funcs):
            #temp.append(func(evaluation_set))
            temp[:,i_inner] = func(evaluation_set)
        
        rows_handled = []
        func_outputs = []
        #from https://stackoverflow.com/questions/11731428/finding-first-non-zero-value-along-axis-of-a-sorted-two-dimensional-numpy-array
        for row, valid_func_id in zip(*np.where(temp != 0)):
            
            valid_func = temp[row, valid_func_id]
            
            if row in rows_handled:
                func_outputs[row].append(valid_func(evaluation_set[row,i_func])[()])
            else:
                rows_handled.append(row)
                func_outputs.append([valid_func(evaluation_set[row,i_func])[()]])
                
        results[:,i_func] = np.array([(sum(x[:-1] if len(x)>1 else x))/max(len(x[:-1]),1) for x in func_outputs])
        #results[:,i_func] = np.array([sum(x[:])/len(x[:]) for x in func_outputs])
        #results[:,i_func] = np.array([x[0] for x in func_outputs])
    
    #print columns
    #print results.mean(axis=0)
    if centered:
        return results.sum(axis=1) + offset
    else:
        return results.mean(axis=1)


# In[ ]:



