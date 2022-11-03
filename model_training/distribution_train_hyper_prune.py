'''declare global imports'''
import sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import *
from sklearn.metrics import *
import csv
import sys
import time
from joblib import dump, load
import joblib 
from joblib import Parallel, delayed
import json
import joblib
import argparse
import os
from collections import Counter
import math
from sklearn.preprocessing import QuantileTransformer
from scipy import stats
from sklearn.metrics import *

max_value = 999999
to_prune = False
num_to_prune = 1
rand_num = 42
alpha = 6

def load_csv(filename, label_column):
    """
    Loads a csv file containin the data, parses it
    and returns numpy arrays the containing the training
    and testing data along with their labels.

    :param filename: the filename
    :return: tuple containing train, test data np arrays and labels
    """
    if 'iris' in filename:
        iris = load_iris()
        return iris.data, iris.target

    X_train = []
    y_train = []
    num = 0
    with open(filename,'rt') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            row1 = [float(item)  for item in row if item != '\0']
            if label_column == 0:
                last_ele = row1.pop(0)
            else:
                last_ele = row1.pop(-1)
            X_train.append(row1)
            #print(min(row1), flush=True)
            y_train.append(int(last_ele))
            num+=1

    f.close()
    return np.array(X_train), np.array(y_train)


def readHyperList(hyper_filename):
    hyperlist = []
    num = 0
    with open(hyper_filename,'rt') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            row1 = [float(item)  for item in row if item != '\0']
            hyperlist.append(row1)
            num+=1

    return hyperlist

def readBboxList(hyper_filename):
    hyperlist = []
    num = 0
    with open(hyper_filename,'rt') as f:
        reader = csv.reader(f, delimiter='\n')
        for row in reader:
            row1 = [float(item)  for item in row if item != '\0']
            hyperlist.extend(row1)
            num+=1

    return hyperlist


def trainModel(X, y, n_trees=100, from_pkl=False, pkl_filename='None', save_rf_model=True, save_rf_filename='None'):
    '''
    Train a tree ensemble model OR read from pickle file. Return the sklearn estimator
    '''
    if from_pkl:
        clf = load(pkl_filename)
        return clf


    clf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, max_features=None)
    #clf = BaggingClassifier(n_estimators=n_trees, n_jobs=-1, bootstrap=False, max_samples=0.67)
    clf.fit(X, y)
    if save_rf_model:
        joblib.dump(clf, save_rf_filename)
    return clf



def getSamplesInLeaves(X_train, y_train, model, num_estimators, importances, X_transformed):
    bbox_min = np.amin(X_train, axis=0)
    bbox_max = np.amax(X_train, axis=0)
    siz = bbox_min.shape[0]
    bbox = []
    actual_bbox  = []
    for i in range(siz):
        actual_bbox.append(99999)
        actual_bbox.append(-99999)
        bbox.append(-99999)
        bbox.append(99999)

    leaf_kde_map_arr = []

    k = X_train.shape[1]

    index_list = model.getIndicesList()
    main_indices = set([i for i in range(X_train.shape[0])])
    to_save_list = []
    hyperplane_list = []
    class_list = []
    card_list = []
    mean_list = []
    var_list = []
    max_list = []
    min_list = []
    median_list = []
    mad_list = []
    mean_t_list = []
    numerator_pooled = np.zeros(k)
    denominator_pooled= 0
    for tree_num in range(num_estimators):
        start_time = time.time() 
        indices2 = index_list[tree_num]
        indices = list(set(list(indices2)))
        indices.sort()
        clf = model.estimators_[tree_num]        
        leaf_class_map = {}
        leaf_obs_map = {}
        leaf_kde_map = {}
        leaves_single_tree = model.estimators_[tree_num].apply(X_train[indices])
        predicted_X_train = model.estimators_[tree_num].predict(X_train[indices])

        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        leaf_obsnum_map = {}
        for iter1, leaf_id in enumerate(leaves_single_tree):
            obs_num = indices[iter1]
            if leaf_id not in leaf_obs_map:
                leaf_obs_map[leaf_id] = []
                leaf_obsnum_map[leaf_id] = []

            leaf_obs_map[leaf_id].append(X_train[obs_num, :])
            leaf_class_map[leaf_id] = predicted_X_train[iter1]
            leaf_obsnum_map[leaf_id].append(obs_num)

        importance_list = []
        for leaf, obs in leaf_obs_map.items():
            sample_id = 0
            curr_imp = 0.0
            x = [obs[0]]
            node_indicator = clf.decision_path(np.array(x))
            node_index = node_indicator.indices[node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]]
            n_nodes=0
            leaf_ids =  clf.apply(x)
            for node_id in node_index:
                if leaf_ids[sample_id] == node_id:
                    continue
                curr_imp += importances[feature[node_id]]
                n_nodes+=1
            importance_list.append( (leaf, curr_imp / float(n_nodes), len(obs) ) )
        importance_list.sort(key = lambda x : (-x[2], -x[1])) 
        

        
        leaf_ids_remaining = [x[0] for x in importance_list[0:65000]]
        leaf_ids_remaining.sort()

        hyper_map = {}
        hyperclass_map = {}

        #for leaf, obs in leaf_obs_map.items():
        for leaf in leaf_ids_remaining:
            obs = leaf_obs_map[leaf]
            num_obs = len(obs)
            if num_obs < 0:
                continue
            transformed_obs_list = []
            for idl in  leaf_obsnum_map[leaf]:
                transformed_obs_list.append(X_transformed[idl])

            transformed_obs_arr = np.array(transformed_obs_list)

            mean_transformed = np.mean(transformed_obs_arr, axis=0)

            obs_array = np.array(obs)
            mean = np.mean(obs_array, axis=0) 
            var = np.var(obs_array, axis=0)
            mad = stats.median_absolute_deviation(obs_array, axis=0) 
            median = np.median(obs_array, axis=0)
            numerator_pooled = np.add(numerator_pooled,  float(num_obs-1)*var)
            denominator_pooled += float(num_obs-1)
            max_obs = np.amax(np.array(leaf_obs_map[leaf]), axis=0)
            min_obs = np.amin(np.array(leaf_obs_map[leaf]), axis=0)
            max_list.append(max_obs)
            min_list.append(min_obs)
            it=0
            flag = 0
            if num_obs >= 0:

                hyperplane  = [i for i in bbox]
                x = [obs[0]]
                x = np.array(x)
                node_indicator = clf.decision_path(x)
                leaf_id = clf.apply(x)
                sample_id = 0
                node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]
                for node_id in node_index:
                # continue to the next node if it is a leaf node
                    if leaf_id[sample_id] == node_id:
                        continue

                # check if value of the split feature for sample 0 is below threshold
                    if (x[sample_id, feature[node_id]] <= threshold[node_id]):
                        threshold_sign = "<="
                        hyperplane[2*feature[node_id] + 1] = threshold[node_id]
                        if hyperplane[2*feature[node_id] + 1] > actual_bbox[2*feature[node_id] + 1]:
                            actual_bbox[2*feature[node_id] + 1] = hyperplane[2*feature[node_id] + 1] 
                    elif(x[sample_id, feature[node_id]] > threshold[node_id]):
                        threshold_sign = ">"
                        hyperplane[2*feature[node_id]] = threshold[node_id]
                        if hyperplane[2*feature[node_id]] < actual_bbox[2*feature[node_id]]:
                            actual_bbox[2*feature[node_id]] = hyperplane[2*feature[node_id]] 

                hyperplane_list.append(np.array(hyperplane))
                mean_list.append(mean)
                var_list.append(var)
                class_list.append(clf.predict(x)[0])
                card_list.append(num_obs)
                median_list.append(median)
                mad_list.append(mad)
                mean_t_list.append(mean_transformed)
    return  actual_bbox, np.array(hyperplane_list), np.array(class_list), np.array(card_list), np.array(mean_list), np.array(var_list), numerator_pooled / float(denominator_pooled), np.array(max_obs), np.array(min_obs), np.array(median_list), np.array(mad_list), np.array(mean_t_list)

def inverse(vec):
    inv_vec = np.where(vec > 0, 1 / vec, vec)
    return inv_vec

def chunks(l, n):
    n = max(1, n)
    lenl = len(l)

    return [ (l[i], l[min(i+n, lenl-1)]) for i in range(0, lenl, n)]


def saveClusterMap(leaf_cluster_map_arr, save_filename='default.joblib', json_filename='default.json', save_pkl=False, save_json=True):
    '''
    Saves the array of obs maps, class maps and cluster maps to disk
    '''
    
    if save_pkl:
        joblib.dump(leaf_cluster_map_arr, save_filename)
    if save_json:
        leaf_json_map_arr = []
        for tree_num, tree_map in enumerate(leaf_cluster_map_arr):
            for leaf_id, leaf_params in tree_map.items():
                new_map = {}
                new_map['treenum'] = int(tree_num)
                new_map['leafid'] = int(leaf_id)
                new_map['mean'] = leaf_params[0]
                new_map['var'] = leaf_params[1]
                new_map['term1'] = float(leaf_params[2])
                new_map['numobs'] = float(leaf_params[3])
                new_map['class'] = int(leaf_params[4])
                leaf_json_map_arr.append(new_map)
        json_obj = json.dumps(leaf_json_map_arr)
        with open(json_filename, "w") as outfile:
            outfile.write(json_obj)

def intersects(lst, interval, dim):
    low = lst[2*dim]
    high = lst[2*dim + 1]
    if high < interval[0] or low > interval[1]:
        return False
    return True

def point_intersects(lst, point, dim):
    low = lst[2*dim]
    high = lst[2*dim + 1]

    if low > point or high < point:
        return False
    return True

def parseCmdArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--labelcol', action='store', dest='label_column',
                    help='Label column', type=int)

    parser.add_argument('--datadir', action='store', dest='file_dir',
                    help='Dataset location directory')

    parser.add_argument('--datafilename', action='store', dest='data_filename',
                    help='Dataset name')

    parser.add_argument('--numtrees', action='store', dest='num_trees',
                    help='Number of trees')
    
    parser.add_argument('--num_hyperplanes_post_prune', action='store', dest='num_hyperplanes_post_prune',
                    help='Number of trees', default=65000)

    parser.add_argument('--numtest', action='store', dest='num_test', nargs='?', 
                    const=100, type=int, help='Number of test samples')

    parser.add_argument('--modelfromfile', action='store', dest='from_file', nargs='?', 
                    const=False, help='Read rf model from pickle file on disk?')

    parser.add_argument('--modelfilename', action='store', dest='rf_pkl_filename', nargs='?', 
                    const='None.pkl', help='Name of rf model pickle file')

    results = parser.parse_args()

    return results



results = parseCmdArgs()

label_column = results.label_column
file_dir = results.file_dir
num_trees = results.num_trees
data_filename = results.data_filename
num_test = results.num_test
from_file = results.from_file
rf_pkl_filename = results.rf_pkl_filename

data_string = data_filename.split('.')[0]
data_path_filename = os.path.join(file_dir, data_filename)
save_poly_filename = os.path.join(file_dir, 'poly_' + data_string + '.json' )
save_train_data = os.path.join(file_dir ,'train_' + data_filename)
save_test_data = os.path.join(file_dir, 'test_' + data_filename)
rf_model_filename = os.path.join(file_dir, 'rf_'+ num_trees + data_string + '.joblib')
num_trees = int(num_trees)
print('args parsed', flush=True)

X, y =  load_csv(data_path_filename, 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=num_test, stratify=y, random_state=rand_num)

concat_arr_train = np.c_[X_train, y_train]
concat_arr_test = np.c_[X_test, y_test]


from_file=False
if from_file:
    clf = trainModel(X_train, y_train, n_trees=num_trees, from_pkl=True, pkl_filename=rf_pkl_filename, save_rf_model=False)
else:
    clf = trainModel(X_train, y_train, n_trees=num_trees, save_rf_model=True, save_rf_filename=rf_model_filename)
print('model trained', flush=True)

import sys

y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred)
print('RF accuracy: ', end='', flush=True)
print(score, flush=True)


num_features = 4
importances = clf.feature_importances_

n = importances.shape[0]

features = np.argsort(-importances)[0:num_features]

features = features.tolist()
features2 = np.argsort(-importances)


importances = importances.tolist()
importances.sort(reverse=True)
print("importances: ")
print (importances)


qt = QuantileTransformer(n_quantiles=100000)
X_transformed = qt.fit_transform(X_train)


bbox, hyperlist, class_list, card_list, mean_list, var_list, pooled, max_list, min_list, median_list, mad_list, mean_t_list = getSamplesInLeaves(X_train, y_train,  clf, num_trees, importances, X_transformed)
np.savetxt(data_path_filename+"hyperrectangles_foo_maxdepth.csv", hyperlist,  delimiter = ",", fmt='%1.3f')
np.savetxt(data_path_filename+"mean_list.csv", mean_list,  delimiter = ",", fmt='%1.9f')
np.savetxt(data_path_filename+"mean_t_list.csv", mean_t_list,  delimiter = ",", fmt='%1.9f')
np.savetxt(data_path_filename+"pooled.csv", pooled,  delimiter = ",", fmt='%1.9f')
np.savetxt(data_path_filename+"class_list.csv", class_list,  delimiter = ",", fmt='%1.3f')
np.savetxt(data_path_filename+"card_list.csv", card_list,  delimiter = ",", fmt='%1.3f')
np.savetxt(data_path_filename+"features.csv",features2,  delimiter = ",", fmt='%1.3f')
