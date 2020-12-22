from wide_field_api import get_grid_size_for_mice, is_resting_state_stimuli

import numpy as np
import scipy.io as sio

from joblib import Parallel, delayed
from tqdm import tqdm
import os
import enum

import matlab.engine
eng = matlab.engine.start_matlab()

import matlab.engine

eng.addpath ('./matlab_src', nargout= 0 )
eng.addpath ('./matlab_src/classifiers', nargout= 0 )
eng.addpath ('./matlab_src/bayes', nargout= 0 )
eng.addpath ('./matlab_src/plots', nargout= 0 )



class classifiers(enum.Enum):
   GMM = 'GMM_Classifier'
   Unimodel_Bayes = 'Unimodel_Bayes_Classifier'
   SVM = 'SVM_Classifier'
   ANN = 'ANN_Classifier'

def get_classifier_index(classifier):
	classifier_index_dict = {
							classifiers.GMM:1,
							classifiers.Unimodel_Bayes: 2,
							classifiers.SVM:3,
							classifiers.ANN:4
							}
	return classifier_index_dict[classifier]

def plot_supervised_classification_result(mouse, stimuli, classifier):
	result_location = 'results/'+mouse.value+'/'+stimuli.value+'/'
	result_location += "/"+classifier.value+"/"
	processed_data_location = "dataset/"+mouse.value+"/"+stimuli.value;
	if not os.path.isfile(result_location+"/results.mat"):
		print("Classifying results not found. Running classifier", flush=True);
		run_supervised_classification(mouse,stimuli,classifier)
	figure_location = eng.plot_supervised_classification_result(processed_data_location,result_location,nargout=1)
	return figure_location

def plot_semi_supervised_classification_result(mouse, stimuli, ):
	result_location = 'results/'+mouse.value+'/'+stimuli.value+'/'
	result_location += "/semi-supervised/"
	processed_data_location = "dataset/"+mouse.value+"/"+stimuli.value;
	if not os.path.isfile(result_location+"/results.mat"):
		print("Classifying results not found. Running classifier", flush=True);
		run_semi_supervised_classification(mouse,stimuli)
	figure_location = eng.plot_semi_supervised_classification_result(processed_data_location,result_location,nargout=1)
	return figure_location

def run_supervised_classification(mouse, stimuli, classifier):
	processed_data_location = "dataset/"+mouse.value+"/"+stimuli.value;
	if not os.path.isfile(processed_data_location+"/data.mat") or not os.path.isfile(processed_data_location+"/map.mat"):
		print("Downloaded data not found please run Step 1")
		return
	eng.addpath ('./', nargout= 0 )
	[acc, cm, names,
	train_prediction, test_predictions,
	train_maps, test_maps] = eng.run_supervised_classification(processed_data_location,
		get_classifier_index(classifier),nargout=7)
	results={}
	results['acc']=acc
	results['cm']=cm
	results['names']=names
	results['train_prediction']=train_prediction
	results['test_prediction']=test_predictions
	results['train_map']=train_maps
	results['test_map']=test_maps
	
	result_location = 'results/'+mouse.value+'/'+stimuli.value+'/'
	result_location += "/"+classifier.value+"/"
	
	if not os.path.exists(result_location):
		os.makedirs(result_location)
	sio.savemat(result_location+"/results.mat", results);

	return acc;

def run_subset_supervised_classification(mouse, stimuli, classifier, duration, is_single_trial=False):
	data_location = "dataset/"+mouse.value+"/"+stimuli.value;
	if not os.path.isfile(data_location+"/data.mat") or not os.path.isfile(data_location+"/map.mat"):
		print("Dataset not found.", flush=True)
		return
	if is_single_trial and  not os.path.isfile(data_location+"/data_single_trial.mat"):
		print("Single Trial data not present for given mice stimuli combo", flush=True)
		return
	[acc, cm, names,
	train_prediction, test_predictions,
	train_maps, test_maps] = eng.run_subset_supervised_classification(data_location,
		get_classifier_index(classifier), duration, is_single_trial, is_resting_state_stimuli(stimuli) ,nargout=7)
	results={}
	results['acc']=acc
	results['cm']=cm
	results['names']=names
	results['train_prediction']=train_prediction
	results['test_prediction']=test_predictions
	results['train_map']=train_maps
	results['test_map']=test_maps
	
	result_location = 'results/'+mouse.value+'/'+stimuli.value+'/'
	result_location += "/"+classifier.value+"/"+str(duration)+"/"
	
	if not os.path.exists(result_location):
		os.makedirs(result_location)
	sio.savemat(result_location+"/results.mat", results);

	return acc



def run_semi_supervised_classification(mouse, stimuli):
	processed_data_location = "dataset/"+mouse.value+"/"+stimuli.value;
	if not os.path.isfile(processed_data_location+"/data.mat") or not os.path.isfile(processed_data_location+"/map.mat"):
		print("Dataset not found.", flush=True)
		raise
	eng.addpath ('./', nargout= 0 )
	
	[acc, predicted_labels, cm, names] = eng.run_semi_supervised_classification(processed_data_location,
		get_grid_size_for_mice(mouse), nargout=4)
	results={}
	results['acc']=acc
	results['cm']=cm
	results['names']=names
	results['predicted_labels']=predicted_labels
	result_location = 'results/'+mouse.value+'/'+stimuli.value+'/'
	result_location += "/semi-supervised/"
	
	if not os.path.exists(result_location):
		os.makedirs(result_location)
	sio.savemat(result_location+"/results.mat", results);
	return acc
	

def run_subset_semi_supervised_classification(mouse, stimuli, classifier, duration, is_single_trial=False):
	data_location = "dataset/"+mouse.value+"/"+stimuli.value;
	if not os.path.isfile(data_location+"/data.mat") or not os.path.isfile(data_location+"/map.mat"):
		print("Dataset not found.", flush=True)
		raise
	if is_single_trial and  not os.path.isfile(data_location+"/data_single_trial.mat"):
		print("Single Trial data not present for given mice stimuli combo", flush=True)
		return
	[testAccuracy, predicted_labels, cm, names] = eng.run_subset_semi_supervised_classification(data_location,
		get_grid_size_for_mice(mouse), duration, is_single_trial, is_resting_state_stimuli(stimuli) ,nargout=4)
	return testAccuracy
