import numpy as np
import scipy.io as sio

from joblib import Parallel, delayed
from tqdm import tqdm
import os
import enum

from allen_api import get_session, get_frame_rate

import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath ('./matlab_src', nargout= 0 )
eng.addpath ('./matlab_src/classifiers', nargout= 0 )
eng.addpath ('./matlab_src/LDA', nargout= 0 )
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

def run_supervised_classification(cre_line, stimuli, classifier):
	processed_data_location = "dataset/processed/"+cre_line+"/"+stimuli+"/"+get_session(stimuli)+"/data.mat";
	if not os.path.isfile(processed_data_location):
		print("Downloaded data not found please run Step 1")
		raise
	eng.addpath ('./', nargout= 0 )
	[acc,cm,names] = eng.run_supervised_classification(processed_data_location,
		get_classifier_index(classifier),nargout=3)
	results={}
	results['acc']=acc
	results['cm']=cm
	results['names']=names
	result_location = 'results/'+cre_line+'/'+stimuli+'/'+get_session(stimuli)
	result_location += "/"+classifier.value+"/"
	
	if not os.path.exists(result_location):
		os.makedirs(result_location)
	sio.savemat(result_location+"/results.mat", results);
	return acc

def run_subset_supervised_classification(cre_line, stimuli, classifier, duration, is_single_trial=False):
	processed_data_location = "dataset/processed/"+cre_line+"/"+stimuli+"/"+get_session(stimuli);
	if not os.path.isfile(processed_data_location+"/data.mat"):
		print("Downloaded data not found please run Step 1")
		raise
	if is_single_trial and not os.path.isfile(processed_data_location+"/data_single_trial.mat"):
		print("Single Trial data not present for given mice stimuli combo", flush=True)
		raise
	[acc,cm,names] = eng.run_subset_supervised_classification(processed_data_location,
		get_classifier_index(classifier), duration, is_single_trial, get_frame_rate() ,nargout=3)
	results={}
	results['acc']=acc
	results['cm']=cm
	results['names']=names
	result_location = 'results/'+cre_line+'/'+stimuli+'/'+get_session(stimuli)
	result_location += "/"+classifier.value+"/"
	
	if not os.path.exists(result_location):
		os.makedirs(result_location)
	sio.savemat(result_location+"/results.mat", results);
	return acc

def plot_confusion_mat(cre_line, stimuli, classifier):
	result_location = 'results/'+cre_line+'/'+stimuli+'/'+get_session(stimuli)
	result_location += "/"+classifier.value+"/"
	if not os.path.isfile(result_location+"/results.mat"):
		print("Classifying results not found. Running classifier", flush=True);
		run_supervised_classification(cre_line,stimuli,classifier)
	figure_location = eng.plot_confusion_mat(result_location,nargout=1)
	return figure_location
