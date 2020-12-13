from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info
from allensdk.brain_observatory.natural_movie import NaturalMovie, StimulusAnalysis

import numpy as np
import scipy.io as sio

from joblib import Parallel, delayed
from tqdm import tqdm
import os


def download_experiment_data(experiment_id, boc):
	#print("Downloading: " + str(experiment_id))
	try:
		boc.get_ophys_experiment_data(experiment_id)
	except:
		if os.path.isfile('dataset/ophys_experiment_data/'+str(experiment_id)+'.nwb'):
			os.remove('dataset/ophys_experiment_data/'+str(experiment_id)+'.nwb')
		os.system('wget -O dataset/ophys_experiment_data/'+str(experiment_id)+'.nwb http://api.brain-map.org/api/v2/well_known_file_download/'+str(experiment_id))	

def get_session(stimuli):
	stimuli_session_dict = {stim_info.NATURAL_MOVIE_ONE:'three_session_A',
							stim_info.NATURAL_MOVIE_TWO:'three_session_C2',
							stim_info.NATURAL_MOVIE_THREE:'three_session_A',
							stim_info.SPONTANEOUS_ACTIVITY:'three_session_C2'
							}
	return stimuli_session_dict[stimuli]

def get_frame_rate():
	return 30

def get_experiment_ids_and_boc(cre_line,stimuli):
	# This class uses a 'manifest' to keep track of downloaded data and metadata.
	# All downloaded files will be stored relative to the directory holding the manifest
	# file.  If 'manifest_file' is a relative path (as it is below), it will be
	# saved relative to your working directory.  It can also be an absolute path.
	boc = BrainObservatoryCache(manifest_file='dataset/manifest.json')

	stimuli_experiments = boc.get_ophys_experiments(cre_lines=[cre_line],
							stimuli=[stimuli],session_types=[get_session(stimuli)]);

	return stimuli_experiments,boc

	
def download_allen_insitute_data(cre_line, stimuli, num_jobs=10):
	[stimuli_experiments, boc] = get_experiment_ids_and_boc(cre_line, stimuli)
	experiment_ids =  []
	experiment_detials = []
	for experiment in stimuli_experiments:
		experiment_ids.append(experiment['id'])
		experiment_detials.append(experiment)
	print('Downloading data for Cre Line: '+cre_line+" and stimuli: "+stimuli,  flush=True)
	results = Parallel(n_jobs=num_jobs)(delayed(download_experiment_data)(experiment_id, boc) for experiment_id in tqdm(experiment_ids))

def prepare_allen_insitute_data(cre_line, stimuli, num_jobs=10):
	[stimuli_experiments, boc] = get_experiment_ids_and_boc(cre_line, stimuli)
	experiment_ids =  []
	experiment_detials = []
	for experiment in stimuli_experiments:
		experiment_ids.append(experiment['id'])
		experiment_detials.append(experiment)

	cells_count = {}
	exps = {}	
	for ind, experiment_id in enumerate(experiment_ids):
		exp = boc.get_ophys_experiment_data(experiment_id)
		area = experiment_detials[ind]['targeted_structure']
		if area not in cells_count:
			exps[area] = [exp]
			cells_count[area] = len(exp.get_cell_specimen_ids())
		else:
			exps[area].append(exp)
			cells_count[area] = cells_count[area] + len(exp.get_cell_specimen_ids())
	
	cell_wise_data = {}
	cell_wise_single_data ={}
	cell_wise_label = {}
	cell_ids_collected = [];
	key_dict={"VISp":"V1", "VISl":"LM", "VISam":"AM", "VISal":"AL", "VISpm": "PM", "VISrl": "RL"}
	min_samples = float("inf")
	for key in cells_count.keys():
		for exp_data in exps[key]:
			if stimuli != stim_info.SPONTANEOUS_ACTIVITY:
				ng = NaturalMovie(exp_data, stimuli)
				no_of_cells = len(exp_data.get_cell_specimen_ids())
				cell_ids = exp_data.get_cell_specimen_ids()
				for cell, cell_id in enumerate(cell_ids):
					if cell_id in cell_ids_collected:
						continue;
					else:
						cell_ids_collected.append(cell_id)
					mean_response=[]
					subset = ng.sweep_response
					cell_subset = np.asarray(subset[str(cell)].tolist())
					mean_response = np.mean(cell_subset,axis=0)
					min_samples=min(min_samples,len(mean_response))
			elif stimuli == stim_info.SPONTANEOUS_ACTIVITY:
				sa = StimulusAnalysis(exp_data)
				stim_table = exp_data.get_stimulus_table(stimuli);
				no_of_cells = len(exp_data.get_cell_specimen_ids())
				start = stim_table.iloc[0].start + 30 * 30  # ignore after initial 30 seconds
				end = stim_table.iloc[0].end - 30 * 30 # ignore final 30 seconds
				for cell in range(0, no_of_cells):
					mean_response = sa.dfftraces[cell, start:end]
					min_samples=min(min_samples,len(mean_response))
			else:
				print('stimuli not configured')
				return

	key_label = 0
	cell_ids_collected = [];
	for key in cells_count.keys():
		for exp_data in exps[key]:
			if stimuli != stim_info.SPONTANEOUS_ACTIVITY:
				ng = NaturalMovie(exp_data, stimuli)
				no_of_cells = len(exp_data.get_cell_specimen_ids())
				cell_ids = exp_data.get_cell_specimen_ids()
				for cell, cell_id in enumerate(cell_ids):
					if cell_id in cell_ids_collected:
						# if the same neuron is collected in multiple experiments
						# we should not consider it as different data points.
						continue;
					else:
						cell_ids_collected.append(cell_id)
					mean_response=[]
					sigle_response=[]
					subset = ng.sweep_response
					cell_subset = np.asarray(subset[str(cell)].tolist())
					mean_response = np.mean(cell_subset,axis=0)
					single_response = cell_subset[0,:]
					if key_dict[key] not in cell_wise_data:
						cell_wise_data[key_dict[key]]=[mean_response[:min_samples]]
						cell_wise_single_data[key_dict[key]] = [single_response[:min_samples]]
						cell_wise_label[key_dict[key]] = [key_label]
					else:
						cell_wise_data[key_dict[key]].append(mean_response[:min_samples])
						cell_wise_single_data[key_dict[key]].append(single_response[:min_samples])
						cell_wise_label[key_dict[key]].append(key_label)
			elif stimuli == stim_info.SPONTANEOUS_ACTIVITY:
				sa = StimulusAnalysis(exp_data)
				stim_table = exp_data.get_stimulus_table(stimuli);
				no_of_cells = len(exp_data.get_cell_specimen_ids())
				start = stim_table.iloc[0].start + 30 * 30  # ignore after initial 30 seconds
				end = stim_table.iloc[0].end - 30 * 30 # ignore final 30 seconds
				for cell in range(0, no_of_cells):
					mean_response = sa.dfftraces[cell, start:end]
					if key_dict[key] not in cell_wise_data:
						cell_wise_data[key_dict[key]]=[mean_response[:min_samples]]
						cell_wise_label[key_dict[key]] = [key_label]
					else:
						cell_wise_data[key_dict[key]].append(mean_response[:min_samples])
						cell_wise_label[key_dict[key]].append(key_label)
			
		cell_wise_data[key_dict[key]] = np.asarray(cell_wise_data[key_dict[key]])
		cell_wise_label[key_dict[key]] = np.asarray(cell_wise_label[key_dict[key]])
		if stimuli != stim_info.SPONTANEOUS_ACTIVITY:
			cell_wise_single_data[key_dict[key]] = np.asarray(cell_wise_single_data[key_dict[key]])
		key_label = key_label+1
		
	data = cell_wise_data
	if not os.path.exists("dataset/processed/"+cre_line+"/"+stimuli+"/"+get_session(stimuli)+"/"):
		os.makedirs("dataset/processed/"+cre_line+"/"+stimuli+"/"+get_session(stimuli)+"/")
	sio.savemat("dataset/processed/"+cre_line+"/"+stimuli+"/"+get_session(stimuli)+"/data.mat",data);
	if stimuli != stim_info.SPONTANEOUS_ACTIVITY:
		data_single_trial = cell_wise_single_data
		sio.savemat("dataset/processed/"+cre_line+"/"+stimuli+"/"+get_session(stimuli)+"/data_single_trial.mat",data_single_trial);
	print("saved data for Cre Line: "+cre_line+" and stimuli: "+stimuli+
		" in dataset/processed/"+cre_line+"/"+stimuli+"/"+get_session(stimuli)+"/data.mat")