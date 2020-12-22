import enum
import scipy.io as sio
import numpy as np
import os


class mice(enum.Enum):
   M1 = 'M1'
   M2 = 'M2'
   M3 = 'M3'
   M4 = 'M4'
   M5 = 'M5'

class stimuli(enum.Enum):
   Direction = 'Direction'
   Natural_Movies = 'Natural_Movies'
   Spatial_Frequency = 'Spatial_Frequency'
   Temporal_Frequency = 'Temporal_Frequency'
   Resting_State = 'Resting_State'


def get_stimuli_for_mice(mouse):
	stimuli_mice={}
	stimuli_mice[mice.M1] = [stimuli.Direction, stimuli.Natural_Movies,
	 			stimuli.Spatial_Frequency, stimuli.Temporal_Frequency]
	stimuli_mice[mice.M2] = [stimuli.Natural_Movies] 	
	stimuli_mice[mice.M3] = [stimuli.Natural_Movies] 
	stimuli_mice[mice.M4] = [stimuli.Natural_Movies, stimuli.Resting_State] 
	stimuli_mice[mice.M5] = [stimuli.Natural_Movies, stimuli.Resting_State] 		
	return stimuli_mice[mouse]

def get_grid_size_for_mice(mouse):
	mice_grid_size={}
	mice_grid_size[mice.M1] = 5
	mice_grid_size[mice.M2] = 6 	
	mice_grid_size[mice.M3] = 7 
	mice_grid_size[mice.M4] = 6 
	mice_grid_size[mice.M5] = 7 		
	return mice_grid_size[mouse]

def is_resting_state_stimuli(stim):
	return stim == stimuli.Resting_State

def check_dataset(mouse, stimuli):
	data_location = "dataset/"+mouse.value+"/"+stimuli.value;
	if not os.path.isfile(data_location+"/meta_info.mat"):
		print("Dataset not found.", flush=True)
		return False
	return True	

def get_meta_info(mouse, stimuli):
	data_location = "dataset/"+mouse.value+"/"+stimuli.value;
	if not os.path.isfile(data_location+"/meta_info.mat"):
		print("Dataset not found.", flush=True)
		return None, None
	meta_data=sio.loadmat(data_location+"/meta_info.mat")
	fram_rate = meta_data['frame_rate'][0,0]
	stimuli_info = meta_data['stimuli_index'][0].tolist()
	stimuli_set = set(stimuli_info)
	stimuli_set = list(stimuli_set)
	stimuli_length = []
	for stimuli in stimuli_set:
		stimuli_length.append(round(stimuli_info.count(stimuli)/fram_rate,2))
	return stimuli_set, stimuli_length



