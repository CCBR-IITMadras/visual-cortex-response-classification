import os


import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath ('./matlab_src/plots', nargout= 0 )
eng.addpath ('./matlab_src', nargout= 0 )


def plot_reponse_correlation(mouse, stimuli):
	processed_data_location = "dataset/"+mouse.value+"/"+stimuli.value+"/data.mat";
	result_location = 'results/'+mouse.value+'/'+stimuli.value
	if not os.path.isfile(processed_data_location):
		print("Downloaded data not found please run Step 1")
		raise
	if not os.path.exists(result_location):
		os.makedirs(result_location)

	file_name = eng.plot_reponse_correlation(processed_data_location, result_location, nargout=1);
	return file_name

def plot_tSNE(mouse, stimuli):
	processed_data_location = "dataset/"+mouse.value+"/"+stimuli.value+"/data.mat";
	result_location = 'results/'+mouse.value+'/'+stimuli.value
	if not os.path.isfile(processed_data_location):
		print("Downloaded data not found please run Step 1")
		raise
	if not os.path.exists(result_location):
		os.makedirs(result_location)

	file_name = eng.plot_tSNE(processed_data_location, result_location, nargout=1);
	return file_name