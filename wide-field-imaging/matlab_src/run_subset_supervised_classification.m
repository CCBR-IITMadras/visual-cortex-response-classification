function [accuracy, cm, area_names, ...
		 train_prediction, test_prediction, ...
		 train_map, test_map] = run_subset_supervised_classification(data_location, classifier, duration, is_single_trial, is_resting_state)


        

	% loading data
	if is_single_trial == true
		data = load(strcat(data_location,"/data_single_trial.mat"));
		load(strcat(data_location,"/meta_info_single_trial.mat"),"frame_rate")
	else
		data = load(strcat(data_location,"/data.mat"));
		load(strcat(data_location,"/meta_info.mat"),"frame_rate")
	end
	
	
	map = load(strcat(data_location,"/map.mat"));

	

	areas = fieldnames(data);
	for k=1:numel(areas)
		if(is_resting_state)
			area_response = data.(areas{k});
			start_frame = ceil(100*frame_rate);
			end_frame = start_frame + ceil(duration * frame_rate);
	 		data.(areas{k}) = area_response(:,start_frame:end_frame)
		else
    		area_response = data.(areas{k});
	 		data.(areas{k}) = area_response(:,1:ceil(duration*frame_rate))
	 	end
       
	end
	
    [accuracy, cm, area_names, ...
		 train_prediction, test_prediction, ...
		 train_map, test_map] = supervised_classification(data, map, classifier)
	
	
end