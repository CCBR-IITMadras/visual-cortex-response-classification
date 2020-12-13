function [accuracy, confusion_mats, area_names] =run_subset_supervised_classification(data_location, classifier, duration, is_single_trial, frame_rate)


	if is_single_trial == true
		data = load(strcat(data_location,"/data_single_trial.mat"));
	else
		data = load(strcat(data_location,"/data.mat"));
	end

	

	areas = fieldnames(data);
	for k=1:numel(areas)
		
		area_response = data.(areas{k});
 		data.(areas{k}) = area_response(:,1:ceil(duration*frame_rate))

	end
	
	[accuracy,...
	 confusion_mats, area_names] = supervised_classification(data, classifier)
	
end