function [accuracy, confusion_mats, area_names] = run_supervised_classification(data_location, classifier)

	% loading data
	data = load(data_location);
	
	[accuracy,...
	 confusion_mats, area_names] = supervised_classification(data, classifier)
	
end