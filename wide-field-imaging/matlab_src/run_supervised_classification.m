function [accuracy, cm, area_names, ...
		 train_prediction, test_prediction, ...
		 train_map, test_map] = run_supervised_classification(data_location, classifier)


        

	% loading data
	data = load(strcat(data_location,"/data.mat"));
	
	
	map = load(strcat(data_location,"/map.mat"));
	
    [accuracy, cm, area_names, ...
		 train_prediction, test_prediction, ...
		 train_map, test_map] = supervised_classification(data, map, classifier)
	
	
end