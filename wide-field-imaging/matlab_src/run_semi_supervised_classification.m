function [testAccuracy, predictedWideFieldsLabels,...
		 cm, area_names] = run_semi_supervised_classification(data_location, gridlength)

	rng('shuffle')
	% loading data
	data = load(strcat(data_location,"/data.mat"));
	map = load(strcat(data_location,"/map.mat"));
	load(strcat(data_location,"/retinotopy.mat"),"retinotopy_labels");

	[testAccuracy, predictedWideFieldsLabels,...
		 cm, area_names] = semi_supervised_classification(data, map, retinotopy_labels, gridlength);
end