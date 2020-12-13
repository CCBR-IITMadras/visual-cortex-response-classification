function [accuracy, confusion_mats, area_names] = supervised_classification(data, classifier)
	% configurations
	trials=5;
	train_percent = 0.5;
	energy = 98;

    classifier_function={@gmm_classify, @bayes_classify, @svm_classify, @ann_classify};
        

	%re-organizing data
	area_names=fieldnames(data);
	[~,ind] = sort(area_names);
	ind2 = [1, 3, 5, 2, 4, 6];
	area_names = {area_names{ind}};
	area_names = {area_names{ind2}}';
	area_data = struct2cell(data);
	area_data = {area_data{ind}};
	area_data = {area_data{ind2}}';

	confusion_mats={}

	train_data = cell(size(area_data));
	test_data = cell(size(area_data));

	all_train_label=[];

	all_test_label=[];

	for i = 1 : length(area_names)
		area_specific_data = area_data{i};
		perm = randperm(size(area_specific_data,1));
		
		train_data_size = round(size(area_specific_data,1)*train_percent);
		
		train_data{i}=area_specific_data(perm(1:train_data_size),:);
		
		test_data{i}=area_specific_data(perm(train_data_size+1:end),:);
		
		all_train_label = [all_train_label; ones([size(train_data{i},1),1])*i];
		
		all_test_label = [all_test_label; ones([size( test_data{i},1),1])*i];
	end

	all_data = cell2mat(area_data);

	all_train_data = cell2mat(train_data);

	all_test_data = cell2mat(test_data);

	mu = mean(all_data);

	[coeff, rd_train_data, lamda] =pca(all_data);

	cumEnergy = 0;
	percentEnergy= lamda/sum(lamda)*100;
	for j=1:length(percentEnergy)
		cumEnergy = cumEnergy + percentEnergy(j);
		topdim=j;
		if cumEnergy> energy
			break;
		end
	end

	total_train_size = size(all_train_data,1)
	topdim = max(299,topdim);
	topdim = min(topdim,round(total_train_size/5*4));
	

	rd_train_data = (all_train_data - mu)* coeff(:,1:topdim); 
	rd_test_data = (all_test_data - mu)* coeff(:,1:topdim);

	[~,W,~] = LDA(rd_train_data, all_train_label);

	rd_train_data = rd_train_data * W;

	rd_test_data = rd_test_data * W;



	[predictedTrainLabels, predictedTestLabels] =classifier_function{classifier}(rd_train_data,all_train_label,rd_test_data);
	c=confusionmat(all_test_label, predictedTestLabels);
	accuracy=sum(diag(c))/numel(predictedTestLabels)*100;
	confusion_mats{1}=c;
end