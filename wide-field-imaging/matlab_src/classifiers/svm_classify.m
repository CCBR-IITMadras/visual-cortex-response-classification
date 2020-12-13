function [predicted_train_labels, predicted_labels] =  svm_classify(rd_train_data,all_train_label,rd_test_data);

	rd_train_data=double(rd_train_data);
	all_train_label=double(all_train_label);
	rd_test_data=double(rd_test_data);
	% tmp = strcat('tmp/svm_files');
	% disp(tmp);
	% mkdir(tmp);
	% dlmwrite(strcat(tmp,'/trian_data'),rd_train_data);
	% dlmwrite(strcat(tmp,'/trian_label'),all_train_label);
	% dlmwrite(strcat(tmp,'/test_data'),rd_test_data);
	% disp(strcat({'python libraries/classifiers/train_and_test_svm.py --trainData '},tmp,{'/trian_data'},...
	% 		{' --testData '},tmp,'/test_data',...
	% 		{' --trainLabel '},tmp,'/trian_label',...
	% 		{' --trainPrediction '},tmp,'/train_pred',...
	% 		{' --testPrediction '},tmp,'/test_pred'));
	% system(strcat({'python libraries/classifiers/train_and_test_svm.py --trainData '},tmp,{'/trian_data'},...
	% 		{' --testData '},tmp,'/test_data',...
	% 		{' --trainLabel '},tmp,'/trian_label',...
	% 		{' --trainPrediction '},tmp,'/train_pred',...
	% 		{' --testPrediction '},tmp,'/test_pred'));
	
	% predicted_labels = dlmread(strcat(tmp,'/test_pred'));

	% predicted_train_labels = dlmread(strcat(tmp,'/train_pred'));

	% system(strcat({'rm -rf '},tmp));

	[models,labelSet] =ovr_train(rd_train_data,all_train_label);
	predicted_labels=ovr_predict(rd_test_data,models,labelSet);
	predicted_train_labels=ovr_predict(rd_train_data,models,labelSet);
    
 %    model = fitcsvm(rd_train_data,all_train_label,['-s 0 -t 0 -c 0.01 -n 0.1']);


    % [predicted_labels, decision_values] = predict(doube(ones([1, size(rd_test_data,1)])),rd_test_data, model);
      
       
    % [predicted_train_labels, decision_values] = predict(all_train_label,rd_train_data, model);
      

    
 end


 function [models,labelSet] =  ovr_train(data, labels)
    labelSet = sort(unique(labels))
    labelSetSize = length(labelSet)
    for label=1:labelSetSize
        binaryLabels = labels == labelSet(label);
        model = fitcsvm(data,binaryLabels','KernelFunction','linear','ScoreTransform','logit')
        models{label}=model
    end
end

function [labels,scores] =  ovr_predict(data, models, labelSet)
    labelSetSize = length(labelSet)
    scores = []
    for label=1:labelSetSize
        [~,score] = predict(models{labelSet(label)},data)
        score = score(:, 2)  % true class score
        scores(:,label)=score
    end
    [~,labels] = max(scores,[],2) 
end
