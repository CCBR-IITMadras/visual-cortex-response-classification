function [predicted_train_labels, predicted_labels] =  gmm_classify(rd_train_data,all_train_label,rd_test_data);
    
    num_of_mixtures=10;

	classes = length(unique(all_train_label));
    models= cell([classes,1]);

    for i=1:classes
        models{i} = fitgmdist(rd_train_data(find(all_train_label==i),:),...
            num_of_mixtures, 'Options',statset('TolFun',1e-5,'MaxIter',100 ),...
            'RegularizationValue',0.1, 'CovarianceType','diagonal');
    end

    test_data_lk=zeros(size(rd_test_data,1),classes);
    for i=1:classes
        test_data_lk(:,i) = pdf(models{i},rd_test_data);
    end

    [~,predicted_labels]=max(test_data_lk,[],2);


    trian_data_lk=zeros(size(rd_train_data,1),classes);
    for i=1:classes
        trian_data_lk(:,i) = pdf(models{i},rd_train_data);
    end

    [~,predicted_train_labels]=max(trian_data_lk,[],2);
 end