function [predicted_train_labels, predicted_labels] =  bayes_classify(rd_train_data,all_train_label,rd_test_data);
    

    case_number=5;
    classes = length(unique(all_train_label));
    models= cell([classes,1]);

    model= BuildBaysianModel([rd_train_data,all_train_label],case_number,[]);
    
    [cl,test_data_lk,post]=BayesianClassify(model,[rd_test_data,ones([size(rd_test_data,1),1])]);
    %Chossing the area model which gives maximum likyhood as class label
    [~,predicted_labels]=max(test_data_lk,[],2);

    [cl,trian_data_lk,post]=BayesianClassify(model,[rd_train_data,all_train_label]);
    %Chossing the area model which gives maximum likyhood as class label
    [~,predicted_train_labels]=max(trian_data_lk,[],2);


    
 end
