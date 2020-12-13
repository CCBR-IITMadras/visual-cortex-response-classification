function [predicted_train_labels,predicted_labels] =  ann_classify(rd_train_data,all_train_label,rd_test_data);

    classes = length(unique(all_train_label));
    models= cell([classes,1]);


    net = patternnet(30);
    net = train(net, transpose(rd_train_data), ind2vec(all_train_label'));

    total_test_data_lk = net(transpose(rd_test_data)); 
    [~,predicted_labels]=max(transpose(total_test_data_lk),[],2);

    total_train_data_lk = net(transpose(rd_train_data)); 
    [~,predicted_train_labels]=max(transpose(total_train_data_lk),[],2);
    
 end
