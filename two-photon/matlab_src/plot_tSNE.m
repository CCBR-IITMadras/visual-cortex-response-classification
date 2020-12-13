function [file_name] = plot_tSNE(data_location, result_location)
	energy=95
	train_percent=0.5

	% loading data
	data = load(data_location);

	area_names=fieldnames(data);
    [~,ind] = sort(area_names);
    ind2 = [1, 3, 5, 2, 4, 6];
    area_names = {area_names{ind}};
    area_names = {area_names{ind2}}';
    area_data = struct2cell(data);
    area_data = {area_data{ind}};
    area_data = {area_data{ind2}}';
	
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

    [coeff, rd_train_data, lamda] = pca(all_data);

    cumEnergy = 0;
    percentEnergy= lamda/sum(lamda)*100;
    for j=1:length(percentEnergy)
        cumEnergy = cumEnergy + percentEnergy(j);
        topdim=j;
        if cumEnergy> energy
            break;
        end
    end
    
    total_train_size = size(all_train_data,1);

    topdim = max(299,topdim);
    topdim = min(topdim,round(total_train_size/5*4));


    disp(["top dimensions:",topdim]);

    rd_train_data = (all_train_data - mu)* coeff(:,1:topdim); 
    rd_test_data = (all_test_data - mu)* coeff(:,1:topdim);



    [~,W,~] = LDA(rd_train_data, all_train_label);

    rd_train_data = rd_train_data * W;

    rd_test_data = rd_test_data * W;

    f=figure();
    hold on;
    
    tsne_Data = tsne(zscore(rd_test_data));
    s=[];   
    for i=1:length(area_names)
        area_tsne = tsne_Data(find(all_test_label==i),:);
        if isempty(area_tsne)
                continue;
        end
        scatter(area_tsne(:,1), area_tsne(:,2),5,'filled');
    end
    legend(s,area_names,'Location','northoutside','Orientation','horizontal');
    file_name = strcat(result_location,'/tSNE.png')
    saveas(f,file_name);
    
end