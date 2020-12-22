function [file_name] = plot_reponse_correlation(data_location, result_location)

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
	
	for i = 1 : 6
		for j = i : 6
			if i == j
				area_data1 = area_data{i};
				[R, P]=  corrcoef(area_data1' * 10000);;
				R = triu(R,1);
				P = triu(P,1);
				R = R(:);
				P = P(:);
				P(R==0)=[];
				R(R==0)=[];

				sig_R = R;
				sig_R(P>0.05) = [];
				sigCorrMat(i,j) = mean(sig_R);


				corrMat(i,j) = mean(R)


			else

				area_data1 = area_data{i};
				data1_size = size(area_data1,1);
				area_data2 = area_data{j};
				data2_size = size(area_data2,1);

				combine_data = [area_data1;area_data2];

				size(combine_data);

				[R,P] = corrcoef(combine_data'* 100000);
				R = R(1:data1_size, data1_size+1:end);
				P = P(1:data1_size, data1_size+1:end);
				R = R(:);
				P = P(:);
				P(R==0)=[];
				R(R==0)=[];

				sig_R = R;
				sig_R(P>0.05) = [];
				sigCorrMat(i,j) = mean(sig_R);
				sigCorrMat(j,i) = mean(sig_R);

				corrMat(i,j) = mean(R);
				corrMat(j,i) = mean(R)
			end
		end
	end

    h= figure();
    plotMat(corrMat,area_names);
    box off;
    file_name = strcat(result_location,'/Correlation.png'); 
	saveas(h, file_name);
    
end