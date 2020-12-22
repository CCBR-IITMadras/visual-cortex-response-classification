function [testAccuracy, predicted_labels,...
		 cm, area_names] = semi_supervised_classification(data, map, retinotopy_labels, gridlength)

	rng('shuffle')

	numOfMixtures=90
	
	%re-organizing data
	area_names=fieldnames(data);
	[~,ind] = sort(area_names);
	ind2 = [1, 3, 5, 2, 4, 6];
	area_names = {area_names{ind}};
	area_names = {area_names{ind2}}';
	area_data = struct2cell(data);
	area_data = {area_data{ind}};
	area_data = {area_data{ind2}}';

	area_map = struct2cell(map);
    area_map = {area_map{ind}};
    area_map = {area_map{ind2}}'; 

    ind3 = [1, 2, 7, 3, 5, 6, 4];
	retinotopy_labels=ind3(retinotopy_labels);

    % converting individual pixel responses to a video where neiboring pixels are adjacent.
    wide_filed_response=zeros(size(retinotopy_labels,1),...
     size(retinotopy_labels,2), size(area_data{1},2));
    pixel_map = zeros(size(retinotopy_labels,1),size(retinotopy_labels,2));
    ind=1;
    for i = 1:length(area_data)
    	for j = 1:size(area_data{i},1)
    		wide_filed_response(area_map{i}(j,1),...
    		 area_map{i}(j,2), :) = area_data{i}(j,:); 
    		pixel_map(area_map{i}(j,1),...
    		 area_map{i}(j,2))=ind;
    		ind=ind+1;
    	end
    end

    
    all_label = []
    for i = 1 : length(area_names)
		area_specific_data = area_data{i};
		all_label = [all_label; ones([size(area_specific_data,1),1])*i];		
	end

	visual_area_responses = cell2mat(area_data);

    visual_area_pixels=sum(sum(retinotopy_labels~=1));

	visual_area_pixel_labels = all_label;
	
	[coeff, visual_area_rd_response, lamda] =pca(visual_area_responses);


	cumEnergy = 0;
	percentEnergy= lamda/sum(lamda)*100;
	percentEnergy= lamda/sum(lamda)*100;
	for j=1:length(percentEnergy)
	    if percentEnergy(j)<0.3
	        break;
	    end
	    topdim=j;
	end

	topdim = max(6,topdim);

	disp(["top dimensions:",topdim]);
	reducedDimensionProjMatrix = coeff(:,1:topdim);
	meanTrainData=mean(visual_area_responses);
	visual_area_rd_response = (visual_area_responses-meanTrainData) * reducedDimensionProjMatrix;



	disp('Spliting data into grids of pixels')
	hind=1;
	ind=1;
	hGridInd=1;
	data=visual_area_rd_response;

	UBMDist = fitgmdist(data, numOfMixtures,...
	    'Options',statset('Display','iter','TolFun',1e-3,'MaxIter',100),...
	    'RegularizationValue',0.1,'CovarianceType','diagonal');

	while hind+ gridlength < size(wide_filed_response,1)
	    vind=1;
	    vGridInd=1;
	    while vind +gridlength< size(wide_filed_response,2)
	        gridGMMInd(hGridInd,vGridInd)=ind;
	        gridMapCallawayInd(hGridInd,vGridInd)= mode(...
	            reshape(...
	            retinotopy_labels(...
	            hind:hind+gridlength-1,...
	            vind:vind+gridlength-1,:...
	            ),...
	            [gridlength*gridlength 1]...
	            )...
	            );
	        if gridMapCallawayInd(hGridInd,vGridInd) ~=1
	            pixels=pixel_map(hind:hind+gridlength-1,vind:vind+gridlength-1);
	            pixels = pixels(:);
	            pixels(pixels==0)=[];
	            
	            rdBatchData{ind}=data(pixels,:);
	            rdGridData{hGridInd,vGridInd}=data(pixels,:);
	            
	            gridGMMInd(hGridInd,vGridInd)=1;
	            isValid(ind)=0;
	        end
	        ind=ind+1;
	        vGridInd=vGridInd+1;
	        vind=vind+gridlength;
	    end
	    hind=hind+gridlength;
	    hGridInd=hGridInd+1;
	end
	gridGMMInd(gridGMMInd~=1)=randperm(length(gridGMMInd(gridGMMInd~=1)));

	gmmDist=cell(1,length(rdBatchData));

	disp('Identifying the center of each area')
	% Diving each areamatrix into train and test sets, get Index only not the test data to save memory
	uniqueAreas= length(unique(retinotopy_labels(:)));
	noOfPixelAreaWise = zeros(1,uniqueAreas);

	%TestSampleAreaWise = zeros(1,uniqueAreas);
	PixelIndexPerArea=cell(1,uniqueAreas);
	[xx,yy]=meshgrid(1:size(gridMapCallawayInd,2),1:size(gridMapCallawayInd,1));
	X=reshape(xx,[size(gridMapCallawayInd,1)*size(gridMapCallawayInd,2),1]);
	Y=reshape(yy,[size(gridMapCallawayInd,1)*size(gridMapCallawayInd,2),1]);
	reshapedGridInd = reshape(gridMapCallawayInd,[size(gridMapCallawayInd,1)*size(gridMapCallawayInd,2),1]);
	AllTrainSamples=[];
	
	for i=2:uniqueAreas
	    AreaIndex=find(reshapedGridInd==i);
	    %AreaIndex=find(retinotopy_labels==i);
	    XPixelIndex = X(AreaIndex);
	    YPixelIndex = Y(AreaIndex);
	    [westX,minInd]=min(XPixelIndex);
	    westY=YPixelIndex(minInd);
	    [eastX,minInd]=max(XPixelIndex);
	    eastY=YPixelIndex(minInd);
	    
	    
	    
	    [northY,minInd]=max(YPixelIndex);
	    northX=XPixelIndex(minInd);
	    [southY,minInd]=min(YPixelIndex);
	    southX=XPixelIndex(minInd);
	    
	    
	    centerpoint= [ mean(XPixelIndex), mean(YPixelIndex)];
	    py1=round( centerpoint(2));
	    px1=round(centerpoint(1));
	    TrainSampleAreaWise{i} = [py1,px1]; %py1-1,px1; py1,px1-1; py1-1,px1-1];
	    AreaIndex=pixel_map(retinotopy_labels==i);
	    AreaIndex=AreaIndex(:);
	    PixelIndexPerArea{i}=AreaIndex(randperm(length(AreaIndex)));
	    TestSampleAreaWise(i)=length(AreaIndex);
	    AllTrainSamples=[AllTrainSamples;TrainSampleAreaWise{i}];
	end
	converged=0;
	itr=1;


	while converged == 0
	    TrainDataAreaWise = cell(1,uniqueAreas);
	    for i=2:uniqueAreas
	        TrainDataAreaWise{i} =[];
	        for j = 1: size(TrainSampleAreaWise{i},1)
	            TrainDataAreaWise{i} = [TrainDataAreaWise{i}; rdGridData{TrainSampleAreaWise{i}(j,1),TrainSampleAreaWise{i}(j,2)}];
	        end
	    end
	    L=retinotopy_labels;
	    L=imresize(L,[size(gridMapCallawayInd,1),size(gridMapCallawayInd,2)],'nearest');
	    MC=zeros(size(gridMapCallawayInd));
	    for i=2:uniqueAreas
	        for j = 1: size(TrainSampleAreaWise{i},1)
	            MC(TrainSampleAreaWise{i}(j,1),TrainSampleAreaWise{i}(j,2))=i;
	        end
	    end
	    
	    %% Test data preparation
	    %Combining all test data
	    
	    
	    %% Testing GMM with inner boundaries only
	    %testing GMM on test data
	    %For every pixel precdicting liklyhood of model trained on each callaway area
	    rdTestData = [data, ones([size(data,1), 1])];
	    allPosterior = [];
	    
	    ind=1;
	    UpdatedSampleAreaWise = cell(1,uniqueAreas);
	    neibhourBICs = cell(size(gridMapCallawayInd));
	    BIC=[];
	    for i=1:size(gridMapCallawayInd,1)
	        for j=1:size(gridMapCallawayInd,2)
	            if(gridMapCallawayInd(i,j)~=1 )
	                neighbourBlocks = CheckNeighbouringBlocksForSemi(i,j,TrainSampleAreaWise);
	                    if ~isempty(neighbourBlocks) && ~ismember([i,j],AllTrainSamples,'rows')
	                        blockData = rdGridData{i,j};
	                        [blockModel] = MyAdaptGMM(rdGridData{i,j},UBMDist,20);
	                        blockLhood =  pdf(blockModel,rdGridData{i,j});
	                        totBlockLhood = sum(blockLhood);
	                        for n=1:length(neighbourBlocks)
	                            areaTrainData= TrainDataAreaWise{neighbourBlocks(n)};
	                            [areaModel]  =MyAdaptGMM(areaTrainData,UBMDist,20);
	                            areaLhood =  pdf(areaModel,areaTrainData);
	                            totAreaLhood = sum(areaLhood);
	                            combinedData = [areaTrainData; blockData];
	                            [combinedModel]  = MyAdaptGMM(combinedData,UBMDist,20);
	                            combinedLhood =  pdf(combinedModel,combinedData);
	                            totCombLhood = sum(combinedLhood);
	                            neibhourBICs{i,j}(n)= totCombLhood - (totAreaLhood+totBlockLhood);
	                        end
	                    BIC=[BIC; max(neibhourBICs{i,j})];
	                end
	            end
	        end
	    end
	    sortedBIC = sort(BIC,'descend');
	    per_remain=(1-(size(AllTrainSamples,1)/sum(reshapedGridInd~=1)));
	    curr_per=per_remain*0.6+0.2;
	    curr_per=min([0.8,curr_per]);
	    curr_per=max([0.2,curr_per]);
	    
	    disp(curr_per);
	    if round(length(sortedBIC)*curr_per) < 1
	        converged = 1;
	        break;
	    end
	    threshold = sortedBIC(round(length(sortedBIC)*curr_per));
	    for i=1:size(gridMapCallawayInd,1)
	        for j=1:size(gridMapCallawayInd,2)
	            if(gridMapCallawayInd(i,j)~=1 )
	                neighbourBlocks = CheckNeighbouringBlocksForSemi(i,j,TrainSampleAreaWise);
	                if ~isempty(neibhourBICs{i,j})
	                    [maxBIC,areaNo] = max(neibhourBICs{i,j});
	                    if maxBIC>=threshold  && ~ismember([i,j],AllTrainSamples,'rows')
	                        UpdatedSampleAreaWise{neighbourBlocks(areaNo)} =[UpdatedSampleAreaWise{neighbourBlocks(areaNo)}; i,j];
	                        AllTrainSamples = [AllTrainSamples;i,j ] ;
	                        
	                    end
	                end
	            end
	        end
	    end
	    for i=2:uniqueAreas
	        TrainSampleAreaWise{i} =[TrainSampleAreaWise{i}; UpdatedSampleAreaWise{i}];
	    end
	    converged = 1;
	    for i=2:uniqueAreas
	        if ~isempty(UpdatedSampleAreaWise{i})
	            converged =0;
	            break;
	        end
	    end
	    if itr<=5
	        itr = itr+1;
	        continue;
	    end
	    disp('Classification using supervised GMM');
	    totalTestDataMapCallawayLabels = [];
	    bayesianTrainData = [];
	    bayesianTestData =[];
	    trainLabel=[];
	    disp(uniqueAreas)
	    for i=2:uniqueAreas
	        areaTestData = data(PixelIndexPerArea{i},:);
	        bayesianTestData =[bayesianTestData; areaTestData];
	        arealabels = ones(1,length(PixelIndexPerArea{i})) * (i-1);
	        totalTestDataMapCallawayLabels=[totalTestDataMapCallawayLabels,arealabels];
	        areaTrainData= TrainDataAreaWise{i};
	        bayesianTrainData = [ bayesianTrainData; areaTrainData];
	        trainLabel = [trainLabel; ones([size(areaTrainData,1),1])*i-1];
	    end

	    [predictedTrainLabels, predictedLabels] =gmm_classify(bayesianTrainData,trainLabel,bayesianTestData);
	    
	    

	    cm=confusionmat(totalTestDataMapCallawayLabels,predictedLabels);
	    
	    %plotconfusion(ind2vec(totalTestDataMapCallawayLabels'),ind2vec(predictedLabels'));
	    %saveas(h,[saveDir{stimulusInd},'/',num2str(expType),'/',num2str(centerPercentage*100),'_cm2.png']);
	    %saveas(h,[saveDir{stimulusInd},'/',num2str(expType),'/',num2str(centerPercentage*100),'_cm2.fig']);
	    testAccuracy=sum(diag(cm(2:end,2:end)))/sum(sum(cm(2:end,2:end)))*100;
	    disp(testAccuracy)
	    


	    %% Generating final plots
	    [~,predictedLabels] = gmm_classify(bayesianTrainData,trainLabel,data);
	    %save([saveDir num2str(SizeTrainData) 'testLabels.mat'],'predictedLabels');
	    
	    ind=1;
	    predictedWideFieldsLabels=zeros(size(retinotopy_labels));
	    for i=1:size(wide_filed_response,1)
	        for j=1:size(wide_filed_response,2)
	            if retinotopy_labels(i,j)~=1
	                predictedWideFieldsLabels(i,j)=predictedLabels(pixel_map(i,j));
	            else
	                predictedWideFieldsLabels(i,j)=0;
	            end
	        end
	    end
	    
	    itr = itr +1;
	end
	predicted_labels{1}=predictedWideFieldsLabels;
end