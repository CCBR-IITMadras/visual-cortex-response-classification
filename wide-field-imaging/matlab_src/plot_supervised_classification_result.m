function [figure_location] = plot_supervised_classification_result(data_location, result_location)

	% loading data
	load(strcat(result_location,"/results.mat"),"train_map", "test_map", "train_prediction", "test_prediction","cm","names");
	train_map = squeeze(train_map(1,:,:));
	test_map = squeeze(test_map(1,:,:));
	load(strcat(data_location,"/retinotopy.mat"), "retinotopy_labels");

	f=figure();
	set (f, 'Position', [300 300 1000 400])
	MC=zeros(size(retinotopy_labels));
    mapCallawayAreas = length(unique(retinotopy_labels));
    MC(sub2ind(size(retinotopy_labels),train_map(:,1),train_map(:,2)))=100;
    sp1=subplot(1,3,1);
    h=imagesc(1-MC);
    colormap(sp1,bone);
    hold on;
    [X,Y]=meshgrid(1:size(retinotopy_labels,2),1:size(retinotopy_labels,1));
    contour(X,Y,retinotopy_labels,length(unique(retinotopy_labels)),'LineColor','k','Linewidth',2);
    MapCallaway=~(retinotopy_labels==1);
   	daspect([1 1 1 ])
   	axis off
   	box off
   	title('Training Pixels')
    set(h,'Alphadata',MapCallaway);

    sp2=subplot(1,3,2);
   	L=retinotopy_labels;

	[height,width]=size(L);
	
	MC=zeros(size(L));
	MC(sub2ind(size(L),train_map(:,1),train_map(:,2)))=train_prediction(1,:);
	MC(sub2ind(size(L),test_map(:,1),test_map(:,2)))=test_prediction(1,:);
	h=imagesc(MC);
	colormap(sp2,parula);
	
	hold on;

    [X,Y]=meshgrid(1:size(L,2),1:size(L,1));
    contour(X,Y,L,length(unique(L)),'LineColor','k','Linewidth',2);
    MapCallaway=~(L==1);
    set(h,'Alphadata',MapCallaway);
   	for K = 1 : 6; hidden_h(K-1+1) = surf(uint8([K K;K K]), 'edgecolor', 'none'); end
	uistack(hidden_h, 'bottom');
	unames=cellstr(names)
	unames{7}= "Retinotopic Borders"	
	legend(hidden_h,unames,'Location','southoutside','Orientation','horizontal')
    daspect([1 1 1 ])
    axis off
   	box off
    title('classification result')
    
    sp3=subplot(1,3,3);
    plotConfMat(squeeze(cm(1,:,:)), names, sp3);
   	daspect([1 1 1 ])
   	title('Confusion Matrix')
    set(h,'Alphadata',MapCallaway);




    figure_location = strcat(result_location,'/supervised_classification.png');
   set(gca,'LooseInset',get(gca,'TightInset'));
	saveas(gcf,figure_location)
    
end