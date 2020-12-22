function [figure_location] = plot_semi_supervised_classification_result(data_location, result_location)

	% loading data
	load(strcat(result_location,"/results.mat"),"predicted_labels","names");
	load(strcat(data_location,"/retinotopy.mat"), "retinotopy_labels");
  size(retinotopy_labels)
	f=figure();
	
  L=retinotopy_labels;

	[height,width]=size(L);
  predicted_labels = squeeze(predicted_labels(1,:,:));
	MC = predicted_labels;
  h=imagesc(MC);
	colormap(parula);
	
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

  

  figure_location = strcat(result_location,'/semosupervised_classification.png');
  set(gca,'LooseInset',get(gca,'TightInset'));
  saveas(gcf,figure_location)
    
end