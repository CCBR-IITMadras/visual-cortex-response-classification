function [figure_location] = plot_confusion_mat(result_location)
	   

	% loading data
	load(strcat(result_location,"/results.mat"),"cm","names");
	

	f=figure();
  plotConfMat(squeeze(cm(1,:,:)), names);
  daspect([1 1 1 ])
  title('Confusion Matrix')




  figure_location = strcat(result_location,'/supervised_classification.png');
  set(gca,'LooseInset',get(gca,'TightInset'));
	saveas(gcf,figure_location)
    
end