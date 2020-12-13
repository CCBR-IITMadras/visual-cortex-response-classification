function plotConfMat(varargin)
%PLOTCONFMAT plots the confusion matrix with colorscale, absolute numbers
%   and precision normalized percentages
%
%   usage: 
%   PLOTCONFMAT(confmat) plots the confmat with integers 1 to n as class labels
%   PLOTCONFMAT(confmat, labels) plots the confmat with the specified labels
%
%   Vahe Tshitoyan
%   20/08/2017
%
%   Arguments
%   confmat:            a square confusion matrix
%   labels (optional):  vector of class labels

% number of arguments
switch (nargin)
    case 0
       confmat = 1;
       labels = {'1'};
    case 1
       confmat = varargin{1};
       labels = 1:size(confmat, 1);
    case 2
      confmat = varargin{1};
      labels = varargin{2};
    otherwise
      confmat = varargin{1};
      secConfmat = varargin{2};
      labels =  varargin{3};   
end

confmat(isnan(confmat))=0; % in case there are NaN elements
numlabels = size(confmat, 1); % number of labels

% calculate the percentage accuracies
confpercent = 100*confmat./repmat(sum(abs(confmat), 1),numlabels,1);
confpercent = confpercent + confpercent';
upperconfpercent = triu(confpercent,0)';
upperconfpercent(upperconfpercent==0) = min(min(confpercent));
% plotting the colors
imagesc(upperconfpercent);

% set the colormap
colormap(flipud(gray));


% Setting the axis labels
set(gca,'XTick',1:numlabels,...
    'XTickLabel',labels,...
    'YTick',1:numlabels,...
    'YTickLabel',labels,...
    'TickLength',[0 0]);
end