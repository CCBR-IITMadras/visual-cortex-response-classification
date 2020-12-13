function [classLabels,lhood,posterior] = BayesianClassify(model, data)
% function [classLabels] = BayesianClassify(model, testData)
%
% Gives the class labels of testData according to the given model
%
% INPUT:
%
% model    : built model in above function (Variable arguments)
% testData     : m x n matrix, m is num of examples & n is
% number of dimensions.
%
% OUTPUT:
%
% classLabels: m x 1 matrix, labels of testData, 1 for class 1, ... , k for
% class k.
%
% See Also : BuildBaysianModel.m
%

m = size(data, 1); % number of examples
n = size(data, 2); % number of feature dimension
k = size(model, 1); % number of classes

classLabels  = zeros(m,1);
% for i=1:size(data,1)
%     for j=1:size(model,1)
%         lhood(i,j) = likilehood(model{j,1},model{j,2}, data(i,1:end-1))*1/k;
%     end
%     [~,classLabels(i)]=max(lhood(i,:));
%     %lhood(i,:)=lhood(i,:)/(sum(lhood(i,:)/k));
% % Complete the function
% 
% end

    for j=1:size(model,1)
        lhood(:,j) = mvnpdf( data(:,1:end-1),model{j,1},model{j,2});
    end
    [~,classLabels]=max(lhood,[],2);
    
    posterior=(lhood)./repmat((sum(lhood')'),1,k);
    
% Complete the function



% Complete the function

end