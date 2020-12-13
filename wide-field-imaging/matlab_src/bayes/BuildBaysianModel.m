function [parameters] = BuildBaysianModel(trainData, caseNumber, itr)
% function [Variable arguments] = BuildBaysianModel(trainData, crossValidationData, caseNumber)
%
% Builds Bayesian model using the given training data and cross validation
% data (optional) for the given case number.
%
% INPUT:
%
% trainData     : m x n+1 matrix, m is num of examples & n is number of
% dimensions. n+1 th column is for class labels (1 -- for class 1, ... k --
% for class k).
%
% crossValidationData     : (Optional) m x n+1 matrix, m is num of examples & n is
% number of dimensions. n+1 th column is for class labels (1 -- for class
% 1, ... , k -- for class k).
%
% caseNumber: 1 -- Bayes with Covariance same for all classes
%             2 -- Bayes with Covariance different for all classes
%             3 -- Naive Bayes with C = \sigma^2*I
%             4 -- Naive Bayes with C same for all
%             5 -- Naive Bayes with C different for all
%
% OUTPUT:
% model    : k x 2 cell, k is num of classes.
%            Each row i is {muHat(mean_vector)_i, C(covariance_matrix)_i}
%
% See Also : BayesianClassify.m
%

m = size(trainData, 1); % number of training examples
n = size(trainData, 2) - 1; % number of feature dimension
k = length(unique(trainData(:, end))); % number of classes
parameters=cell(k,2);


switch caseNumber
    case 1
         for i=1:k
         feature_vector=trainData(trainData(:,end)==i,1:end-1);
         mu(i,:)=mean(feature_vector,1);
         sigma(:,:,i)=cov(feature_vector);
         end
         sigma=mean(sigma,3);
         for i=1:k
         parameters{i,1}=mu(i,:);
         parameters{i,2}=sigma;
         end
    case 2
        for i=1:k
        feature_vector=trainData(trainData(:,end)==i,1:end-1);
        mu=mean(feature_vector,1);
        sigma=cov(feature_vector);
        parameters{i,1}=mu;
        parameters{i,2}=sigma;

        end
    case 3
        for i=1:k
        feature_vector=trainData(trainData(:,end)==i,1:end-1);
        mu(i,:)=mean(feature_vector,1);
        sigma(:,:,i)=cov(feature_vector);
         end
         sigma=mean(sigma,3);
         sigma= diag(sigma);
         sigma = max(max(sigma))*eye(n);
         for i=1:k
            parameters{i,1}=mu(i,:);
            parameters{i,2}=sigma;
        end
    case 4
        for i=1:k
            feature_vector=trainData(trainData(:,end)==i,1:end-1);
            mu(i,:)=mean(feature_vector,1);
            
            %sigma(:,:,i)=cov(feature_vector);
        end
         %sigma=mean(sigma,3);
         sigma=cov(trainData(:,1:end-1));
         for i=1:k
         sigma= diag(diag(sigma));
         parameters{i,1}=mu(i,:);
         parameters{i,2}=sigma;
        end
    case 5
        for i=1:k
            feature_vector=trainData(trainData(:,end)==i,1:end-1);
            mu=mean(feature_vector,1);
            sigma=cov(feature_vector);
            sigma= diag(diag(sigma));
            parameters{i,1}=mu;
            parameters{i,2}=sigma;
        end
    case 6
        for i=1:k
            feature_vector=trainData(trainData(:,end)==i,1:end-1);
            mu=mean(feature_vector,1);
            sigma=ones(size(feature_vector,2)) * 0.0001 * itr/2;
            sigma= diag(diag(sigma));
            parameters{i,1}=mu;
            parameters{i,2}=sigma;
        end    
end

%class= BayesianClassify(parameters,trainData)
%trainData(:,end)
%trainAccuracy=trace(confusionmat(class',trainData(:,end)))/size(trainData,1)*100;


end


