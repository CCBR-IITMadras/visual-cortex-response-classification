function [adaptedGMMDist,alphaK]= MyAdaptGMM(trainData,gmmDist,relevanceFactor)
mixturePrior=gmmDist.ComponentProportion;
mixtureMean=gmmDist.mu;
%mixtureCovariance=squeeze(gmmDist.Sigma);
gammaNK=posterior(gmmDist,trainData);
gammaK=sum(gammaNK);
%[TopGammak,TopGammaInd]=sort(gammaK,'descend');
%fprintf('%d ',TopGammaInd(1:5));
%fprintf('\n'); 
%fprintf('%d ',TopGammak(1:5));
%fprintf('\n'); 
adaptDataExpectation=bsxfun(@rdivide,trainData'*gammaNK,gammaK);
adaptDataExpectation(isnan(adaptDataExpectation))=realmin;
%adaptDataVariance=bsxfun(@rdivide,(trainData.^2)'*gammaNK,gammaK);
%alphaK=(gammaK/noOfTrainFeatures)./((gammaK/noOfTrainFeatures)+relevanceFactor);
alphaK=(gammaK)./((gammaK)+relevanceFactor);

%[TopGammak,TopGammaInd]=sort(alphaK,'descend');
%fprintf('%d ',TopGammak(1:5));
%fprintf('\n'); 
%adaptedGMMPriors=((alphaK.*gammaK)/size(trainData,1))+((1-alphaK).*mixturePrior);

%if sum(adaptedGMMPriors) < 1
%    scaleFactor=(1-sum(adaptedGMMPriors))/length(adaptedGMMPriors);
%    adaptedGMMPriors=adaptedGMMPriors*scaleFactor;
%elseif sum(adaptedGMMPriors) > 1
%    scaleFactor=(sum(adaptedGMMPriors)-1)/length(adaptedGMMPriors);
%    adaptedGMMPriors=adaptedGMMPriors/scaleFactor;        
%end

adaptedGMMMeans=bsxfun(@times,adaptDataExpectation,alphaK)+...
                bsxfun(@times,mixtureMean',1-alphaK);
%adaptedGMMCovaraiance=bsxfun(@times,adaptDataVariance,alphaK)+...
%                bsxfun(@times,(mixtureMean.^2)'+(mixtureCovariance.^2),1-alphaK)...
%                -(mixtureMean.^2)';

%adaptedGMMCovaraiance = reshape(adaptedGMMCovaraiance, [1, size(adaptedGMMCovaraiance)]);
%mixtureCovariance = reshape(mixtureCovariance, [1, size(mixtureCovariance)]);
adaptedGMMDist=gmdistribution(adaptedGMMMeans',gmmDist.Sigma,mixturePrior);
end