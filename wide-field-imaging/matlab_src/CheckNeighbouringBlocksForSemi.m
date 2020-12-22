function neighbourArea = CheckNeighbouringBlocksForSemi(i,j,TrainSampleAreaWise)
    neighbourArea = [];
    for area = 2:length(TrainSampleAreaWise)
        if ismember([i+1,j],TrainSampleAreaWise{area},'rows')
            neighbourArea = [neighbourArea; area];
            continue;
        elseif ismember([i,j+1],TrainSampleAreaWise{area},'rows')
            neighbourArea = [neighbourArea; area];
            continue;
        elseif ismember([i-1,j],TrainSampleAreaWise{area},'rows')
            neighbourArea = [neighbourArea; area];
            continue;
        elseif ismember([i,j-1],TrainSampleAreaWise{area},'rows')
            neighbourArea = [neighbourArea; area];
            continue;    
        end
    end
end