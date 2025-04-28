function RDM = RDM_maker(data , region , dataType)


% data: 3D array [nNeurons x nStimuli x nTimes] = (858 x 144 x 171)
[nNeurons, nStimuli, nTimes] = size(data);

% Initialize the RDM array: [144 x 144 x 171]
RDM = zeros(nStimuli, nStimuli, nTimes);

for t = 1:nTimes
    % Extract data across all neurons/stimuli for time slice t
    % sliceData is [858 x 144]
    sliceData = data(:,:,t);
    
    % Compute correlation among stimuli (columns)
    % corr(sliceData) returns a 144 x 144 correlation matrix
    
    for i = 1: size(sliceData,2)
        for j = 1: size(sliceData,2)
            corMat = corr(sliceData(:,i) , sliceData(:,j));
    
            % Convert correlation to dissimilarity = 1 - correlation
            RDM(i,j,t) = 1 - corMat;
        end
    end
end
% imagesc for a sample time slice
imagesc(RDM(i,j,60));

save(['RDM_Matrix_',dataType ,'_', region, '.mat'], 'RDM', '-v7.3');
end


