function [class, Mdl] = gen_fx_MC_SVM(sample, train, grp)
% gen_fx_MC_SVM - Multi-class SVM classification using one-vs-one strategy with voting.
%
% Syntax:
%   [class, Mdl] = gen_fx_MC_SVM(sample, train, grp)
%
% Inputs:
%   sample - Feature matrix of samples to classify (MxD)
%   train  - Feature matrix of training samples (NxD)
%   grp    - Class labels for the training samples (Nx1)
%
% Outputs:
%   class - Predicted class labels for the input samples (Mx1)
%   Mdl   - Trained ECOC (Error-Correcting Output Codes) model using SVM learners
%
% Description:
%   This function performs multi-class classification by constructing binary
%   SVM classifiers for each pair of classes (one-vs-one scheme). For each
%   sample, it uses a majority voting scheme to assign the final class.
%   Additionally, it trains an ECOC model as a compact representation for the final output.

classNo = double(unique(grp));    % Unique class labels
cls_nu = length(classNo);         % Number of classes

% Generate all possible binary class pairs (one-vs-one strategy)
r = nchoosek(1:cls_nu, 2);

vote = [];  % Will hold votes from all pairwise classifiers

% Loop through each class pair and train a binary SVM
for i = 1:length(r(:,1))
    % Select samples from only the two classes
    ix = (grp == classNo(r(i,1))) | (grp == classNo(r(i,2)));
    
    % Train binary SVM on selected samples
    svmStruct = fitcsvm(train(ix,:), grp(ix));
    
    % Predict class for all test samples and collect the votes
    vote = [vote predict(svmStruct, sample)];
end

% Determine final predicted class by majority vote
class = mode(vote, 2);

% Train a multi-class SVM using ECOC as a model output (not used in prediction here)
Mdl = fitcecoc(train, grp);
% Optionally, one could use:
% class = predict(Mdl, sample);
% to classify using the ECOC model directly instead of voting
end
