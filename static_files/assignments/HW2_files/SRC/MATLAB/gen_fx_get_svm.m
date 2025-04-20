unction [out , mdl] = gen_fx_get_svm(grp, I, rate, rep)
% gen_fx_get_svm - Trains and evaluates an SVM classifier over multiple repetitions.
%
% Syntax:
%   [out, mdl] = gen_fx_get_svm(grp, I, rate, rep)
%
% Inputs:
%   grp  - A vector of class labels for each sample (Nx1)
%   I    - Feature matrix (NxD), where N is the number of samples and D is the number of features
%   rate - Proportion of data to use for training in each repetition (e.g., 0.7 for 70%)
%   rep  - Number of repetitions to run the evaluation
%
% Outputs:
%   out - Struct with fields:
%         pt  - Vector of classification accuracies for each repetition
%         C   - Confusion matrices for each repetition
%         tu  - Vector of per-class accuracies for each repetition
%         model - The SVM model from the last repetition
%   mdl - Same as out.model (for backward compatibility or convenience)
%
% This function repeatedly splits the dataset into training and testing subsets,
% trains an SVM classifier, evaluates its performance, and accumulates the results.

Pt = [];  % Accuracy scores for each repetition
Tu = [];  % Per-class accuracies for each repetition

for cnt = 1:rep
    % Split the data into training and testing using equal class distribution
    [test, train] = gen_fx_get_equal_part(grp, rate);

    % Train and test the SVM classifier
    [cls , mdl] = gen_fx_MC_SVM(I(test,:), I(train,:), grp(train));

    % Calculate accuracy for this repetition
    Pt = [Pt; sum(cls == grp(test)) / sum(test)];

    % Compute the confusion matrix
    Ct(:,:,cnt) = confusionmat(cls, grp(test));

    % Compute per-class accuracy for this repetition
    Tu = [Tu diag(Ct(:,:,cnt)) ./ (sum(Ct(:,:,cnt)))'];
end

% Store results in the output structure
out.C = Ct;
out.pt = Pt;
out.tu = Tu;
out.model = mdl;
end