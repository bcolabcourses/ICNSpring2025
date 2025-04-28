function [test, train] = gen_fx_get_equal_part(grp, rate)
% gen_fx_get_equal_part - Splits data into balanced training and testing sets for each class.
%
% Syntax:
%   [test, train] = gen_fx_get_equal_part(grp, rate)
%
% Inputs:
%   grp  - A vector of class labels for each sample (Nx1)
%   rate - Proportion of samples per class to assign to the training set (e.g., 0.7 for 70%)
%
% Outputs:
%   test  - Logical vector indicating the test samples (1 for test, 0 otherwise)
%   train - Logical vector indicating the train samples (1 for train, 0 otherwise)
%
% This function ensures each class is equally represented by selecting the same number
% of samples (equal to the smallest class size) for both training and testing sets.

catNo = unique(grp);  % Unique class labels
asiz = [];            % Stores the number of samples in each class

% Get the number of samples in each class
for cat = catNo'
    asiz = [asiz sum(grp == cat)];
end

% Minimum number of samples among all classes
minL = min(asiz);

% Calculate number of training and testing samples per class
TrSiz = floor(rate * minL);
TeSiz = minL - TrSiz;

% Initialize training and testing index vectors
train = zeros(size(grp));
test = zeros(size(grp));

% For each class, randomly split into training and testing
for cat = catNo'
    ix = grp == cat;        % Find indices for current class
    iix = find(ix);         % Get linear indices
    Rrp = randperm(length(iix));  % Shuffle the indices randomly
    
    % Assign the first TrSiz to training and the remaining to testing
    train(iix(Rrp(1:TrSiz))) = 1;
    test(iix(Rrp(end-TeSiz+1:end))) = 1;
end

% Convert to logical vectors
train = logical(train);
test = logical(test);
end
