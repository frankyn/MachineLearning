function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Possible steps
multiplicative_steps = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% Dimension of multiplicative steps
m = size(multiplicative_steps,2);

% Possible matrix
possible_matrix = zeros(m,m);

for poss_c = 1:m
	for poss_s = 1:m
		% Test_C and Test_S multiplicative_steps
		% poss_c and poss_s
		test_C = multiplicative_steps(poss_c);
		test_S = multiplicative_steps(poss_s);
		
		model = svmTrain(X, y, test_C, @(x1, x2) gaussianKernel(x1, x2, test_S)); 

		predictions = svmPredict(model, Xval);

		% Update possible matrix
		possible_matrix (poss_c,poss_s) = mean(double(predictions ~= yval));
	end
end

[max_val, position] = min(possible_matrix(:)); 

[C,sigma] = ind2sub(size(possible_matrix),position);
C = multiplicative_steps(C);
sigma = multiplicative_steps(sigma);


% =========================================================================

end
