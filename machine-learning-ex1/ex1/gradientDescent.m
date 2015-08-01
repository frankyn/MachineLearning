function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    interm_sum = zeros(length(theta), 1);
    % Iterate through each dimension
    for dim = 1:length(theta)
        interm_sum(dim) = 0;
        % Loop through all points 
        for grad = 1:m
            interm_sum(dim) = interm_sum(dim) + (dot(theta', X(grad,:)) - y(grad)) * X(grad,:)(dim);
        end
        
        interm_sum(dim) = theta(dim) - (alpha * 1/m * interm_sum(dim)) ;
    end

    theta = interm_sum;






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
