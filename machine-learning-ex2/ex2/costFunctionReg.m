function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


interm_sum = 0;
reg_sum = 0;

% Sum up theta for regularized cost (skip theta(1,:))
for it = 2:length(theta)
	reg_sum = reg_sum + theta(it,:).^2;
end 

h_t = sigmoid(X*theta);

for it = 1:m
	interm_sum = interm_sum + (-y(it,:)*log(h_t(it,:)) - (1-y(it,:))*log(1-h_t(it,:)));
end

% Calculated cost
J = 1/m * interm_sum + lambda/(2*m) * reg_sum;

% Gradient
hy = h_t - y;

tmp_grad = grad;

% J_theta 0

for dim = 1:length(grad)
	interm_sum = 0;
	for it = 1:m
		interm_sum = interm_sum + hy(it,:) .* X(it,dim);
	end
	if dim == 1
		tmp_grad(dim,:) = 1/m * interm_sum;
	else
		tmp_grad(dim,:) = 1/m * interm_sum + lambda/m * theta(dim,:);
	end
end

grad = tmp_grad;

% =============================================================

end
