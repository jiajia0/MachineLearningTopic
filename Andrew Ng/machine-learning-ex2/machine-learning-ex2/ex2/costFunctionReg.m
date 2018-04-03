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

% 先计算出没有正则项的
J = (-y' * log(sigmoid(X * theta)) - (1 - y)' * log(1 - sigmoid(X * theta))) / m;

% 再计算有正则项的损失函数,这里计算正则项时不计算第一个参数
J = J + (lambda/(2 * m)) * sum(theta(2:end) .^ 2);

% 计算第一个参数的导数
grad(1,:) = ((sigmoid(X*theta) - y)' * X(:,1)) / m; 

% 计算其他参数的导数
grad(2:end,:) = (X(:,2:end)' * (sigmoid(X*theta) - y)) / m + lambda/m * theta(2:end,:);

% =============================================================

end
