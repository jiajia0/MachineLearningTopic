function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% 先用特征矩阵和参数矩阵相乘，计算出总和，这个总和就是用来计算sigmoid的参数z
z = X * theta;

% 然后使用sigmoid计算出数值，也就是预测数值
h = sigmoid(z);

% 然后计算出损失值
J = ((-y' * log(h)) - ((1 - y)' * (log(1 - h)))) / m;

% 计算出每个预测值和真实值的差值
dis = h - y;

% 利用矩阵相乘计算出参数
grad = (X' * dis) / m;

% 计算出最后的参数
grad = sum(grad,2);

% =============================================================

end
