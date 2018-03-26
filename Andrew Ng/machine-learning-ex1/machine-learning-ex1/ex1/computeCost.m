function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
J = X * theta; % 参数与对应的数据相乘，得到一个m*1的矩阵
J = J - y; % 计算出每个数据的误差
J = J' * J; % 平方运算并相加
J = J ./ (length(X)*2);

% =========================================================================

end
