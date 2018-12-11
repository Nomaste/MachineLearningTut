function g = predict2(x)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 

%g = zeros(size(z));
thetaA = [1 -1.5 3.7; 1 5.1 2.3];
thetaAswap = [1 5.1 2.3; 1 -1.5 3.7];
thetaB = [1 0.6 -0.8];
thetaBswap = [1 -0.8 .6];

a3_1 = sigmoid(thetaB*[1;sigmoid(thetaA * x)]);
a3_2 = sigmoid(thetaBswap*[1;sigmoid(thetaAswap * x)]);
g = a3_1-a3_2
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).





% =============================================================

end
