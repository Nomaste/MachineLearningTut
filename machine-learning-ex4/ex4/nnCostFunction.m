function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Part 1
m = size(X, 1);
a1 = [ones(m, 1) X];
a2 = sigmoid(Theta1*a1');
b2 = a2';
a22 = [ones(m, 1) b2];
a3 = sigmoid(Theta2*a22');
b3=a3';
%fprintf('Value of size - b3 \n')
%size(b3)
ya = yrecode(y);
ga = ones(5000,10);

%vectorization method
%J_vector = sum(sum(-ya.*log(b3) - (ga-ya).*log(ga-b3))/m);

%using loop

J_loop = 0;k =0; i=0;

for i=1:m 
	for k=1:num_labels
		
			J_loop = J_loop + (-ya(i,k)*log(b3(i,k))-(1-ya(i,k))*(log(1-b3(i,k))));
		end
end

J_loop = J_loop/m;

J = J_loop;

%Part 2

%Theta1 = Theta1(:,2:end);
%Theta2 = Theta2(:,2:end);

Reg_Term = (sum(sum(Theta1(:,2:end).^2,[1,:]))+sum(sum(Theta2(:,2:end).^2,[1,:])))*lambda/(2*m);

J=J+Reg_Term;

% Part 3
Delta = 0; Delta1 =zeros(size(Theta1)); Delta2=zeros(size(Theta2));

for t=1:m

	a1 = X(t,:); % size 1x400
	a1 = [1 a1]; % 401 x1
	a2 = sigmoid(Theta1*a1'); % size 25 x 401 * 401 x 1 --> 25 x 1
	a2 = [1; a2]; % 26 x1
	%a22 = [1 b2];
	a3 = sigmoid(Theta2*a2); %10x1
	for k = 1:num_labels
		delta3(k) = a3(k) - ya(t,k); %size 1x10
	end
	%size(Theta2(:,2:end));
	delta2 = Theta2'*delta3'.*(a2.*(1-a2)); %26x1
	delta2 = delta2(2:end);%25x1
	Delta1 = Delta1 + delta2*a1; %10x26
	size(Delta1);
	Delta2 = Delta2 + delta3'*a2'; %
	size(Delta2);
end

%grad = [Delta1/m;Delta2/m]

%J =  sum(-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta)))/m

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

grad = [(Delta1/m)(:);(Delta2/m)(:)];


end
