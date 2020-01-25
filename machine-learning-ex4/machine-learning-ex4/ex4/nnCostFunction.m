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
         
         y2 = zeros(m,num_labels);
for h=1:m
  y2(h,y(h))=1;
endfor
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
a1 = [ones(m, 1) X];  %5000x401
a2 = [ones(m, 1) sigmoid(a1*Theta1')];
a3 = sigmoid(a2*Theta2');  %5000x10



theta1_r = Theta1(:,2:end);  % all but first column
theta2_r =  Theta2(:,2:end);
reg = (lambda/(2*m)) * (sum(sum(sum(theta1_r.*theta1_r)) + sum(sum(theta2_r.*theta2_r))));



J = sum(sum(-y2.*log(a3)-((1-y2).*log(1-a3))))/m + reg;




for t=1:m
a1_t = [1 X(t,:)];
z2_t = a1_t*Theta1';
a2_t= [1 sigmoid(z2_t)];
z3_t = a2_t*Theta2';
a3_t = sigmoid(z3_t);

d3_temp = (a3_t - y2(t,:));
d2_temp = (Theta2(:,2:end)'*d3_temp'.*sigmoidGradient(z2_t)')';

d2(t,1:size(d2_temp,2))=d2_temp;
d3(t,1:size(d3_temp,2))=d3_temp;
endfor
D1 = d2'*a1;
D2 = d3'*a2;

Theta1(:,1)=0;
Theta2(:,1)=0;
Theta1 = (lambda/m)*Theta1
Theta2 = (lambda/m)*Theta2



Theta1_grad = (D1/m)+Theta1;
Theta2_grad = (D2/m) + Theta2;
%D2 = d3*a2;
%J = d2;















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
