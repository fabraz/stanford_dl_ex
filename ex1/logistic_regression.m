function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  y_hat = [];

  for index = 1:m
      yi_hat = 0;
      for theta_index = 1:size(theta)
          yi_hat = yi_hat + theta(theta_index)*X(theta_index,index);
      end
      y_hat = [y_hat;sigmoid(yi_hat)];
  end

  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.

  for index = 1:m
      f = f + (y(index)*log(y_hat(index)) + (1-y(index))*log(1-y_hat(index))); 
  end
  
  f = -1 * f;

  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %

  for theta_index = 1:size(theta)
      for index = 1:m
          g(theta_index) = g(theta_index) + X(theta_index,index)*(h(index)-y(index));
      end
  end
