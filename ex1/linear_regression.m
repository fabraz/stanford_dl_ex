function [f,g] = linear_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  
  m=size(X,2);
  n=size(X,1);

  f=0;
  g=zeros(size(theta));

  y_hat = [];

  for index = 1:m
      yi_hat = 0;
      for theta_index = 1:size(theta)
          yi_hat = yi_hat + theta(theta_index)*X(theta_index,index);
      end
      y_hat = [y_hat;yi_hat];
  end


  %
  % TODO:  Compute the linear regression objective by looping over the examples in X.
  %        Store the objective function value in 'f'.

  for index = 1:m
      f = f + (y_hat(index) - y(index)).^2;
  end

  f = f/2;


  %
  % TODO:  Compute the gradient of the objective with respect to theta by looping over
  %        the examples in X and adding up the gradient for each example.  Store the
  %        computed gradient in 'g'.

  for theta_index = 1:size(theta)
      for index = 1:m
          g(theta_index) = g(theta_index) + X(theta_index,index)*(y_hat(index)-y(index));
      end
  end

