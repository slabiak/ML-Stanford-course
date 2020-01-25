% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
theta2_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals),length(theta2_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
       for k = 1:length(theta2_vals)
	  t = [theta0_vals(i); theta1_vals(j); theta2_vals(k)];
	  J_vals(i,j,k) = computeCost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals,theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');ylabel('\theta_2');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals,theta2_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
% hold on;
% plot(theta(2),theta(3) 'rx', 'MarkerSize', 10, 'LineWidth', 2);