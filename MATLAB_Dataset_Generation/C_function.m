function C_val = C_function(h)
    % Function to compute c(h) based on the given equation
    global theta_s theta_r alpha n
    
    % Compute the individual terms
    term1 = (theta_s - theta_r) * alpha * n * (1 - 1/n);  % First part of the equation
    term2 = (-alpha * h)^(n - 1);                          % (-alpha * h)^(n-1)
    term3 = (1 + (-alpha * h)^n)^((1/n) - 2);              % [1 + (-alpha * h)^n]^(1/n-2)
    
    % Final c(h) value
    C_val = term1 * term2 * term3;
end