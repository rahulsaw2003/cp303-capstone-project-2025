function gradK = Num_Jacobian_K(fun_name, h0) 
    % Function to calculate the gradient (numerical derivative) of K(h) w.r.t. h
    % Inputs:
    %   fun_name: Name of the function (as string) for K(h)
    %   h0: Value at which the derivative is calculated (initial h value)
    
    % Step size (perturbation)
    eps = h0 / 10000; 
    
    % If eps becomes zero (for small h values), set a default small value
    if eps == 0
        eps = 1 / 10000;
    end
    
    % Compute K(h) for h0 + eps
    Hp = h0 + eps;  % Perturbed value of h
    fp = feval(fun_name, Hp);  % Function value at h0 + eps
    
    % Compute K(h) for h0 - eps
    Hn = h0 - eps;  % Perturbed value of h
    fn = feval(fun_name, Hn);  % Function value at h0 - eps
    
    % Gradient of K(h) (numerical derivative using finite difference)
    gradK = (fp - fn) / (2 * eps);
end
