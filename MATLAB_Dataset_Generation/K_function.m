function K_val = K_function(h)
    % function to compute the value of k
    
    global Ks alpha n

    % Precompute common terms
    term1 = (-alpha * h)^n;            % (-alpha * h)^n
    term2 = 1 + term1;                 % 1 + (-alpha * h)^n
    term3 = term2^(-1 + 1/n);           % (1 + (-alpha * h)^n)^(-1 + 1/n)
    term4 = term3^(1/2);               % outer square root term
    
    % Compute the inner part for the second term
    term5 = term3^(n / (n - 1));       % ((1 + (-alpha * h)^n)^(-1 + 1/n))^(n / (n - 1))
    term6 = 1 - term5;                 % 1 - (previous term)
    term7 = term6^(1 - (1/n));         % (1 - (previous term))^(1 - 1/n)
    
    % The full second term is squared
    term8 = 1 - term7; 

    term9 = term8^2;

    % Final result K(h)
    K_val = Ks * term4 * term9;
    
end