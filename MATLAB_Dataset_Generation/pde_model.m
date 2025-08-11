function [dH_dt] = pde_model(H, I_val)
    
global alpha Kc ET0 Z_r delZ   P_t H1   

    % Number of nodes (h1 to h26)
    N = length(H);  % N = 24
    
    % Initialize the derivative vector
    dH_dt = zeros(N, 1);
    
    % Assume no precipitation
    P_t = 0;  

    % H(1) means H value at node2
    % H(24) means h value at node 25
    
    % for firstNode, h2
    term1_2 = (K_function(H(1)) / (C_function(H(1)) * delZ)) * ((H(2) - 2*H(1) + H1) / delZ);

    gradK_2 = Num_Jacobian_K(@K_function, H(1));  % Use function handle

    H_diff_term = (H(2) - H(1)) / delZ;

    term2_2 = (1 / C_function(H(1))) * gradK_2 * H_diff_term * (H_diff_term + 1);

    C0_2 = (alpha * Kc * ET0) / Z_r;

    dH_dt(1) = (term1_2 + term2_2 - C0_2/C_function(H(1)));

    
    % Loop through all nodes to set up the equations (skip boundaries at top and bottom)

    % for 3rd node to 24th node
    for n = 2:N-1  % 2 to 23
        
        % Compute the finite differences for the second derivative term
        term1 = (K_function(H(n)) / (C_function(H(n)) * delZ)) * ((H(n+1) - 2*H(n) + H(n-1)) / delZ);

        % Get Jacobian value for K derivative
        gradK = Num_Jacobian_K(@K_function, H(n));  % Use function handle

        H_diff_term = (H(n+1) - H(n)) / delZ;
        
        % Compute the finite difference for the first derivative term
        term2 = (1 / C_function(H(n))) * gradK * H_diff_term * (H_diff_term + 1);
        
        % % Small Gaussian noise
        % noise = randn * 1e-9;  % Adjust the scale to keep it minimal

        C_0 = (alpha * Kc * ET0) / Z_r;
        
        % The equation for dh_n/dt
         dH_dt(n) = term1 + term2 + - C_0/C_function(H(n));

    end

    % for 25th node

     term1_25 = (K_function(H(24)) / (C_function(H(24)) * delZ)) * ((H(24) - 2*H(24) + H(23)) / delZ);
    
     gradK_25 = Num_Jacobian_K(@K_function, H(24));  % Use function handle

     H_diff_term = (H(24) - H(24)) / delZ;
        
     term2_25 = (1 / C_function(H(24))) * gradK_25 * H_diff_term * (H_diff_term + 1);
        
     C0_25 = (alpha * Kc * ET0) / Z_r;


     dH_dt(24) = term1_25 + term2_25 - C0_25/C_function(H(24));

     % total 24 differential equations

     % dh1/dt corresponds to 2nd node => calculate h1 using nonlinear
     % equation from BC
     % dh24/dt corresponds to 25th node => calculate h26 from bottom BC
     % (h26 = h25)