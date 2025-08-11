function theta_h_value = calc_soil_moisture(h)
    % Computes soil moisture content θ(h) given capillary potential h
    
    global theta_s theta_r alpha n

    h
    
    % Compute the moisture content
    term1 = (theta_s - theta_r) .* (1 ./ (1 + (-alpha .* h).^n)).^(1 - (1/n));
    term2 = theta_r;

    % Small Gaussian noise for each element in h
    % noise = randn(size(h)) * 1e-10;  % Generate noise of the same size as h

    % Add noise while ensuring moisture stays within physical limits [theta_r, theta_s]
    theta_h_value = term1 + term2;
end
