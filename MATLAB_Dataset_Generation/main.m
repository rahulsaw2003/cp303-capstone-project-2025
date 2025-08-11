% Main code file for MPC
clear all
clc

SetGraphics

% Define Constants
global delZ theta_s theta_r alpha n Ks ET0 Z_r Kc H1 I_val H2  P_t  

alpha = 7.5;
Kc = 0.88;
ET0 = 3.10e-08;
Z_r = -0.13;
Ks = 1.23*10^-5;
theta_s = 0.41;
theta_r = 0.538;
n = 2;
delZ = 0.0192;

  P_t = 0;  

n_nodes = 26; % Number of nodes (depth layers)
nodes_of_interest = [1];

% Define the number of additional irrigation steps
num_steps = 65 ; % Increase the number of steps significantly

% Define the range for irrigation rates (in m/s)
irrigation_min = 10^-6 * 0.1; % Minimum irrigation rate
irrigation_max = 10^-6 * 10; % Maximum irrigation rate

% Generate random irrigation values within the range
irrigation_values = irrigation_min + (irrigation_max - irrigation_min) * rand(1, num_steps);

% Define step intervals for each irrigation value (random durations)
step_intervals = randi([8, 12], 1, num_steps); % Random durations between 2 and 10 time units

% Total simulation time
total_steps = sum(step_intervals)
signal_time = 0:total_steps-1;

% Generate the irrigation profile
irrigation_profile = gen_irrigation_signal(signal_time, irrigation_values, step_intervals);
time_interval = total_steps;

% -------------old irrigation signal----------------------------

% Define initial conditions: Guess for h1, h2, ..., h26
H_intial  = 10^-5*[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.1]';
% stores the initial values from node2 to node25 (node1 and node26 initial
% values are calculated using these values with the help of boundary
% conditions
time_span = [0 0.1]; % Time range for the simulation (start, end)

% Define step intervals and irrigation values
% irrigation_values = 10^-6*[0.2, 2, 4, 1, 3, 7, 5, 2, 7, 2.5, 2, 7, 6, 8.6, 4, 6, 1, 4, 2, 5, 3, 6, 2, 8.5, 1, 3, 1.4, 0.9, 4.8, 7.3, 3.2, 5, 1, 6, 3, 5, 1];  % Irrigation rate values in m/s (scaled)
% step_intervals = [2, 5, 7, 10, 4, 8, 6, 3, 5, 4, 7, 6, 7, 5, 8, 10, 4, 3, 6, 5, 9, 4, 3, 5, 4, 4, 2, 5, 4, 3, 5, 7, 3, 8, 3, 6, 5, 4];  % Duration of each irrigation rate level
% 
% total_steps = sum(step_intervals); 
% signal_time = 0:total_steps-1;

% Generate irrigation profile at time t
 % irrigation_profile = gen_irrigation_signal(signal_time, irrigation_values, step_intervals);
 % time_interval = sum(step_intervals);

 % -------------old irrigation signal----------------------------

 H1 = 0.00005;

% Solve the system of differential equations using ode45

options = optimoptions('fsolve','Display','iter','StepTolerance', 10^-6);

% Initialize matrix to store head at all nodes (1 to 26)
H_total = zeros(time_interval, n_nodes);  % rows = time steps, cols = nodes

for i = 1:time_interval

    I_val = irrigation_profile(1, i);
    
    H2 = H_intial(1);
    H1 = fsolve('algebric_equation', H1);

    [t, H_vals] = ode45(@(t, H) pde_model(H, I_val), time_span, H_intial);

    % Extract the final heads from the ode solution (nodes 2 to 25)
    H_current = H_vals(end, :);      % H2 to H25
    H26 = H_current(end);            % H26 = H25
    H_full = [H1, H_current, H26];   % Node 1 to Node 26

    % Store into H_total
    H_total(i, :) = H_full;

    % For compatibility with next iteration
    H_intial = H_current';           % H2 to H25
    H2 = H_intial(1);
    H1 = fsolve('algebric_equation', H1);

    fprintf('Calculated H_total for i = %d\n', i);
end

for j = 1:length(nodes_of_interest)
    node_idx = nodes_of_interest(j);
    soil_moisture_profile(:, j) = calc_soil_moisture(H_total(:, node_idx));
end


% Loop through each node of interest and create a new figure for each
for j = 1:length(nodes_of_interest)
    node_idx = nodes_of_interest(j);

    figure;

    % Subplot 1: Soil Moisture for the current node
    subplot(2, 1, 1);
    plot(1:time_interval, soil_moisture_profile(:, j), 'r', 'LineWidth', 2);  
    xlabel('Time Interval');
    ylabel('Soil Moisture');
    title(['Soil Moisture at Node ', num2str(node_idx)]);
    grid on;

    subplot(2, 1, 2);
    plot(signal_time, irrigation_profile, 'k', 'LineWidth', 1.5);  
    xlabel('Time Interval');
    ylabel('Irrigation Rate (m/s)');
    title('Irrigation Profile, I(t)');
    set(gca, 'YScale', 'linear');
    grid on;
end

% Example input and output data (replace with your actual data)
data_inputs = irrigation_profile';          % Replace with your irrigation profile
data_outputs = soil_moisture_profile;      % Replace with your soil moisture profile

% Create a table with column headers
data_table = table(data_inputs, data_outputs, ...
    'VariableNames', {'Irrigation', 'SoilMoisture'});

% Write table to CSV file
writetable(data_table, 'simulation_data.csv');

disp('Data saved successfully to "simulation_data.csv"');

disp('Size of soil_moisture_profile:');
disp(size(soil_moisture_profile));

% Save the data to a .mat file
save('simulation_data.mat', 'data_inputs', 'data_outputs');

disp('Data saved successfully to "simulation_data.mat"');