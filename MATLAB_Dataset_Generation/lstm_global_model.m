% Load the data from the .mat file
load('simulation_data.mat', 'data_inputs', 'data_outputs');

% Extract surface node outputs (assuming node 1 corresponds to the first column)
surface_node_outputs = data_outputs(:, 1);

% Split data into training (70%) and testing (30%)
split_idx = round(0.7 * size(data_inputs, 1)); % Index for splitting

% training set
train_inputs = data_inputs(1:split_idx, :);
train_outputs = surface_node_outputs(1:split_idx);

% testing set
test_inputs = data_inputs(split_idx+1:end, :);
test_outputs = surface_node_outputs(split_idx+1:end);

% Prepare input sequences for LSTM (convert to cell arrays)
train_input_sequences = num2cell(train_inputs', 1); % Transpose for LSTM format
train_output_sequences = num2cell(train_outputs', 1);

test_input_sequences = num2cell(test_inputs', 1);
test_output_sequences = num2cell(test_outputs', 1);

% Define LSTM network architecture
input_size = size(train_inputs, 2); % Number of features in input
output_size = size(train_outputs, 2); % Number of features in output

layers = [
    sequenceInputLayer(input_size)               % Input layer
    lstmLayer(50, 'OutputMode', 'sequence')     % LSTM layer with 50 hidden units
    fullyConnectedLayer(output_size)            % Fully connected layer for output
    regressionLayer                             % Regression layer for continuous outputs
];

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.01, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the LSTM model
net = trainNetwork(train_input_sequences, train_output_sequences, layers, options);

% Test the trained model on test data
predicted_sequences = predict(net, test_input_sequences);

% Convert cell arrays back to matrices for comparison
predicted_outputs = cell2mat(predicted_sequences');
actual_outputs = cell2mat(test_output_sequences');

% Plot predicted vs actual outputs for comparison
figure;
plot(actual_outputs(:, 1), 'r-', 'LineWidth', 2); hold on;
plot(predicted_outputs(:, 1), 'b--', 'LineWidth', 2);
legend('Actual Outputs', 'Predicted Outputs');
xlabel('Time Step');
ylabel('Soil Moisture / Capillary Potential');
title('LSTM Model Performance');
grid on;

% Calculate Mean Squared Error (MSE)
mse_value = mean((actual_outputs - predicted_outputs).^2);
disp(['Mean Squared Error: ', num2str(mse_value)]);