% Load the data from the .mat file
load('simulation_data.mat', 'data_inputs', 'data_outputs');

% Debug: Check if data is loaded correctly
disp('Size of data_inputs:');
disp(size(data_inputs)); % Should be [number_of_samples, number_of_features]
disp('Size of data_outputs:');
disp(size(data_outputs)); % Should be [number_of_samples, number_of_nodes]

% Extract surface node outputs (assuming node 1 corresponds to the first column)
surface_node_outputs = data_outputs(:, 1);

% Debug: Check if surface node outputs are extracted correctly
disp('Size of surface_node_outputs:');
disp(size(surface_node_outputs)); % Should be [number_of_samples, 1]

% Calculate split index (70% training, 30% testing)
num_samples = size(data_inputs, 1); % Total number of samples

disp('num_samples');
disp(num_samples);

if num_samples < 10
    error('Dataset is too small for splitting. Use all data for training.');
else
    split_idx = round(0.7 * num_samples); % Index for splitting

    if split_idx >= num_samples
        error('Not enough data for testing. Ensure at least 30% of the data is available for testing.');
    end

    % Training set
    train_inputs = data_inputs(1:split_idx, :);
    train_outputs = surface_node_outputs(1:split_idx);

    % Testing set
    test_inputs = data_inputs(split_idx+1:end, :);
    test_outputs = surface_node_outputs(split_idx+1:end);

    % Debug: Verify sizes of training and testing sets
    disp('Size of train_inputs:');
    disp(size(train_inputs)); % Should be [70% of samples, number_of_features]
    disp('Size of test_inputs:');
    disp(size(test_inputs));  % Should be [30% of samples, number_of_features]
end

if isempty(test_inputs)
    error('Error: test_inputs is empty. Check your data splitting logic.');
end

% train_inputs

% Prepare input sequences for LSTM (convert rows to cells)
train_input_sequences = num2cell(train_inputs, 2); % Each row becomes a sequence
train_output_sequences = num2cell(train_outputs, 2); % Each row becomes a target sequence

test_input_sequences = num2cell(test_inputs, 2); % Each row becomes a sequence
test_output_sequences = num2cell(test_outputs, 2); % Each row becomes a target sequence

% Debug: Verify sizes of input sequences
disp('Size of train_input_sequences:');
disp(size(train_input_sequences)); % Should be [number_of_training_samples, 1]
disp('Size of test_input_sequences:');
disp(size(test_input_sequences));  % Should be [number_of_testing_samples, 1]

% Define LSTM network architecture
input_size = size(train_inputs, 2); % Number of features in input

layers = [
    sequenceInputLayer(input_size)               % Input layer
    lstmLayer(50, 'OutputMode', 'last')         % LSTM layer with output at last time step
    fullyConnectedLayer(1)                      % Fully connected layer for single output
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
if ~isempty(test_input_sequences)
    predicted_outputs = predict(net, test_input_sequences);

    % Convert predictions from cell array to matrix for comparison
    predicted_outputs = cell2mat(predicted_outputs);

    % Compare predicted vs actual outputs
    actual_outputs = cell2mat(test_output_sequences);

    % Plot results
    figure;
    plot(actual_outputs, 'r-', 'LineWidth', 2); hold on;
    plot(predicted_outputs, 'b--', 'LineWidth', 2);
    legend('Actual Outputs', 'Predicted Outputs');
    xlabel('Time Step');
    ylabel('Surface Node Output');
    title('LSTM Model Performance on Surface Node');
    grid on;

    % Calculate Mean Squared Error (MSE)
    mse_value = mean((actual_outputs - predicted_outputs).^2);
    disp(['Mean Squared Error: ', num2str(mse_value)]);
else
    warning('No test data available. Skipping testing phase.');
end
