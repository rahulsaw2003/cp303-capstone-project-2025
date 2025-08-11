function irrigation_rate = gen_irrigation_signal(time, irrigation_values, step_intervals)
    % This function generates the irrigation rate signal based on the provided time,
    % irrigation rate values, and step intervals.

    % Initialize the irrigation rate signal
    irrigation_rate = zeros(size(time)); 
    
    index = 1;
    
    % Loop to generate step signal for irrigation rate
    for i = 1:length(irrigation_values)
        
        % Update the irrigation rate at the specified time intervals
        irrigation_rate(index:index+step_intervals(i)-1) = irrigation_values(i);

        % Move the index forward by the current step interval
        index = index + step_intervals(i);
    end
end
