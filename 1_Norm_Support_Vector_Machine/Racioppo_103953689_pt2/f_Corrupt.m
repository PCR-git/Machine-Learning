
% Corrupts the data:
function test_data_c = f_Corrupt(obj,test_data)

    Ones = ones(length(test_data),obj.M); % Ones matrix
    P = Ones.*obj.p; % Broadcast to matrix
    r = binornd(1,1-P); % Draw from binomial distribution
    r(r == 0) = NaN; % Set 0s in r to NaNs
    test_data_c = test_data.*r; % Element-wise multiply test_data

end