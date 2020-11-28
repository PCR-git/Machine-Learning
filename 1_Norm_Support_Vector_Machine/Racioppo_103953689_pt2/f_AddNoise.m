
% Add Gaussian noise to training data:
function train_data = f_AddNoise(obj,train_data,scale)
    std = scale*(max(train_data,[],1)-min(train_data,[],1))/100; % Std
    % Draw from normal distribution:
    r = zeros(length(train_data),obj.M); % Initialize r
    for i = 1:obj.M
        % Compute Guassian noise for each data point
        r(:,i) = binornd(0,std(i),[length(train_data),1]);
    end
    
    % Remove values above min and max:
    minr = min(min(r));
    maxr = max(max(r));
    r = max(r,minr);
    r = min(r,maxr);
    
    train_data = train_data + r; % Add noise to data
end