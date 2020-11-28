% Fits the linear model
function obj = f_FitModel(obj,train_data,train_label)

    obj.W = zeros(obj.M, obj.K); % Initialize W
    obj.w = zeros(obj.K, 1); % Initialize w

    % For every class combination:
    for i = 1:obj.K
        % Indices corresponding to classes i & ~i:
        indices1 = find(train_label==obj.c(i));
        indices2 = find(train_label~=obj.c(i));

        % Batch indices:
        batch1_idx = f_batch(indices1,obj.b);
        batch2_idx = f_batch(indices2,obj.b);
        % For each batch, compute W and w. Then average.
        for m = 1:obj.b
            % Training data for mth batch:
            t_m = [train_data(batch1_idx(:,m),:); ...
                   train_data(batch2_idx(:,m),:)];
               
            % Labels for mth batch:
            sz1 = size(batch1_idx(:,m));
            sz2 = size(batch2_idx(:,m));
            l_m = [ones(sz1);-ones(sz2)];

%             gamma = 0.5; % Constant multiplying the 1-norm
            gamma = obj.gamma;
            % Computes hyperplane boundary
            [a_i,b_i] = SeparatingHyperplane2(t_m.',l_m,gamma);

            % Increment W and w:
            obj.W(:,i) = obj.W(:,i) + a_i;
            obj.w(i) = obj.w(i) + b_i;
        end
    end
    % Divide by obj.b to compute averages:
    obj.W = obj.W/obj.b;
    obj.w = obj.w/obj.b; 
            
end