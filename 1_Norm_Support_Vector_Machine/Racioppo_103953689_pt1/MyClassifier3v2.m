
classdef MyClassifier3v2 < handle
    
    properties (Access = public)
        K                     % Number of classes
        M                     % Number of features
        W                     % Hyperplanes vectors
        w                     % Hyperplane biases
        c                     % Combinations
        l                     % Labels
        b                     % Batches
        trm                   % Training mean
        p
        prob
        gamma
        counter
        % You may add any extra properties you would like
        
    end
        
    methods (Access = public)
        
        function obj = MyClassifier3v2(K,M)    % Class Constructor
            obj.K = K;
            obj.M = M;
            obj.W = [];
            obj.w = [];
            obj.c = [];
            obj.l = [];
            obj.trm = [];
            obj.b = 5; % Default: 5 batches
            obj.p = poissrnd(10, [1,M])/100;
            obj.gamma = [];
            obj.counter = [];
            
            % You may initialize other properties and add them here
        end
        
        
        function obj = train(obj,train_data,train_label)
            
            %%% THIS IS WHERE YOU SHOULD WRITE YOUR TRAINING FUNCTION
            %
            % The inputs to this function are:
            %
            % obj: a reference to the classifier object.
            % train_data: a matrix of dimesions N_train x M, where N_train
            % is the number of inputs used for training. Each row is an
            % input vector.
            % trainLabel: a vector of length N_train. Each element is the
            % label for the corresponding input column vector in trainData.
            %
            % Make sure that your code sets the classifier parameters after
            % training. For example, your code should include a line that
            % looks like "obj.W = a" and "obj.w = b" for some variables "a"
            % and "b".
            
            obj.trm = mean(train_data,1); % Training mean
            
            % Add noise to training data:
            train_data = f_AddNoise(obj,train_data,1);
            
            % Corrupt the data:
            train_data = f_Corrupt(obj,train_data);
            
            % Replace corrupted values with mean values:
            for i = 1:length(train_data)
                for j = 1:obj.M
                    if isnan(train_data(i,j))
                        train_data(i,j) = obj.trm(j);
                    end
                end
            end
            
            obj.l = unique(train_label); % Unique labels
            obj.c = 1:obj.K; % A list of 1:num_classes
            
            % Fit the linear model:
            obj.gamma = 0.5;
            % Line search on gamma:
%             obj = f_LineSearch1(obj,train_data,train_label);
            % Fit model:
            obj = f_FitModel(obj,train_data,train_label);
        end
        
        function s = f(obj, input)
            %%% THIS IS WHERE YOU SHOULD WRITE YOUR CLASSIFICATION FUNCTION
            %
            % The inputs of this function are:
            %
            % input: the input to the function f(*), equal to g(y) = W^T y + w 
            %
            % The outputs of this function are:
            %
            % s: this should be a scalar equal to the class estimated from
            % the corresponding input data point, equal to f(W^T y + w)

            sgn = sign(input); % 1 or -1, depending on side of hyperplane
            mask1 = [sgn==1, sgn==2]; % Mask, 2nd column = zeros
            % Only select first column:
            count = (obj.l)*0; % Initialize count (counts # of votes)
            L = length(obj.l);
            for i = 1:L
                mask2 = (obj.c==obj.l(i)).'; % Where pt class = current label
                vote = mask1.*mask2; % Vote=1 iff ith hyperplane picks pt
                count(i) = sum(sum(vote)); % Number of 1s in vote
            end
            
            max_count = max(count); % Maximum value(s) in count
            idx = find(count==max_count); % Index of max value(s)
            
            % If multiple labels tie as most chosen:
            if size(idx,1) ~= 1
                for j = idx'
                    mask2 = (obj.c==obj.l(j)).'; % Picks label j from obj.c
                    margin = abs((mask1.*mask2).*[input,input]); % Margin
                    count(j) = sum(sum(margin)); % Sum of margins
                end
                max_count = max(count); % Maximum margin size(s) in count
                idx = find(count == max_count); % Index of max value
            end
            
            % Update s:
            if size(idx,1) == 1
                s = obj.l(idx);
            else
                % Mode picks min index in case of tie:
                s = obj.l(mode(idx));
            end
        end
        
        function [test_results] = TestCorrupted1(obj,test_data,p)
            %%% THIS FUNCTION OUTPUTS ESTIMATED CLASSES FOR A DATA MATRIX
            %%% (DON'T EDIT THIS FUNCTION)
            %
            % The inputs of this function are:
            %
            % obj: a reference to the classifier object.
            % test_data: a matrix of dimesions N_test x M, where N_test
            % is the number of inputs used for training. Each row is an
            % input vector.
            %
            %
            % The outputs of this function are:
            %
            % test_results: this should be a vector of length N_test,
            % containing the estimations of the classes of all the N_test
            % inputs.
            
            obj.p = p;
            
            % Corrupt the data:
            test_data = f_Corrupt(obj,test_data);
            
            % Replace corrupted values with  mean values:
            obj.counter = zeros(1,obj.M);
            for i = 1:length(test_data)
                for j = 1:obj.M
                    if isnan(test_data(i,j))
                        test_data(i,j) = obj.trm(j);
                        obj.counter(j) = obj.counter(j) + 1;
                    end
                end
            end
            
            if (isempty(obj.W) || isempty(obj.w))
                error('Classifier is not trained yet.');
            end
            
            N_test = size(test_data, 1); % Number of elements in test data
            test_results = zeros(N_test, 1); % Initialize test_results
            
            for i = 1:N_test
                test_results(i) = obj.f(obj.W' * test_data(i,:)' + obj.w);
            end
        end
    end
end
