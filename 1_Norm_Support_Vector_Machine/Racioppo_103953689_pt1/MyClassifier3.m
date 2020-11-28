
classdef MyClassifier3 < handle
    
    properties (Access = public)
        K                     % Number of classes
        M                     % Number of features
        W                     % Hyperplanes vectors
        w                     % Hyperplane biases
        c                     % Combinations
        l                     % Labels
        b                     % Batches
        p
        all_train_data
        all_train_label
        gamma
        % You may add any extra properties you would like
        
    end
        
    methods (Access = public)
        
        function obj = MyClassifier3(K,M)    % Class Constructor
            obj.K = K;
            obj.M = M;
            obj.W = [];
            obj.w = [];
            obj.c = [];
            obj.l = [];
            obj.b = 5; % Default: 5 batches
            obj.p = poissrnd(10, [1,M])/100;
            obj.all_train_data = [];
            obj.all_train_label = [];
            obj.gamma=8;
            
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
            
            % Every combination of 2 of the obj.K classes:
            obj.all_train_data = train_data;
            obj.all_train_label = train_label;
            obj.l = unique(train_label);
            obj.c = combnk(obj.l,2);
            
            N = length(obj.c); % Number of combinations
            obj.W = zeros(obj.M, N); % Initialize W
            obj.w = zeros(N, 1); % Initialize w
            W_temp = zeros(size(obj.W,1),size(obj.W,2),obj.b);
            w_temp = zeros(1,obj.b);
            
            for i=1:length(obj.l)
                indices1 = find(train_label==obj.c(i,1));
                 temp = f_batch(indices1, 10);
                 if i==1
                     test__batch_idx=temp(:,1);
                 else
                    test__batch_idx=[test__batch_idx; temp(:,1)];
                 end
            end
            
            test_batch_data=train_data(test__batch_idx,:);
            test_batch_label=train_label(test__batch_idx);
            
            train_data(test__batch_idx,:)=[];
            train_label(test__batch_idx)=[];
            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            if length(obj.gamma)>1
                gamma=8;
                counter=0;
                max_eff=0;
                past_eff=zeros(1,2);
                batch1_idx = f_batch(1:length(train_label),10);
                batch_data = train_data(batch1_idx(:,1),:);
                batch_label = train_label(batch1_idx(:,1),:);
                for j=logspace(0.6,-0.2,10)
                    counter=counter+1;
                    for i = 1:N

                        indices1 = find(batch_label==obj.c(i,1));
                        indices2 = find(batch_label==obj.c(i,2));

                        t_m = [batch_data(indices1,:); ...
                               batch_data(indices2,:)];

                        sz1 = size(indices1);
                        sz2 = size(indices2);
                        l_m = [ones(sz1);-ones(sz2)];
                        [a_i,b_i] = SeparatingHyperplane2(t_m.',l_m,gamma);

                        obj.W(:,i) = a_i;
                        obj.w(i) = b_i;
                    end
                    t_m = test_batch_data;
                    l_m = test_batch_label;
                    temp_test_result = classify(obj,t_m);
                    temp_test_result= sum(temp_test_result==l_m)/length(l_m);
                    if counter ~= 1
                        if max_eff(1) < temp_test_result
                            max_eff = [temp_test_result, gamma];
                        end
                        if past_eff(1)>temp_test_result
                            dir_sw=1;
                        else
                            dir_sw=-1;
                        end

                        if past_eff(2)>gamma
                            past_eff(2)=gamma;
                            gamma=gamma+j*dir_sw;
                            if gamma<0, gamma=0; end
                        else
                            past_eff(2)=gamma;
                            gamma=gamma-j*dir_sw;
                            if gamma<0, gamma=0; end
                        end
                    else
                        max_eff = [temp_test_result, gamma];
                        past_eff(2)=gamma;
                        gamma=gamma-j;

                    end
                    past_eff(1)=temp_test_result;
                end
            end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % For every class combination:
            for i = 1:N
                % Indices corresponding to classes 1 and 2
                % in the ith iteration: 
                indices1 = find(train_label==obj.c(i,1));
                indices2 = find(train_label==obj.c(i,2));
                
%                 L1 = length(indices1);
%                 L2 = length(indices2);   
%                 
%                 % Nki = Number of elements in class i / Number of classes
%                 Nk1 = floor(L1/obj.K);
%                 Nk2 = floor(L2/obj.K);
%                 
%                 % Nki random indices for each of the 2 classes
%                 rand1 = datasample(indices1,Nk1,'Replace',false);
%                 rand2 = datasample(indices2,Nk2,'Replace',false);
%                 
%                 % Training data and labels for the 2 classes
%                 Train1 = train_data(rand1,:);
%                 Train2 = train_data(rand2,:);
%                 Label1 = train_label(rand1);
%                 Label2 = train_label(rand2);
%        
%                 % Vectors of training data and labels
%                 Train_i = [Train1; Train2]';
%                 Label_i = [Label1; Label2]';
%                 Label_i = Label_i';
%                 
%                 % Compute separating hyperplane for these classes
%                 [a_i,b_i] = SeparatingHyperplane(Train_i,Label_i);
%                 obj.W(:,i) = a_i;
%                 obj.w(i) = b_i;
                
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
                    
                    % Compute separating hyperplane:
                    [a_i,b_i] = SeparatingHyperplane(t_m.',l_m);
                  
                    % Increment W and w:
                    W_temp(:,i,m) = a_i;
                    w_temp(i,m) = b_i;
                    obj.W(:,i) = obj.W(:,i) + a_i;
                    obj.w(i) = obj.w(i) + b_i;
                end
            end
            
            % Divide by obj.b to compute averages:
            obj.W = obj.W/obj.b;
            obj.w = obj.w/obj.b;
            W_original=obj.W;
            w_original=obj.w;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            

            
%             % Batch testing data for mth batch:
%             t_m = [train_data(batch1_idx(:,obj.b),:); ...
%                 train_data(batch2_idx(:,obj.b),:)];
%             % Labels for mth batch:
%             sz1 = size(batch1_idx(:,obj.b));
%             sz2 = size(batch2_idx(:,obj.b));
%             l_m = [ones(sz1);-ones(sz2)];
            t_m = test_batch_data;
            l_m = test_batch_label;
            temp_test_result = classify(obj,t_m);
            temp_test_result= sum(temp_test_result==l_m)/length(l_m);
            comb_idx(1).value=[];
            max_eff=0;
            
            for i=1:obj.b
                comb_idx(i).value = combnk(1:obj.b,i);
            end
            for i=1:obj.b
                for j=1:size(comb_idx(i).value,1)
                    obj.W = zeros(obj.M, N); % Initialize W
                    obj.w = zeros(N, 1); % Initialize w
                    for m=1:size(comb_idx(i).value,2)
                        focus_idx=comb_idx(i).value(j,m);
                        obj.W = obj.W+W_temp(:,:,focus_idx);
                        obj.w = obj.w+w_temp(:,focus_idx);
                    end
                    obj.W = obj.W/size(comb_idx(i).value,2);
                    obj.w = obj.w/size(comb_idx(i).value,2);

                    temp_test_result = classify(obj,t_m);
                    temp_test_result= sum(temp_test_result==l_m)/length(l_m);
                    if max_eff < temp_test_result
                        max_eff = temp_test_result;
                        max_idx = [i, j, max_eff];
                    end
                end
            end
            
%             disp(max_eff)

            obj.W = zeros(obj.M, N); % Initialize W
            obj.w = zeros(N, 1); % Initialize w
            for m=1:size(comb_idx(max_idx(1)).value,2)
                focus_idx=comb_idx(max_idx(1)).value(max_idx(2),m);
                obj.W = obj.W+W_temp(:,:,focus_idx);
                obj.w = obj.w+w_temp(:,focus_idx);
            end
            obj.W = obj.W/size(comb_idx(max_idx(1)).value,2);
            obj.w = obj.w/size(comb_idx(max_idx(1)).value,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



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
            if sum(isnan(input))==obj.M
                s = mode(obj.all_train_label);
            elseif sum(isnan(input))>=1
                idx = find(1 == isnan(input)); % Index of max value
                temp_train_data=obj.all_train_data;
                temp_train_data(:,idx)=[];
                temp_input=input;
                temp_input(:,idx)=[];
%                 distance = sum(sqrt((sum(temp_train_data.^2,2)+ ...
%                     sum(temp_input.^2))' - ...
%                     2*bsxfun(@times,temp_input',temp_train_data')),1);
%                 distance=sqrt(sum(bsxfun(@minus,temp_input',temp_train_data') .^ 2,1));
                distance=sqrt(sum(abs(bsxfun(@minus,temp_input',temp_train_data')),1));
                min_val = min(distance); % min margin size(s) in count
                idx2 = find(distance == min_val); % Index of max value
%                 s=obj.all_train_label(idx2(1));
                input2=input;
                input2(idx)=obj.all_train_data(idx2(1),idx);
                s= obj.f(obj.W' * input2' + obj.w);
                
            else
                
                sgn = sign(input); % 1 or -1, depending on side of hyperplane
                mask1 = [sgn==1, sgn==-1]; % Parse sgn into binary vector
                count = (obj.l)*0; % Initialize count (counts # of votes)
                L = length(obj.l);
                for i = 1:L
                    mask2 = (obj.c == obj.l(i)); % Picks out label i from obj.c
                    vote = mask1.*mask2; % Vote=1 iff ith hyperplane picks pt
                    count(i) = sum(sum(vote)); % Number of 1s in vote
                end
                
                max_count = max(count); % Maximum value(s) in count
                idx = find(count == max_count); % Index of max value(s)
                
                % If multiple labels tie as most chosen:
                if size(idx,1) ~= 1
                    for j = idx'
                        mask2 = (obj.c==obj.l(j)); % Picks label j from obj.c
                        margin = abs((mask1.*mask2).*[input,input]); % Margin
                        count(j) = sum(sum(margin)); % Sum of margins
                    end
                    max_count = max(count); % Maximum margin size(s) in count
                    idx = find(count == max_count); % Index of max value
                    if size(idx,1) ~= 1
                        disp('help')
                        s = obj.l(idx(1));
                    else
                        s = obj.l(idx);
                    end
                else
                    s = obj.l(idx);
                end
            end
            
        end
        
        function [test_results] = classify(obj,test_data)
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
            
            if (isempty(obj.W) || isempty(obj.w))
                error('Classifier is not trained yet.');
            end
            
            N_test = size(test_data, 1); % Number of elements in test data
            test_results = zeros(N_test, 1); % Initialize test_results
            
            for i = 1:N_test
                if sum(isnan(test_data(i,:)'))>=1
                    test_results(i) = obj.f(test_data(i,:));
                else
                    test_results(i) = obj.f(obj.W' * test_data(i,:)' + obj.w);
                end
            end
        end
        
        function [test_results] = TestCorrupted1(obj,test_data,p)
            obj.p = p;
            test_data = f_Corrupt(obj,test_data);
            test_results = classify(obj,test_data);
        end
    end
end
