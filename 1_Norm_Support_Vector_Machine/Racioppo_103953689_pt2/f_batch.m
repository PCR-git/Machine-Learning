
% Splits the training data into batches
function batch = f_batch(index_v,Nb)

    Li = length(index_v); % Number of indices
    L = floor(Li/Nb); % Number of indices per batch
    rand_v = index_v(randperm(Li)); % Shuffle indices
    batch = zeros(L,Nb); % Initialize batch variable

    % Split the shuffled indices into Nb sets
    for i = 1:Nb
        start = (i-1)*L + 1; % First element of ith set
        stop = i*L; % Last element of ith set
        batch(:,i) = rand_v(start:stop); % Batches
    end

end