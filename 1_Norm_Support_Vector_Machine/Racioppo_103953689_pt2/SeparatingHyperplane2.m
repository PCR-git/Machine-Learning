
% Computes a separating hyperplane using a linear program
% Inputs: p (data) & labels
% Outputs: Hyperplane slope (a) and intercept (b)
function [a,b] = SeparatingHyperplane2(p,labels,gamma)

% n = Dimension of data; m = Number of data points
[n, m] = size(p);

% Matrix in LP:
Ones = ones(m,1);

LH = -labels.*[p.', Ones];
I = -eye(size(labels,1));
A = [LH, I];

% Linear Program:
cvx_begin quiet    
    variables a(n) b u(m)
    minimize sum(u) + gamma*sum(abs(a))
    subject to
        A*[a;b;u] <= -1;
        u >= 0;
cvx_end

end
