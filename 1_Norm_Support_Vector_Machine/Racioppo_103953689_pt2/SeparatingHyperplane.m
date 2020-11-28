
% Computes a separating hyperplane using a linear program
% Inputs: p (data) & labels
% Outputs: Hyperplane slope (a) and intercept (b)
function [a,b] = SeparatingHyperplane(p,labels)

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
%     minimize sum(u)
    minimize sum(u)
    subject to
        A*[a;b;u] <= -1;
        u >= 0;
cvx_end

end
