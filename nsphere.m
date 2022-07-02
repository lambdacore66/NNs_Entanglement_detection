function M = nsphere(n, number)

%---------------------------------------------------------------------
%|                                                                   |
%|  Simple method to generate unbiased random points on a n-sphere   |
%|                                                                   |
%---------------------------------------------------------------------
%
% Gives a matrix 'M' that contains 'number' random columns vectors of the
% unit 'n'-sphere.
%
%                    Antonio de la M. Sojo López

% Generate 'number' random vectors with 'n+1' random normally distributed 
% coordinates N(0,1)
M = randn([n+1, number]);
% Normalize
for i = 1:number
    M(:,i) = M(:,i) / norm( M(:,i) );
end
end