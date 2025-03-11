clear all;
syms s x a0 a1 a2 b0 alfa1 alfa2 alfa3 alfa4
A = [-a2 1 0; -a1 0 1; -a0 0 0];
B = [0; 0; b0*(alfa1+2*alfa2*x + 3*alfa3*x^2 + 4*alfa4*x^3)];
C = [1 0 0];
D = 0;

G = C*inv(s*eye(3)-A)*B+D;
%disp(G)

