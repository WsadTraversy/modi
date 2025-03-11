fplot(@(u) b0/a0*(alfa1*u + alfa2*u^2 + alfa3*u^3 + alfa4*u^4), [-1 1]);
%title('Charakterystyka statyczna')
xlabel('u');
ylabel('y');
grid;