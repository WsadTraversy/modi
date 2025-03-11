linearization_points = [-0.4 0.1 0.8];

% pochodne
syms u
diff_y_stat = diff(b0/a0*(alfa1*u + alfa2*u^2 + alfa3*u^3 + alfa4*u^4));
diff_y_stat_1 = subs(diff_y_stat, linearization_points(1));
diff_y_stat_2 = subs(diff_y_stat, linearization_points(2));
diff_y_stat_3 = subs(diff_y_stat, linearization_points(3));
assume(u, 'clear')  

% charakterystyka statyczna
y_stat = @(u) b0/a0*(alfa1*u + alfa2*u^2 + alfa3*u^3 + alfa4*u^4);

% charakterystyki statyczne zlinearyzowane
y_stat_lin_1 = y_stat(linearization_points(1)) + ...
            diff_y_stat_1 * (u - linearization_points(1));
y_stat_lin_2 = y_stat(linearization_points(2)) + ...
            diff_y_stat_2 * (u - linearization_points(2));
y_stat_lin_3 = y_stat(linearization_points(3)) + ...
            diff_y_stat_3 * (u - linearization_points(3));

figure(1)
%fplot(y_stat_lin_1, [-1 1]);
%hold on
%fplot(y_stat_lin_2, [-1 1]);
%hold on
fplot(y_stat_lin_3, [-1 1]);
hold on
fplot(y_stat, [-1 1]);
hold off
xlabel('u');
ylabel('y');
axis([-1 1 -1 2.5]);
legend('y(u) zlinearyzowana', 'y(u) nieliniowa');
grid