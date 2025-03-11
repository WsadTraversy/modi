figure;
plot(out.liniowy);
hold on;
plot(out.nieliniowy);
grid on;
xlabel('t[s]');
ylabel('y');
title('')
axis([0 150 -0.2 4]);
legend('Model zlinearyzowany', 'Model nieliniowy', 'Location', 'southeast');
