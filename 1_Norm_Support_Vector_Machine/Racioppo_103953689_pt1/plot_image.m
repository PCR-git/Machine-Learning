function plot_image(y)

plot(y(1:2:16), y(2:2:16), 'k');
hold on;
scatter(y(1), y(2), 'k', 'filled');
hold off;