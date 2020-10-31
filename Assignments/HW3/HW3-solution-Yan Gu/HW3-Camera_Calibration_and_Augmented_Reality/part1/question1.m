%% Draw the image points
% real world coordinate
real_coordinate = [2 2 2;
                   -2 2 2;
                   -2 2 -2;
                   2 2 -2;
                   2 -2 2;
                   -2 -2 2;
                   -2 -2 -2;
                   2 -2 -2];
figure("Name", "Real world coordinate")
%f1 = axes
% circle and black
fig1 = scatter3(real_coordinate(:, 1), real_coordinate(:, 2), real_coordinate(:, 3), 8, "black", "o");
axis equal;
axis([-3, 3, -3, 3, -3, 3]);
grid on;
saveas(fig1, './part1/1/1.jpg');

% pixel positions in the camera image
image_pixel = [422 323; 178 323; 118 483; 482 483;
               438 73; 162 73; 78 117; 522 117];
figure("Name", "Pixel positions in the camera image")
fig2 = scatter(image_pixel(:, 1), image_pixel(:, 2), 8, "black", "o");
axis equal;
axis([0, 600, 0, 600]);
grid on;
saveas(fig2, './part1/1/2.jpg');
clear figure;