%% Print matrix P
% real world coordinate
real_coordinate = [2 2 2;
                   -2 2 2;
                   -2 2 -2;
                   2 2 -2;
                   2 -2 2;
                   -2 -2 2;
                   -2 -2 -2;
                   2 -2 -2];
% pixel positions in the camera image
image_pixel = [422 323; 178 323; 118 483; 482 483;
               438 73; 162 73; 78 117; 522 117];
           
[x, y] = size(real_coordinate);
P = zeros(2 * x, 3 * (y + 1));

for i = 1:x
    % return 2 rows.
    P(2 * i - 1: 2 * i, :) = cal_matrix_P(real_coordinate(i, :), image_pixel(i, :));
end
% print P
% disp("P=");
% disp(P)
P
% P;