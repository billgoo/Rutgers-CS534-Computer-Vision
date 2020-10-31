% Q2: Function for calculating matrix P
function [P] = cal_matrix_P(real_coordinate, image_pixel)
    %{
        real_coordinate = [2 2 2;
                           -2 2 2;
                           -2 2 -2;
                           2 2 -2;
                           2 -2 2;
                           -2 -2 2;
                           -2 -2 -2;
                           2 -2 -2];
        image_pixel = [422 323; 178 323; 118 483; 482 483;
                       438 73; 162 73; 78 117; 522 117];
    %}
    [x, y] = size(real_coordinate); % [1, 3]
    real_coordinate = [real_coordinate 1];
    P = zeros(2 * x, 3 * (y + 1)); % [2, 12]
    %{
        u_i = pixel_i_x
        v_i = pixel_i_y
        P = [
                P_i_T 0_T -u_i*P_i_T
                0_T P_i_T -v_i*P_i_T
            ]
    %}
    %{
    for i = 1:x
        P(2 * i - 1,:) = [real_coordinate(i, :) Comp_zeros -image_pixel(i, 1)*real_coordinate(i, :)];
        P(2 * i,:) = [Comp_zeros real_coordinate(i, :) -image_pixel(i, 2)*real_coordinate(i, :)];
    end
    %}
    P(1, :) = [real_coordinate zeros(1, (y + 1)) -image_pixel(1, 1)*real_coordinate];
    P(2, :) = [zeros(1, (y + 1)) real_coordinate -image_pixel(1, 2)*real_coordinate];
