%% Part III: Augmented Reality 101
% If run open as Live Script, we can remove all pause 
%% Prerequisite from Part II
 
% Extract the four corners of the image
images = {'images2.png', 'images9.png', 'images12.png', 'images20.png'};

% bottom left, bottom right, top right, top left
real_corners = [0 270 270 0;
                0 0 210 210;
                1 1 1 1;];       
homography_list = [];

for i = 1:4
    file_name = char(images(1, i));
    image = imread(file_name);
    % show figure
    figure("Name", file_name)
    imshow(image);
    
    [corners_x, corners_y] = ginput(4);
    corner_matrix = [corners_x'; corners_y'; ones(size(corners_x))';];
    H = homography2d(real_corners, corner_matrix);
    homography_list = [homography_list; H / H(end, end)];
end

homo_list =[];
R_list ={};
t_list ={};

for i = 1:4
	file_name = char(images(1,i));
	H = homography_list(i * 3 - 2: i * 3, :);
    
	% 1: compute approx location
	real_corners = [];
	pixel_i = 0;
	for j = 1:10
		pixel_j = 0;
		for k = 1:8
			real_corners = [real_corners; pixel_i, pixel_j];
			pixel_j = pixel_j + 30;
		end
		pixel_i = pixel_i + 30;
    end
    
	real_corners = [real_corners'; ones(size(real_corners, 1), 1)';];
	p_approx = H * real_corners;
	p_approx = p_approx ./ repmat(p_approx(3, :), size(p_approx, 1), 1);
    %{
	figure("Name", 'Figure 1 : Projected grid corners')
	image = imread(file_name);
	imshow(image);
	hold on
	title('Figure 1 : Projected grid corners')
    
	plot(p_approx(1, :), p_approx(2, :), 'or'); % circle and red
	pause(0.8)
    %}
    
	% 2: detect Harris corners
    sigma = 2;
    thresh = 500;
    radius = 2;
	[cim, r, c, rsubp, csubp] = harris(rgb2gray(image), sigma, thresh, radius, 0);
	harris_corner=[csubp, rsubp];
    %{
	figure("Name", 'Figure 2 : Harris corners')
	image=imread(file_name);
	imshow(image);
	hold on
	title('Figure 2 : Harris corners')
    
	plot(harris_corner(:, 1), harris_corner(:, 2), 'dy'); % rhombus and yellow
	pause(0.8)
    %}
    
	% 3: compute closest Harris corner
	for j = 1:size(p_approx, 2)
		n = dist2(harris_corner, p_approx(1: 2, j)');
		[min_val, row_index] = min(n);
		row_list(j) = row_index;
    end
    
	p_correct = harris_corner(row_list(:), :);
    %{
	figure("Name", 'Figure 3 : grid points')
	image = imread(file_name);
	imshow(image);
	hold on
	title('Figure 3 : grid points')
    
	plot(p_correct(:, 1), p_correct(:, 2), '+m'); % cross and pinkish red
	pause(0.8)
    %}
    
	% 4: Print new H
	p_correct = [p_correct'; ones(1, size(p_correct, 1));];
    H = homography2d(real_corners, p_correct);
    homo_list = [homo_list; H / H(end,end)];
    
	% 6: error between points
    %{
	image_corners = H * real_corners;
	image_corners = image_corners ./ repmat(image_corners(3, :),size(p_approx, 1), 1);
	err_reprojection = sqrt(sum(sum((image_corners - p_correct).^2))) / size(image_corners, 2);
    %}
end

% 5: estimate K and R, t
V_prime = [];
for i = 1:4
	H = homo_list(i * 3 - 2: i * 3, :);
	V_prime = [V_prime; get_v(H)];
end

[V_U, V_Sigma, V_V] = svd(V_prime);
b = V_V(:, end);
B = [b(1) b(2) b(4);
     b(2) b(3) b(5);
     b(4) b(5) b(6);];

v_0 = (B(1, 2) * B(1, 3) - B(1, 1) * B(2, 3)) / (B(1, 1) * B(2, 2) - B(1, 2)^2);
lambda = B(3, 3) - (B(1, 3)^2 + v_0 * (B(1, 2) * B(1, 3) - B(1, 1) * B(2, 3))) / B(1, 1);
alpha = sqrt(lambda / B(1, 1));
beta = sqrt(lambda * B(1, 1) / (B(1, 1) * B(2, 2) - B(1, 2)^2));
gamma = -B(1, 2) * (alpha^2) * beta / lambda;
u_0 = gamma * v_0 / alpha - B(1, 3) * (alpha^2) / lambda;
% Print K
K = [alpha gamma u_0;
     0 beta v_0;
     0 0 1;]
 
for i = 1:4
	file_name = char(images(1, i))
	H = homo_list(i * 3 - 2: i * 3, :)
	lambda = 1 / sqrt(sum((inv(K) * H(:, 2)).^2));
	r_1 = lambda * inv(K) * H(:, 1);
	r_2 = lambda * inv(K) * H(:, 2);
	r_3 = cross(r_1, r_2);
    % Print R
    R = [r_1 r_2 r_3]
	R_list{i} = R;
    % Print t
	t = lambda * inv(K) * H(:, 3)
	t_list{i} = t;
end

%% Augmenting an Image
% Because last 4 digits of my RUID is 1028, so I used picture 1.gif
% important to use the right path
% I don't know why it can't properly read the *.gif and *.jpg using imread
% When imread the gif and jpg files, the alpha = []
% So, I transfer the 1.gif to the 1.png using Photoshop
[clip_image space alpha] = imread('./part3/clipart/1.png');
[row, col, space] = size(clip_image);

% Resize test to choose a good scaler
scalar = min(210 / col, 130 / row);

for i = 1:4
	file_name = char(images(1, i))
	H = homo_list(i * 3 - 2: i * 3, :);
	image = imread(file_name);
	
	for j = 1:row
		for k = 1:col
			if alpha(row + 1 - j, k) ~= 0
				p = H * [k * scalar; j * scalar; 1;];
				p = round(p / p(end, end));
				image(p(2), p(1), :) = clip_image(row + 1 - j, k, :);
			end
		end
    end
    
	figure("Name", file_name)
	imshow(image)
    output = sprintf('%s%d%s%s', './part3/AR_Image/', i, '_', file_name)
    imwrite(image, output);
	% pause(0.8)
end
figure clear

%% Augment Reality Object
% Put object in the middle of the picture
% the object is 60 px * 60 px * 60 px
obj_points = [120 120 0;
			  180 120 0;
			  180 180 0;
			  120 180 0;
			  120 120 60;
			  180 120 60;
			  180 180 60;
			  120 180 60;];
          
for i = 1:4
	file_name = char(images(1, i))
	image = imread(file_name);
	projection = K * [R_list{i} t_list{i}];
	f1 = figure("Name", file_name)
	imshow(image)
	hold on
    
    obj_pixels = zeros(size(obj_points, 1), 2);
    for j = 1:size(obj_points, 1)
		p = projection * [obj_points(j, 1); obj_points(j, 2); obj_points(j, 3); 1;];
		p = round(p / p(end, end));
        obj_pixels(j, :) = p(1: 2)';
    end
    
    % bottom plane edges
    plot([obj_pixels(1, 1) obj_pixels(2, 1)], [obj_pixels(1, 2) obj_pixels(2, 2)], 'r', 'LineWidth', 2)
    plot([obj_pixels(2, 1) obj_pixels(3, 1)], [obj_pixels(2, 2) obj_pixels(3, 2)], 'r', 'LineWidth', 2)
    plot([obj_pixels(3, 1) obj_pixels(4, 1)], [obj_pixels(3, 2) obj_pixels(4, 2)], 'r', 'LineWidth', 2)
    plot([obj_pixels(4, 1) obj_pixels(1, 1)], [obj_pixels(4, 2) obj_pixels(1, 2)], 'r', 'LineWidth', 2)
    % color the bottom plane
    % patch(X, Y, color)
    p1 = patch([obj_pixels(1, 1) obj_pixels(2, 1) obj_pixels(3, 1) obj_pixels(4, 1)], ...
               [obj_pixels(1, 2) obj_pixels(2, 2) obj_pixels(3, 2) obj_pixels(4, 2)], ...
               'p');
    p1.FaceVertexAlphaData = 0.1; % Set constant transparency
    p1.FaceAlpha = 'flat' ; 

    % top plane edges
    plot([obj_pixels(5, 1) obj_pixels(6, 1)], [obj_pixels(5, 2) obj_pixels(6, 2)], 'r', 'LineWidth', 2)
    plot([obj_pixels(6, 1) obj_pixels(7, 1)], [obj_pixels(6, 2) obj_pixels(7, 2)], 'r', 'LineWidth', 2)
    plot([obj_pixels(7, 1) obj_pixels(8, 1)], [obj_pixels(7, 2) obj_pixels(8, 2)], 'r', 'LineWidth', 2)
    plot([obj_pixels(8, 1) obj_pixels(5, 1)], [obj_pixels(8, 2) obj_pixels(5, 2)], 'r', 'LineWidth', 2)

    % side edges (height)
    plot([obj_pixels(1, 1) obj_pixels(5, 1)], [obj_pixels(1, 2) obj_pixels(5, 2)], 'r', 'LineWidth', 2)
    plot([obj_pixels(2, 1) obj_pixels(6, 1)], [obj_pixels(2, 2) obj_pixels(6, 2)], 'r', 'LineWidth', 2)
    plot([obj_pixels(3, 1) obj_pixels(7, 1)], [obj_pixels(3, 2) obj_pixels(7, 2)], 'r', 'LineWidth', 2)
    plot([obj_pixels(4, 1) obj_pixels(8, 1)], [obj_pixels(4, 2) obj_pixels(8, 2)], 'r', 'LineWidth', 2)
    
    output = sprintf('%s%d%s%s', './part3/AR_Object/', i, '_', file_name)
    saveas(f1, output);
    pause(1)
end
figure clear

%% Extra Credit

%% 1: augment a general mesh from a 3D file.
% *.stl can be opened using 3D printer or other Windows 10 default
% software.
% I choose stl file because we can easily get face-vertex vectors,
% and change them to xyz coordinates.
% Then use patch to draw a 3D model on our images.

% function stlread.m is downloaded from 
% https://www.mathworks.com/matlabcentral/fileexchange/6678-stlread
% draw big ben
% [x, y, z, c] = stlread('big_ben.stl');
% draw opera de sydney
% [x, y, z, c] = stlread('opera_de_sydney.stl');
% draw lamborghini
[x, y, z, c] = stlread('car_lamborghini.stl');

% show top view of the origin 3D model
patch(x, y, z, c)

% m = 3, 3 points can draw a plane
% n is the number of triangular planes
[m, n] = size(x)
% size(c)

% re-arrange the x, y, z coordinates from (3, n) to (1, 3 * n)
x_coor = [];
y_coor = [];
z_coor = [];
for i = 1:n
   x_coor = [x_coor, x(1, i), x(2, i), x(3, i)];
   y_coor = [y_coor, y(1, i), y(2, i), y(3, i)];
   z_coor = [z_coor, z(1, i), z(2, i), z(3, i)];
end

% re-scaler the coordinate
% for original -inf < x, y, z < +inf, change them to 0 < x, y, z < size(images)
% such that we can have model on the images (640 * 480).
% choose the max(size_x, size_y, size_z) <= size(images) / 4.
% get scale
scal = 160 / max([
                  (max(x_coor(1, :)) - min(x_coor(1, :))),...
                  (max(y_coor(1, :)) - min(y_coor(1, :))),...
                  (max(z_coor(1, :)) - min(z_coor(1, :)))
                 ]);
% change xyz to 0 < x, y, z < size(images) by translation and re-scaler
x_coor = (x_coor + (min(x_coor(1, :)) * (-1))) * scal;
y_coor = (y_coor + (min(y_coor(1, :)) * (-1))) * scal;
z_coor = (z_coor + (min(z_coor(1, :)) * (-1))) * scal;

%{
verify
min(x_coor(:))
min(y_coor(:))
min(z_coor(:))
size(x_coor)
size(y_coor)
size(z_coor)
%}

obj_points = [x_coor', y_coor', z_coor']; % matrix(3 * n, 3)
% size(obj_points)

% the same as in part III we did before
for i = 1:4
	file_name = char(images(1, i))
	image = imread(file_name);
	projection = K * [R_list{i} t_list{i}];
	f1 = figure("Name", file_name)
	imshow(image)
	hold on
    
    obj_pixels = zeros(size(obj_points, 1), 2);
    for j = 1:size(obj_points, 1)
		p = projection * [obj_points(j, 1); obj_points(j, 2); obj_points(j, 3); 1;];
		p = round(p / p(end, end));
        obj_pixels(j, :) = p(1: 2)';
    end
    
    % divide the vertics in groups of three and use those groups in the
    % function patch to draw the surface planes of the 3D object
    x_group = [];
    y_group = [];
    for ig = 1:n
        x_group = [x_group,...
                   [obj_pixels(3 * ig - 2, 1); obj_pixels(3 * ig - 1, 1); obj_pixels(3 * ig, 1)]
                  ]; % matrix(3, n)
        y_group = [y_group,...
                   [obj_pixels(3 * ig - 2, 2); obj_pixels(3 * ig - 1, 2); obj_pixels(3 * ig, 2)]
                  ]; % matrix(3, n)
    end
    % color the surface
    % patch(X, Y, color)
    % size(x_group)
    % size(y_group)
    p1 = patch(x_group, y_group, c);
    p1.FaceVertexAlphaData = 0.1; % Set constant transparency
    p1.FaceAlpha = 'flat' ;
    
    % output = sprintf('%s%d%s%s', './part3/extra1/', i, 'big_ben_', file_name)
    % output = sprintf('%s%d%s%s', './part3/extra1/', i, 'opera_de_sydney_', file_name)
    output = sprintf('%s%d%s%s', './part3/extra1/', i, 'car_lamborghini', file_name)
    saveas(f1, output);
    pause(1)
end
figure clear

%% 2: estimate from only two images
% from the chapter given
% If n = 2, we can impose the skewless constraint = 0, i.e., [0, 1, 0, 0, 0, 0]b = 0
% That means we assume the pixels construct a rectanguler
extra_credit(homo_list, images)