%% Part II: Camera Calibration using 2D calibration object

%% Corner Extraction and Homography computation 
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

% Print H
for i = 1:4
	file_name = char(images(1, i))
	H = homography_list(i * 3 - 2: i * 3, :)
end

%% Computing the Intrinsic and Extrinsic parameters
% Compute matrix B
V=[];
for i = 1:4
	H = homography_list(i * 3 - 2: i * 3, :);
	V = [V; get_v(H)];
end
[V_U, V_Sigma, V_V] = svd(V);

b = V_V(:, end);
% Print matrix B
B = [b(1) b(2) b(4);
     b(2) b(3) b(5);
     b(4) b(5) b(6);]

% Compute intrinsic parameters
v_0 = (B(1, 2) * B(1, 3) - B(1, 1) * B(2, 3)) / (B(1, 1) * B(2, 2) - B(1, 2)^2);
lambda = B(3, 3) - (B(1, 3)^2 + v_0 * (B(1, 2) * B(1, 3) - B(1, 1) * B(2, 3))) / B(1, 1);
alpha = sqrt(lambda / B(1, 1));
beta = sqrt(lambda * B(1, 1) / (B(1, 1) * B(2, 2) - B(1, 2)^2));
gamma = -B(1, 2) * (alpha^2) * beta / lambda;
u_0 = gamma * v_0 / alpha - B(1, 3) * (alpha^2) / lambda;

% Print intrinsic parameters
K = [alpha gamma u_0;
     0 beta v_0;
     0 0 1;]

% Computing rotation matrix R and t
for i = 1:4
	file_name = char(images(1, i))
	H = homography_list(i * 3 - 2: i * 3, :);
    % lambda = 1 / sqrt(sum((inv(K) * H(:, 1)).^2)); = 1 / sqrt(sum((inv(K) * H(:, 2)).^2));
	lambda = 1 / sqrt(sum((inv(K) * H(:, 2)).^2));
	r_1 = lambda * inv(K) * H(:, 1);
	r_2 = lambda * inv(K) * H(:, 2);
    r_3 = cross(r_1, r_2);
    % Print R and t
	R = [r_1 r_2 r_3]
	t = lambda * inv(K) * H(:, 3)
    
    % Print R_T * R
	R_T_R = R' * R

	% SVD
	[R_U, R_Sigma, R_V] = svd(R);
    % Print
	new_R = R_U * R_V'
	new_R_T_R = new_R' * new_R
end

%% Improving accuracy

homo_list =[];
R_list ={};
t_list ={};

for i = 1:4
	file_name = char(images(1,i))
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
	figure("Name", 'Figure 1 : Projected grid corners')
	image = imread(file_name);
	imshow(image);
	hold on
	title('Figure 1 : Projected grid corners')
    
	plot(p_approx(1, :), p_approx(2, :), 'or'); % circle and red
	% pause(0.8)
    
	% 2: detect Harris corners
    sigma = 2;
    thresh = 500;
    radius = 2;
	[cim, r, c, rsubp, csubp] = harris(rgb2gray(image), sigma, thresh, radius, 0);
	harris_corner=[csubp, rsubp];
    
	figure("Name", 'Figure 2 : Harris corners')
	image=imread(file_name);
	imshow(image);
	hold on
	title('Figure 2 : Harris corners')
    
	plot(harris_corner(:, 1), harris_corner(:, 2), 'dy'); % rhombus and yellow
	% pause(0.8)
    
	% 3: compute closest Harris corner
	for j = 1:size(p_approx, 2)
		n = dist2(harris_corner, p_approx(1: 2, j)');
		[min_val, row_index] = min(n);
		row_list(j) = row_index;
    end
    
	p_correct = harris_corner(row_list(:), :);
	figure("Name", 'Figure 3 : grid points')
	image = imread(file_name);
	imshow(image);
	hold on
	title('Figure 3 : grid points')
    
	plot(p_correct(:, 1), p_correct(:, 2), '+m'); % cross and pinkish red
	% pause(0.8)
    
	% 4: Print new H
	p_correct = [p_correct'; ones(1, size(p_correct, 1));];
    H = homography2d(real_corners, p_correct)
    homo_list = [homo_list; H / H(end,end)];
    
	% 6: error between points
	image_corners = H * real_corners;
	image_corners = image_corners ./ repmat(image_corners(3, :),size(p_approx, 1), 1);
	err_reprojection = sqrt(sum(sum((image_corners - p_correct).^2))) / size(image_corners, 2)
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
     b(4) b(5) b(6);]

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
	H = homo_list(i * 3 - 2: i * 3, :);
	lambda = 1 / sqrt(sum((inv(K) * H(:, 2)).^2));
	r_1 = lambda * inv(K) * H(:, 1);
	r_2 = lambda * inv(K) * H(:, 2);
	r_3 = cross(r_1, r_2);
    % Print R
    R = [r_1 r_2 r_3]
	R_list{i} = R;
    % Print t
	t = lambda * inv(K) * homo(:, 3)
	t_list{i} = t;
end
