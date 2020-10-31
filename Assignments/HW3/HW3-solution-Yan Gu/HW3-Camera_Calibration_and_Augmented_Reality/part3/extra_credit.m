function extra_credit(homo_list, file_list)
V=[];
for i = 1:2
	homo = homo_list(i * 3 - 2: i * 3, :);
	V = [V; get_v(homo)];
end

[V_U, V_S, V_V] = svd(V);
% We add constraint this part
b = [V_V(1, end); 0; V_V(3: 6, end)];
B = [b(1) b(2) b(4);
     b(2) b(3) b(5);
     b(4) b(5) b(6)]

v_0 = (B(1, 2) * B(1, 3) - B(1, 1) * B(2, 3)) / (B(1, 1) * B(2, 2) - B(1, 2)^2);
lambda = B(3, 3) - (B(1, 3)^2 + v_0 * (B(1, 2) * B(1, 3) - B(1, 1) * B(2, 3))) / B(1, 1);
alpha = sqrt(lambda / B(1, 1));
beta = sqrt(lambda * B(1, 1) / (B(1, 1) * B(2, 2) - B(1, 2)^2));
gamma = 0;
u_0 = gamma * v_0 / alpha - B(1, 3) * (alpha^2) / lambda;
K = [alpha gamma u_0;
     0 beta v_0;
     0 0 1;]

for i = 1:2
	file_name = char(file_list(1, i))
	homo = homo_list(i * 3 - 2: i * 3, :);
	r_1 = lambda * inv(K) * homo(:, 1);
	r_2 = lambda * inv(K) * homo(:, 2);
    r_3 = cross(r_1, r_2);
	R = [r_1 r_2 r_3]
	t = lambda * inv(K) * homo(:, 3)
end

end
