function U = Convective_Allen_Cahn2D_variable_optimized(NK, ord)
% Optimized solver for 2D Convective Allen-Cahn equation

    % Parameters
    epsilon = 0.01;
    D_val = epsilon^2;  % Avoid conflict with built-in D
    theta_val = 0.8;
    theta_c_val = 2 * theta_val;
    xb = 0; xe = 1;
    yb = 0; ye = 1;
    T0 = 0; Te = 0.1;  poten = 1;
    N = 64; M = N; 
    hx = (xe - xb) / N; 
    hy = (ye - yb) / M; 
    dt = (Te - T0) / NK;
    x = xb:hx:xe; 
    y = yb:hy:ye; 
    T = T0:dt:Te;
    
    % Potential-dependent parameters
    if poten == 1
        beta_val = 2/3 * sqrt(3);
        kappa_val = 3;
    elseif poten == 2
        beta_val = 0.986783601343632;
        kappa_val = 28.87;
    end

    % Display parameters
    % fprintf('\n *************************************************\n');
    % fprintf(' --- Parameters and Mesh Sizes ---\n');
    % fprintf(' epsilon = %e, D = %e, kappa = %e\n', epsilon, D_val, kappa_val);
    % fprintf(' theta = %f, theta_c = %f, poten = %f\n', theta_val, theta_c_val, poten);
    % fprintf(' hx = %f, hy = %f, dt = %f\n', hx, hy, dt);
    % fprintf(' N = %d, M = %d, NK = %d\n', N, M, NK);
    
    % Mesh generation
    [X_grid, Y_grid] = meshgrid(x, y);
    X = X_grid(:); 
    Y = Y_grid(:);
    TRIM = delaunay(X, Y);
    num_nodes = (N+1)*(M+1);
    
    % Precompute triangle areas and gradients
    areas = compute_triangle_areas(TRIM, X, Y);
    area = mean(areas);
    grad_phi = compute_gradients(TRIM, X, Y, areas);
    
    % Quadrature setup
    QPW = quadrature_weights();
    
    % Assemble matrices
    [A, K] = assemble_AK_matrices(TRIM, areas, grad_phi, QPW, num_nodes);
    I = speye(num_nodes);
    
    % Initial condition
    U = reshape(arrayfun(@(x,y) u0(x,y), X_grid, Y_grid), [], 1);
    % U = imgaussfilt(reshape(U, N+1, M+1), 2);
    U = U(:);
    
    % Preallocate statistics
    uinf = zeros(1, NK+1);
    vol = zeros(1, NK+1);
    egy = zeros(1, NK+1);
    uinf(1) = max(abs(U));
    vol(1) = sum(U) * areas(1);
    egy(1) = comput_Egy(K, U, areas(1), epsilon, poten, theta_val, theta_c_val);
    
    % Time stepping
    kstop = NK;
    tic;
    tol = 1e-6;  % Projection tolerance
    for i = 1:NK
        t = i * dt;
        [CE, SA] = assemble_convective_stabilized_matrix(X, Y, TRIM, t, epsilon, QPW);
        A_lumped = diag(sum(A + SA, 2));
        diagA = diag(A_lumped);
        Lh = A_lumped \ (-D_val * K - CE) - kappa_val * I;
        
        % Time integration
        if ord == 1
            F = -FF(U, poten, theta_val, theta_c_val, kappa_val, area);
            U = phipm(dt, Lh, [U, F], tol, false);
            
        elseif ord == 2
            F = -FF(U, poten, theta_val, theta_c_val, kappa_val, area);
            U1 = phipm(dt, Lh, [U, F], tol, false);
            U1 = projection(U1, beta_val, vol(1), diagA, tol);
            
            F1 = -FF(U1, poten, theta_val, theta_c_val, kappa_val, area);
            dF = (F - F1) / dt;  % Derivative approximation
            U = phipm(dt, Lh, [U, F, dF], tol, false);
            
        elseif ord == 3
            % ========== 3rd ORDER SCHEME ========== %
            % Stage 1
            F0 = -FF(U, poten, theta_val, theta_c_val, kappa_val, area);
            U1 = phipm(dt/3, Lh, [U, (dt/3)*F0], tol, false);
            U1 = projection(U1, beta_val, vol(1), diagA, tol);
            
            % Stage 2
            F1 = -FF(U1, poten, theta_val, theta_c_val, kappa_val, area);
            term1 = phipm(2/3*dt, Lh, U, tol);
            term2 = phipm(1/3*dt, Lh, (2/3*dt)*F1, tol);
            U2 = term1 + term2;
            U2 = projection(U2, beta_val, vol(1), diagA, tol);
            
            % Stage 3
            F2 = -FF(U2, poten, theta_val, theta_c_val, kappa_val, area);
            term1 = phipm(dt, Lh, U, tol);
            term2 = phipm(dt, Lh, (dt/4)*F0, tol);
            term3 = phipm(dt/3, Lh, (3/4*dt)*F2, tol);
            U = term1 + term2 + term3;
            
        elseif ord == 4
            % ========== 4th ORDER SCHEME ========== %
            % Stage 1
            F0 = -FF(U, poten, theta_val, theta_c_val, kappa_val, area);
            term1 = phipm(dt/2, Lh, U, tol);
            term2 = phipm(dt/2, Lh, (dt/2)*F0, tol);
            U1 = term1 + term2;
            U1 = projection(U1, beta_val, vol(1), diagA, tol);
            
            % Stage 2
            F1 = -FF(U1, poten, theta_val, theta_c_val, kappa_val, area);
            term1 = phipm(dt/2, Lh, U, tol);
            term2 = (dt/2) * F1;  % Explicit Euler step
            U2 = term1 + term2;
            U2 = projection(U2, beta_val, vol(1), diagA, tol);
            
            % Stage 3
            F2 = -FF(U2, poten, theta_val, theta_c_val, kappa_val, area);
            term1 = phipm(dt, Lh, U, tol);
            term2 = phipm(dt/2, Lh, dt*F2, tol);
            U3 = term1 + term2;
            U3 = projection(U3, beta_val, vol(1), diagA, tol);
            
            % Stage 4
            F3 = -FF(U3, poten, theta_val, theta_c_val, kappa_val, area);
            term1 = phipm(dt, Lh, U, tol);
            term2 = dt * ( ...
                phipm(dt, Lh, (1/6)*F0, tol) + ...   % phi-function applied
                phipm(dt/2, Lh, (1/3)*F1, tol) + ...  % phi-function applied
                phipm(dt/2, Lh, (1/3)*F2, tol) + ...  % phi-function applied
                (1/6)*F3 ...                           % explicit term
            );
            U = term1 + term2;
        end
        
        U = projection(U, beta_val, vol(1), diagA, tol);
        
        % Update statistics
        uinf(i+1) = max(abs(U));
        vol(i+1) = sum(U) * area;
        egy(i+1) = comput_Egy(K, U, area, epsilon, poten, theta_val, theta_c_val);
    end
    wtime = toc;

    % Post-processing and plotting
   % fprintf('\n Runtime: %.2f seconds\n', wtime);
   % wtime = toc;
   % fprintf (1,'\n MY_PROGRAM took %f seconds to run !!!\n',wtime);
   % 
   % if kstop==NK
   %    fprintf(1,'\n --- A Steady State Had Not Been Reached at T = %e ---\n',T(kstop+1));
   % else
   %    fprintf(1,'\n --- A Steady State Was Reached at T = %e ---\n',T(kstop+1));
   % end
   % fprintf(1,'\n Minimal value = %f, Maximal value = %f\n',min(U(:)),max(U(:)));
   % fprintf(1,'\n Initial energy = %e, Final energy is %e\n',egy(1),egy(kstop+1));
   % fprintf(1,'\n Initial volume = %e, Final volume is %e\n',vol(1),vol(kstop+1));
   % 
   % figure(2)
   % surf(x,y,reshape(U,N+1,M+1));
   % shading interp
   % colormap jet
   % xlabel('X');
   % ylabel('Y');
   % 
   % figure(3)
   % pcolor(x,y,reshape(U,N+1,M+1))
   % shading interp
   % colormap jet
   % xlabel('X');
   % ylabel('Y');
   % axis off;
   % 
   % figure(4);
   % plot(T,uinf,'.-');
   % xlabel('Time');
   % ylabel('Supremum norm');
   % hold on;
   % if poten==1
   %    plot(T,beta_val*ones(size(T,2)),'--r','LineWidth',2); 
   % elseif poten==2
   %    plot(T,beta_val*ones(size(T,2)),'--r','LineWidth',2);  
   % end
   % 
   % figure(5);
   % plot(T,vol,'.-');
   % xlabel('Time');
   % ylabel('Volume');
   % 
   % figure(6);
   % plot(T,egy,'.-');
   % xlabel('Time');
   % ylabel('Energy');

end

% Helper functions
function areas = compute_triangle_areas(TRIM, X, Y)
    p1 = [X(TRIM(:,1)), Y(TRIM(:,1))];
    p2 = [X(TRIM(:,2)), Y(TRIM(:,2))];
    p3 = [X(TRIM(:,3)), Y(TRIM(:,3))];
    v1 = p2 - p1;
    v2 = p3 - p1;
    areas = 0.5 * abs(v1(:,1).*v2(:,2) - v1(:,2).*v2(:,1));
end

function grad_phi = compute_gradients(TRIM, X, Y, areas)
    p1 = [X(TRIM(:,1)), Y(TRIM(:,1))];
    p2 = [X(TRIM(:,2)), Y(TRIM(:,2))];
    p3 = [X(TRIM(:,3)), Y(TRIM(:,3))];
    a = [p2(:,2)-p3(:,2), p3(:,2)-p1(:,2), p1(:,2)-p2(:,2)];
    b = [p3(:,1)-p2(:,1), p1(:,1)-p3(:,1), p2(:,1)-p1(:,1)];
    grad_phi = cat(3, a, b) ./ (2 * areas);
end

function QPW = quadrature_weights()
    QPW = [1/3,            1/3,             1/3,             0.2250;
           0.0597158717,   0.4701420641,    0.4701420641,    0.1323941527;
           0.4701420641,   0.0597158717,    0.4701420641,    0.1323941527;
           0.4701420641,   0.4701420641,    0.0597158717,    0.1323941527;
           0.7974269853,   0.1012865073,    0.1012865073,    0.1259391805;
           0.1012865073,   0.7974269853,    0.1012865073,    0.1259391805;
           0.1012865073,   0.1012865073,    0.7974269853,    0.1259391805];
end

function [A, K] = assemble_AK_matrices(TRIM, areas, grad_phi, QPW, num_nodes)
    [ii, jj, vA, vK] = deal([]);
    for q = 1:size(QPW,1)
        w = QPW(q,4);
        phi = QPW(q,1:3);
        for i = 1:3
            for j = 1:3
                idx = TRIM(:,i);
                jdx = TRIM(:,j);
                A_vals = w * phi(i) * phi(j) * areas;
                K_vals = w * sum(grad_phi(:,i,:) .* grad_phi(:,j,:), 3) .* areas;
                ii = [ii; idx];
                jj = [jj; jdx];
                vA = [vA; A_vals];
                vK = [vK; K_vals];
            end
        end
    end
    A = sparse(ii, jj, vA, num_nodes, num_nodes);
    K = sparse(ii, jj, vK, num_nodes, num_nodes);
end

function [CE, SA] = assemble_convective_stabilized_matrix(X, Y, TRIM, t, epsilon, QPW)
    % Correct implementation of convective and stabilized matrix assembly
    num_nodes = numel(X);
    num_tri = size(TRIM, 1);
    
    % Preallocate sparse matrix storage
    CE_rows = zeros(9 * num_tri, 1);
    CE_cols = zeros(9 * num_tri, 1);
    CE_vals = zeros(9 * num_tri, 1);
    SA_rows = zeros(9 * num_tri, 1);
    SA_cols = zeros(9 * num_tri, 1);
    SA_vals = zeros(9 * num_tri, 1);
    idx = 1;
    
    for tri = 1:num_tri
        % Extract triangle vertices
        nodes = TRIM(tri, :);
        v1 = nodes(1); v2 = nodes(2); v3 = nodes(3);
        
        % Get coordinates
        p1 = [X(v1), Y(v1)];
        p2 = [X(v2), Y(v2)];
        p3 = [X(v3), Y(v3)];
        
        % Calculate triangle area
        v12 = p2 - p1;
        v13 = p3 - p1;
        area = 0.5 * abs(v12(1)*v13(2) - v12(2)*v13(1));
        
        % Compute gradients of basis functions
        grad_phi = zeros(3, 2);
        grad_phi(1, :) = [p2(2)-p3(2), p3(1)-p2(1)] / (2*area);  % grad φ1
        grad_phi(2, :) = [p3(2)-p1(2), p1(1)-p3(1)] / (2*area);  % grad φ2
        grad_phi(3, :) = [p1(2)-p2(2), p2(1)-p1(1)] / (2*area);  % grad φ3
        
        % Compute element diameter (max edge length)
        edge1 = norm(p1 - p2);
        edge2 = norm(p2 - p3);
        edge3 = norm(p3 - p1);
        h = max([edge1, edge2, edge3]);
        
        % Initialize local matrices
        E_local = zeros(3);
        SA_local = zeros(3);
        SE_local = zeros(3);
        vv = Beta(p1(1)/2+p2(1)/3+p3(1)/3,p1(2)/2+p2(2)/3+p3(2)/3,t);
        normV = norm(vv);
        Pei=normV*h/(2*epsilon);
            if Pei>1
               delta=h/(2*normV)*(1-1/Pei);
            else
               delta=0;
            end
        
        % Process all quadrature points
        for q = 1:size(QPW, 1)
            bary = QPW(q, 1:3);
            w = QPW(q, 4);
            
            % Quadrature point in physical space
            xq = bary(1)*p1(1) + bary(2)*p2(1) + bary(3)*p3(1);
            yq = bary(1)*p1(2) + bary(2)*p2(2) + bary(3)*p3(2);
            
            % Velocity vector at quadrature point
            V = Beta(xq, yq, t);
            
            
            % Basis function values at quadrature point (barycentric coords)
            phi = bary;
            
            % Velocity dot gradient (1x3 vector)
            v_dot_grad = V(1)*grad_phi(:, 1)' + V(2)*grad_phi(:, 2)';
            
            % Standard convective term
            E_local = E_local + w * (phi' * v_dot_grad);
            
            % Stabilized mass term
            SA_local = SA_local + w * delta * (v_dot_grad' * phi);
            
            % Stabilized convective term
            SE_local = SE_local + w * delta * (v_dot_grad' * v_dot_grad);
        end
        
        % Scale by triangle area
        E_local = area * E_local;
        SA_local = area * SA_local;
        SE_local = area * SE_local;
        
        % Combined convective matrix: CE = E + SE
        CE_local = E_local + SE_local;
        
        % Add to global sparse matrix storage
        for i = 1:3
            for j = 1:3
                % CE matrix
                CE_rows(idx) = nodes(i);
                CE_cols(idx) = nodes(j);
                CE_vals(idx) = CE_local(i, j);
                
                % SA matrix
                SA_rows(idx) = nodes(i);
                SA_cols(idx) = nodes(j);
                SA_vals(idx) = SA_local(i, j);
                
                idx = idx + 1;
            end
        end
    end
    
    % Create sparse matrices
    CE = sparse(CE_rows, CE_cols, CE_vals, num_nodes, num_nodes);
    SA = sparse(SA_rows, SA_cols, SA_vals, num_nodes, num_nodes);
end


function V = Beta(x, y, t)
    % Velocity field function
    % Inputs:
    %   x, y : Spatial coordinates
    %   t    : Time
    % Output:
    %   V    : Velocity vector [Vx, Vy]
    
    Vx = exp(-t) * sin(2*pi*y);
    Vy = exp(-t) * sin(2*pi*x);
    V = [Vx, Vy];
end

function V = FF(U, poten, theta, theta_c, kappa, area)
    if poten == 1
        V = U .* (U.^2 - 1);
    elseif poten == 2
        V = (theta/2) * (log(1+U) - log(1-U)) - theta_c * U;
    end
    V = V - sum(V(:))*area;
    V = V - kappa * U;
end

function u_proj = projection(v, beta, mass_target, diagA, tol)
    xi = 0;
    for iter = 1:100
        u_proj = min(beta, max(-beta, v - xi));
        current_mass = dot(diagA, u_proj);
        residual = current_mass - mass_target;
        if abs(residual) < tol, break; end
        if iter == 1
            xi = xi - residual * 0.1;
        else
            xi = xi - residual * delta_xi / (residual - residual_prev);
        end
        residual_prev = residual;
        delta_xi = -residual * 0.1; % Initial step size for next iteration
    end
end

function egy = comput_Egy(K, U, area, epsilon, poten, theta, theta_c)
    if poten == 1
        egy1 = 0.25 * (1 - U.^2).^2;
    elseif poten == 2
        egy1 = theta/2*((1+U).*log(1+U) + (1-U).*log(1-U)) - (theta_c^2)/2*U.^2;
    end
    egy2 = 0.5 * epsilon^2 * (U' * K * U);
    egy = sum(egy1) * area + egy2;
end


function V = u0(x, y)
    V = sin(2*pi*x) .* sin(2*pi*y);
end