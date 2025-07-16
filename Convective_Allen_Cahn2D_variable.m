function U = Convective_Allen_Cahn2D_variable(NK,ord)

% This program solve the two-dimensional Convective Allen-Cahn equation
% u_t + v(x,y) \nabla u = D*\Delta(u) + f(u)

%Domain
%  [xb,xe] x [yb,ye]
%Number of grid sizes 
%   N x M
%Time interval
%  [T0,Te] 
%Number of step sizes
%   NK
%potential functions
%   poten

%potential function type:
%poten = 1; % doubel-well
%poten = 2; % Flory-Huggins

%Maximum bound
%   beta

% clc,clear;
% parameters for Allen-Cahn equation
global epsilon D;
global kappa beta;
global theta theta_c poten;
global area;

epsilon = 0.01; D = epsilon^2;
poten = 1; theta = 0.8; 
theta_c = 2*theta; 
% domain and partition
xb = 0; xe = 1;
yb = 0; ye = 1;
T0 = 0; Te = 0.1; 
N = 128; M = N;
hx = (xe-xb)/N; hy = (ye-yb)/M; dt = (Te-T0)/NK ; 
x = xb: hx:xe; y = yb:hy:ye; T = T0:dt:Te; 

poten = 1;  % Potential function type

if poten==1
    beta = 2/3*sqrt(3); 
    kappa = 3;
elseif poten==2
    beta = 0.986783601343632;
    kappa = 28.87;
end

% if poten==1
%    fprintf(1,'\n Potential function Type: Doubel-well\n');
% end
% if poten==2
%    fprintf(1,'\n Potential function Type: Flory-Huggins\n');
% end

% fprintf(1,'\n *************************************************\n');
% fprintf(1,'\n --- Parameters and Mesh Sizes ---\n');
% fprintf(1,'\n epsilon = %e, D = %e, kappa = %e',epsilon,D,kappa);
% fprintf(1,'\n theta = %f, theta_c = %f, poten = %f\n',theta,theta_c,poten);
% fprintf(1,'\n hx = %d, hy = %d, dt = %d\n',hx,hy,dt);
% fprintf(1,'\n N = %d, M = %d, NK = %d\n',N,M,NK);


% Generate 2D coordinate nodes
  X=kron(ones(N+1,1),x'); 
  Y=kron(y',ones(M+1,1));
% Generate 2D mesh
  TRIM=delaunay(X,Y);
  TRIN=length(TRIM);
  Nx=(N+1)^2; My = (M+1)^2;

% Draw the triangular region and its corresponding coordinate points
  % figure(1);
  % trisurf(TRIM,X,Y,zeros(Nx,1))
  % axis equal
  % colormap jet
  % hold on
  % for k=1:size(X,1)
  %     text(X(k),Y(k),num2str(k));hold on;
  % end

   % Initial condition
    U=zeros(N+1,M+1);
    for i=1:N+1
        for j = 1:M+1
            U(i,j)=u0(x(i),y(j));
        end
    end
  QPW=[1/3            1/3             1/3             0.2250
    0.0597158717    0.4701420641    0.4701420641    0.1323941527
    0.4701420641    0.0597158717    0.4701420641    0.1323941527
    0.4701420641    0.4701420641    0.0597158717    0.1323941527
    0.7974269853    0.1012865073    0.1012865073    0.1259391805
    0.1012865073    0.7974269853    0.1012865073    0.1259391805
    0.1012865073    0.1012865073    0.7974269853    0.1259391805];

A=sparse(Nx,My);
K=sparse(Nx,My);
ATri=cell(TRIN,1);KTri=cell(TRIN,1);
for i=1:TRIN
    II=TRIM(i,1);JJ=TRIM(i,2);KK=TRIM(i,3);
    xi=X(II);yi=Y(II);
    xj=X(JJ);yj=Y(JJ);
    xm=X(KK);ym=Y(KK);
   
    coords(1,1) = xi; coords(1,2) = yi;
    coords(2,1) = xj; coords(2,2) = yj;
    coords(3,1) = xm; coords(3,2) = ym;
    
    
    S2d=1/2*det([xi,yi,1;xj,yj,1;xm,ym,1]);
    ai=det([yj,1;ym,1]);aj=det([ym,1;yi,1]);am=det([yi,1;yj,1]);
    bi=-det([xj,1;xm,1]);bj=-det([xm,1;xi,1]);bm=-det([xi,1;xj,1]);
    ci=det([xj,yj;xm,ym]);cj=det([xm,ym;xi,yi]);cm=det([xi,yi;xj,yj]);
    AA=[ai,bi]/(2*S2d);
    BB=[aj,bj]/(2*S2d);
    CC=[am,bm]/(2*S2d);
    Kt=([AA;BB;CC]*[AA;BB;CC]');
    At=0;
    NN=@(x,y) (1/(2*S2d)*[ai*x+bi*y+ci,aj*x+bj*y+cj,am*x+bm*y+cm]'*1/(2*S2d)*[ai*x+bi*y+ci,aj*x+bj*y+cj,am*x+bm*y+cm]);
    % midpoint=(XYZpp(II,:)+XYZpp(JJ,:)+XYZpp(KK,:))/3;
    
    for qq=1:length(QPW)
        xq = [xi,xj,xm]*QPW(qq,1:3)';
        yq = [yi,yj,ym]*QPW(qq,1:3)';
        At=At+QPW(qq,4)*NN(xq,yq); % standard Mass matrix 
        %%%%%%%%%%%%%%%%%
    end
    ATri{i}=S2d*At;
    KTri{i}=S2d*Kt;
end

for i=1:TRIN 
    II=TRIM(i,1);JJ=TRIM(i,2);KK=TRIM(i,3);
    K([II,JJ,KK],[II,JJ,KK])=K([II,JJ,KK],[II,JJ,KK])+KTri{i}; % Stifness matrix
    A([II,JJ,KK],[II,JJ,KK])=A([II,JJ,KK],[II,JJ,KK])+ATri{i}; % Standard mass matrix
end
   area = S2d;

   % stabilization technique
   I = speye(Nx);
   % Initial coarsening dynamic
   % U = 0.9*randi([-1 1],N+1,M+1);

   % 高斯滤波初始化
   U = imgaussfilt(U, 2);
   U = reshape(U,(N+1)*(M+1),1);

   uinf(1) = max(abs(U(:)));
   vol(1) =  sum(U(:))*area;
   egy(1) = comput_Egy(K, U, area);
   kstop = NK; 
   tic;
   for i = 1:NK
       [CE,SA] = assemble_convective_stabilized_matrix(Nx,My,X,Y,TRIN,TRIM,i*dt);
       A = diag(sum(A+SA,2)); 
       Lh = A\(-D*K - CE) - kappa*I;
       if ord==1
          U = phipm(dt,Lh,[U,-FF(U)],eps,false);
       elseif ord==2
          U1 = phipm(dt,Lh,[U,-FF(U)],eps,false);
          % U1 = projection(U1, beta, vol(1), area, 1e-8);
          U = phipm(dt,Lh, [U, -FF(U), 1/dt*(FF(U)-FF(U1))],eps,false);
       elseif ord==3
           % step 1 
           U1 = phipm(dt/3, Lh, U,eps) + phipm(dt/3, Lh, dt/3*(-FF(U)));

           % step 2
           U2 = phipm(2/3*dt, Lh, U, eps) + phipm(1/3*dt, Lh, 2/3*dt*(-FF(U1)), eps);

           % step3
           U = phipm(dt, Lh, U, eps) + phipm(dt, Lh, dt/4*(-FF(U))) ...
                    + phipm(dt/3, Lh, 3/4*dt*(-FF(U2)));
       elseif ord==4
           % step 1
           U1 = phipm(dt/2, Lh, U, eps) + phipm(dt/2, Lh, dt/2*(-FF(U)), eps);

          % step 2
           U2 = phipm(dt/2, Lh, U, eps) + dt/2*(-FF(U1));

         % step 3
           U3 = phipm(dt, Lh, U, eps) + phipm(dt/2, Lh, dt*(-FF(U2)));

         % step 4
           U = phipm(dt, Lh, U, eps) + dt*( phipm(dt, Lh, 1/6*(-FF(U)), eps) ...
                    + phipm(dt/2, Lh, 1/3*(-FF(U1)), eps) ...
                    +  phipm(dt/2, Lh, 1/3*(-FF(U2)), eps) + 1/6*(-FF(U3)) );
       end
       % U = projection(U, beta, vol(1), diag(A), 1e-8); 
       uinf = [uinf max(abs(U(:)))];
       egy(i+1) = comput_Egy(K, U, S2d);
       vol(i+1) = sum(U(:))*area;
   % 
   %   %    figure(2);
   %   %    drawnow;
   %   %    trisurf(TRIM,X,Y,Un); 
   %   %    shading('interp');
   %   %    axis image; 
   %   %    xlabel('x');ylabel('y');
   %   %    colorbar('SouthOutside');
   %   %    colormap jet
   %   %    title(['T=',num2str(k*dt)])
   %   % % pause(0.0001)
   end

   wtime = toc;
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
   % 
   % figure(4);
   % plot(T,uinf,'.-');
   % xlabel('Time');
   % ylabel('Supremum norm');
   % hold on;
   % if poten==1
   %    plot(T,beta*ones(size(T,2)),'--r','LineWidth',2); 
   % elseif poten==2
   %    plot(T,beta*ones(size(T,2)),'--r','LineWidth',2);  
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

% Energy calculation
function egy = comput_Egy(K, U, area)
global epsilon poten theta theta_c;
if poten == 1  % Double-well potential
   % Potential energy part
   egy1 = 1/4*(1 - U.^2).^2;
   % Gradient energy part
   egy2 = epsilon^2/2 * (U' * K * U);
   egy = sum(egy1)*area + egy2*area;
elseif poten == 2  % Flory-Huggins potential
   % Potential energy part
   egy1 = theta/2*((1+U).*log(1+U) + (1-U).*log(1-U)) - (theta_c^2)/2*U.^2;
   % Gradient energy part
   egy2 = epsilon^2/2 * (U' * K * U);
   egy = sum(egy1)*area + egy2*area;
end
end

% Solution projection for bound constraints
function u_proj = projection(v, beta, mass_target, diagA, tol)
    xi_prev = 0;        % Previous Lagrange multiplier
    xi_curr = 0;        % Current Lagrange multiplier
    max_iter = 100;     % Maximum iterations
    residual_prev = 0;  % Previous mass residual

    for iter = 1:max_iter
        % Projected solution candidate
        val = v - xi_curr;
        u_proj = val;
        % Apply [-β, β] bounds
        u_proj(val > beta) = beta;
        u_proj(val < -beta) = -beta;

        % Calculate current mass
        current_mass = diagA'*u_proj;
        residual = current_mass - mass_target;  % Mass residual

        % Check convergence
        if abs(residual) < tol
            break;
        end

        % Secant method update for xi
        if iter == 1
            xi_next = xi_curr - residual * 0.1;  % Initial guess adjustment
        else
            delta_xi = xi_curr - xi_prev;
            delta_res = residual - residual_prev;
            if abs(delta_res) < 1e-12
                break;
            end
            xi_next = xi_curr - residual * delta_xi / delta_res;
        end

        % Update variables for next iteration
        xi_prev = xi_curr;
        xi_curr = xi_next;
        residual_prev = residual;
    end
end


function V = FF_org(U)
global poten;
global theta theta_c;
global area;
if poten==1
   V = U.*(U.*U-1); 
end
if poten==2
   V = (theta/2*(log(1+U)-log(1-U))-theta_c*U);   
end
V = V - sum(V(:))*area;
end

function V = FF(U) 
global kappa;
V = FF_org(U) - kappa*U;
end

function V = u0(x,y)
 % V = cos(2*pi*x)*cos(2*pi*y);
 % if abs((x-0.3)^2 + (y-0.3)^2)<0.2^2
 %     V = 0.9;
 % else
 %     V = -0.9;
 % end
 V = 0.9*sin(100*pi*x)*sin(100*pi*y);
end

function V = Beta(x, y, t)
    vx = exp(-t).*sin(2*pi*y);  % x-component of velocity
    vy = exp(-t).*sin(2*pi*x);  % y-component of velocity
    V = [vx,vy];
end


function [CE,SA] = assemble_convective_stabilized_matrix(Nx,My,X,Y,TRIN,TRIM,t)
global epsilon;
QPW=[1/3            1/3             1/3             0.2250
    0.0597158717    0.4701420641    0.4701420641    0.1323941527
    0.4701420641    0.0597158717    0.4701420641    0.1323941527
    0.4701420641    0.4701420641    0.0597158717    0.1323941527
    0.7974269853    0.1012865073    0.1012865073    0.1259391805
    0.1012865073    0.7974269853    0.1012865073    0.1259391805
    0.1012865073    0.1012865073    0.7974269853    0.1259391805];

E=sparse(Nx,My);
SA=sparse(Nx,My);
SE=sparse(Nx,My);
ETri=cell(TRIN,1);
SATri=cell(TRIN,1);
SETri=cell(TRIN,1);
XYZpp=[X,Y];
for i=1:TRIN
    II=TRIM(i,1);JJ=TRIM(i,2);KK=TRIM(i,3);
    xi=X(II);yi=Y(II);
    xj=X(JJ);yj=Y(JJ);
    xm=X(KK);ym=Y(KK);
   
    S2d=1/2*det([xi,yi,1;xj,yj,1;xm,ym,1]);
    ai=det([yj,1;ym,1]);aj=det([ym,1;yi,1]);am=det([yi,1;yj,1]);
    bi=-det([xj,1;xm,1]);bj=-det([xm,1;xi,1]);bm=-det([xi,1;xj,1]);
    ci=det([xj,yj;xm,ym]);cj=det([xm,ym;xi,yi]);cm=det([xi,yi;xj,yj]);
    AA=[ai,bi]/(2*S2d);
    BB=[aj,bj]/(2*S2d);
    CC=[am,bm]/(2*S2d);
    Et=0;SAt=0;SEt=0;
    Nt=@(x,y) (1/(2*S2d)*[ai*x+bi*y+ci,aj*x+bj*y+cj,am*x+bm*y+cm]);
    grad_phi = [AA;BB;CC];
    hi=max(  [norm(XYZpp(II,:)-XYZpp(JJ,:)),norm(XYZpp(KK,:)-XYZpp(JJ,:)),norm(XYZpp(II,:)-XYZpp(KK,:))] );
    % midpoint=(XYZpp(II,:)+XYZpp(JJ,:)+XYZpp(KK,:))/3;
    for qq=1:length(QPW)
        xq = [xi,xj,xm]*QPW(qq,1:3)';
        yq = [yi,yj,ym]*QPW(qq,1:3)';
        %%%%%%%%%%%%%%% stabilized coefficient
        Pei=norm(Beta(xq,yq,t))*hi/(2*epsilon);
         if Pei>1
            delta=hi/(2*norm(Beta(xq,yq,t)))*(1-1/Pei);
         else
            delta=0;
         end
        Et=Et+QPW(qq,4)*Nt(xq,yq)'*(Beta(xq,yq,t)*grad_phi'); % convective matrix
        %%%%%%%%%%%%%%%%%
        % stabilized mass matrix
        SAt=SAt+delta*QPW(qq,4)*(Beta(xq,yq,t)*grad_phi')'*Nt(xq,yq);
        % stabilized convective matrix
        SEt=SEt+delta*QPW(qq,4)*(Beta(xq,yq,t)*grad_phi')'*(Beta(xq,yq,t)*grad_phi');
    %%%%%%%%%%%%%%%%%%%%%%%
    end
    ETri{i}=S2d*Et;
    %%%%%%%%%%%%%%%%%%%%%%%
    SATri{i}=S2d*SAt;
    SETri{i}=S2d*SEt;
    
end

for i=1:TRIN 
    II=TRIM(i,1);JJ=TRIM(i,2);KK=TRIM(i,3);
    E([II,JJ,KK],[II,JJ,KK])=E([II,JJ,KK],[II,JJ,KK])+ETri{i}; % standard convective matrix
    SA([II,JJ,KK],[II,JJ,KK])=SA([II,JJ,KK],[II,JJ,KK])+SATri{i}; % Stabilized mass matrix
    SE([II,JJ,KK],[II,JJ,KK])=SE([II,JJ,KK],[II,JJ,KK])+SETri{i}; % Stabilized convective matrix    
end
   CE = E + SE;
end