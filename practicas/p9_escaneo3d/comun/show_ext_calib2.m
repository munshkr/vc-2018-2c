%%%%%%%%%%%%%%%%%%%% SHOW EXTRINSIC RESULTS %%%%%%%%%%%%%%%%%%%%%%%%
%%% Camera Centered %%%

if ~exist('show_camera')
    show_camera = 1;
end

if ~exist('n_ima')|| ~exist('fc')
   fprintf(1,'No calibration data available.\n');
   return;
end

if ~exist(['Rc']) || ~exist(['Tc'])
   fprintf(1,'No calibration data available.\n');
   return;
end

if ~exist('alpha_c')
   alpha_c = 0;
end

% Color code for each image:

colors = 'brgkcm';

no_grid = 0;


if ~exist('dX')
   eval('dX = norm(Tc{1})/10;');
   dY = dX;
end


IP = 5*dX*[1 -alpha_c 0;0 1 0;0 0 1]*[1/fc(1) 0 0;0 1/fc(2) 0;0 0 1]*[1 0 -cc(1);0 1 -cc(2);0 0 1]*[0 nx-1 nx-1 0 0 ; 0 0 ny-1 ny-1 0;1 1 1 1 1];
BASE = 5*dX*([0 1 0 0 0 0;0 0 0 1 0 0;0 0 0 0 0 1]);
IP = reshape([IP;BASE(:,1)*ones(1,5);IP],3,15);

if ishandle(4)
	figure(4);
   [a,b] = view;
else
   figure(4);
   a = 50;
   b = 20;
end

if show_camera
    figure(4);
    plot3(BASE(1,:),BASE(3,:),-BASE(2,:),'b-','linewidth',2);
    hold on;
    plot3(IP(1,:),IP(3,:),-IP(2,:),'r-','linewidth',2);
    text(6*dX,0,0,'X_c');
    text(-dX,5*dX,0,'Z_c');
    text(0,0,-6*dX,'Y_c');
    text(-dX,-dX,dX,'O_c');
else
    figure(4);
    clf;
    hold on;
end


for kk = 1:n_ima
     
    %eval(['XX_kk = X_' num2str(kk) ';']);
    %eval(['omc_kk = omc_' num2str(kk) ';']);
    eval(['XX_kk = XX{' num2str(kk) '};']);
    eval(['R_kk = Rc{' num2str(kk) '};']);
    eval(['Tc_kk = Tc{' num2str(kk) '};']);
    N_kk = size(XX_kk,2);

    if ~exist('n_sq_x') || ~exist('n_sq_y')
        no_grid = 1;
    elseif (N_kk ~= ((n_sq_x+1)*(n_sq_y+1)))
        no_grid = 1;
    end

    %R_kk = rodrigues(omc_kk);
    

    YY_kk = R_kk * XX_kk + Tc_kk * ones(1,length(XX_kk));

    uu = [-dX;-dY;0]/2;
    uu = R_kk * uu + Tc_kk; 

    if ~no_grid
        YYx = zeros(n_sq_x+1,n_sq_y+1);
        YYy = zeros(n_sq_x+1,n_sq_y+1);
        YYz = zeros(n_sq_x+1,n_sq_y+1);

        YYx(:) = YY_kk(1,:);
        YYy(:) = YY_kk(2,:);
        YYz(:) = YY_kk(3,:);

        figure(4);
        hhh= mesh(YYx,YYz,-YYy);
        set(hhh,'edgecolor',colors(rem(kk-1,6)+1),'linewidth',1); %,'facecolor','none');
        %plot3(YY_kk(1,:),YY_kk(3,:),-YY_kk(2,:),['o' colors(rem(kk-1,6)+1)]);
        %text(uu(1),uu(3),-uu(2),num2str(kk),'fontsize',14,'color',colors(rem(kk-1,6)+1));
    else
        figure(4);
        plot3(YY_kk(1,:),YY_kk(3,:),-YY_kk(2,:),['.' colors(rem(kk-1,6)+1)]);
        %text(uu(1),uu(3),-uu(2),num2str(kk),'fontsize',14,'color',colors(rem(kk-1,6)+1));
    end
end

%%

figure(4);rotate3d on;
axis('equal');
title('Extrinsic parameters (camera-centered)');
%view(60,30);
view(a,b);
grid on;
hold off;
axis vis3d;
axis tight;
set(4,'color',[1 1 1]);
if ~show_camera
    xlabel('X_c');
    ylabel('Z_c');
    zlabel('<-- Y_c');
end

set(4,'Name','3D','NumberTitle','off');

%fprintf(1,'To generate the complete movie associated to the optimization loop, try: check_convergence;\n');


if exist('h_switch2')==1
    if ishandle(h_switch2)
        delete(h_switch2);
    end
end

if n_ima ~= 0
    if show_camera
        h_switch2 = uicontrol('Parent',4,'Units','normalized', 'Callback','show_camera=0;show_ext_calib2;', 'Position',[1-.30 0.04  .30  .04],'String','Remove camera reference frame','fontsize',8,'fontname','clean','Tag','Pushbutton1');
    else
        h_switch2 = uicontrol('Parent',4,'Units','normalized', 'Callback','show_camera=1;show_ext_calib2;', 'Position',[1-.30 0.04  .30  .04],'String','Add camera reference frame','fontsize',8,'fontname','clean','Tag','Pushbutton1');
    end
end


if exist('h_switch')==1
    if ishandle(h_switch)
        delete(h_switch);
    end
end

if n_ima ~= 0
h_switch = uicontrol('Parent',4,'Units','normalized', 'Callback','show_ext_calib', 'Position',[1-.30 0  .30  .04],'String','Switch to world-centered view','fontsize',8,'fontname','clean','Tag','Pushbutton1');
end

figure(4);
rotate3d on;
