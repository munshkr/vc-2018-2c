%% Load data from disk

load([laser_color_images_path '/../LaserPlane_Results.mat'],'laserPlane','LPts');

swap_axes = 0;
%% show laser plane 
if swap_axes
  figure(4)
else
  figure, hold on, grid on;
  view(230,30);
  xlabel('X'); ylabel('Y'); zlabel('Z');
  rotate3d on;
end
n = laserPlane(1:3);
d = laserPlane(4);

% intersection with plane Z = 0
xy_intZ0 = @(x) (-d - n(1)*x) / n(2);
yx_intZ0 = @(x) (-d - n(2)*x) / n(1);

% intersection with plane Z = k
k = 450;
xy_intZk = @(x) (-d - n(3)*k - n(1)*x) / n(2);
yx_intZk = @(x) (-d - n(3)*k - n(2)*x) / n(1);

y = 100;
qp1 = [yx_intZ0(y) y 0]';
y = -100;
qp2 = [yx_intZ0(y) y 0]';
y = -100;
qp3 = [yx_intZk(y) y k]';
y = 100;
qp4 = [yx_intZk(y) y k]';

rect_corners = [qp1, qp2, qp3, qp4];
%rect_corners = [qp1, qp2, qp3, qp4, qp1];
if swap_axes
    patch(rect_corners(1,:),rect_corners(3,:),-rect_corners(2,:),'yellow')
    %plot3(rect_corners(1,:),rect_corners(3,:),-rect_corners(2,:))
    scatter3(LPts(1,:),LPts(3,:),-LPts(2,:),'+');
else
    patch(rect_corners(1,:),rect_corners(2,:),rect_corners(3,:),'yellow')
    %plot3(rect_corners(1,:),rect_corners(2,:),rect_corners(3,:))
    scatter3(LPts(1,:),LPts(2,:),LPts(3,:),'+');
end

%% Show with camera and chessboard planes
load([laser_color_images_path '/../LaserChessboardCorners.mat'],'iPt','wPt','Rl','Tl');
load(calib_resuls_filename,'KK', 'kc')

imageFileNames = dir(fullfile(laser_color_images_path,'*.png'));
imageFileNames = fullfile({imageFileNames.folder}, {imageFileNames.name});

assert(length(imageFileNames)==length(Rl));

I = imread(imageFileNames{1});

n_ima = length(iPt);
XX = cell(n_ima,1);
Rc = cell(n_ima,1);
Tc = cell(n_ima,1);
for kk=1:n_ima
    eval(['XX{' num2str(kk) '} = wPt{' num2str(kk) '''};']);
    eval(['Rc{' num2str(kk) '} = Rl{' num2str(kk) '};']);
    eval(['Tc{' num2str(kk) '} = Tl{' num2str(kk) '};']);
end

fc = [KK(1,1) KK(2,2)];
cc = [KK(1,3) KK(2,3)];

[ny,nx,~] = size(I);
%show_ext_calib;
show_ext_calib2; % camera centered
hold on;
swap_axes = 1;

% execute cell from above 
