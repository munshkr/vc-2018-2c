%% Load data from disk
load([laser_color_images_path '/../LaserChessboardCorners.mat'],'Rl','Tl');
load(calib_resuls_filename,'KK', 'kc')

imageFileNames = dir(fullfile(laser_color_images_path,'*.png'));
imageFileNames = fullfile({imageFileNames.folder}, {imageFileNames.name});
fprintf('%d images to process\n',length(imageFileNames));

assert(length(imageFileNames)==length(Rl));

%% 
n_ima = length(imageFileNames);
pointcloud = cell(n_ima,1);

for i=1:n_ima
    %%
    I = imread(imageFileNames{i});
    
    % detect laser line
    if ~exist('mask','var')
    %
    [~,rect] = imcrop(I);
    title('Select Laser Plane ROI');
    mask = zeros(size(I,1),size(I,2));
    rect = round(rect);
    mask(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)) = 1;
    figure; imagesc(mask); 
    end
    %
    [Lx,Ly] = detectLaser2(I,mask);

    if 1
    LI = zeros(size(I,1),size(I,2),1);
    LI(sub2ind(size(I),Ly,Lx)) = 1;
    subplot(121)
    imagesc(I)
    subplot(122)
    imagesc(LI)
    title(num2str(i))
    pause(0.1);
    end
 
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % estimate checkerboardPlane
    R = Rl{i};
    T = Tl{i};
    checkerboardPlane = estimateCheckerboardPlane(R,T);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % triangulate laser pixels
    laserpoints = triangulateLaserLine(checkerboardPlane,[Lx,Ly]',KK,kc);

    pointcloud{i} = laserpoints;
end

%%%%%%%%%%%%%%%%%%%%%%
% estimatePlane 
[laserPlane,LPts] = estimateLaserPlane(pointcloud);
laserPlane
swap_axes = 0;

save([laser_color_images_path '/../LaserPlane_Results.mat'],'laserPlane','LPts');

figure, hold on, grid on;
view(230,30);
xlabel('X'); ylabel('Y'); zlabel('Z');
scatter3(LPts(1,:),LPts(2,:),LPts(3,:),'+');


