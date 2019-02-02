%% Load calibration from file

load(fullfile(turntable_images_path,'CameraPose.mat'),'worldRotation','worldTranslation');
load(calib_resuls_filename,'KK', 'kc')
load([turntable_images_path '/TurntableCalib.mat'],'angs');
load([laser_color_images_path '/../LaserPlane_Results.mat'],'laserPlane');

%% Load images of the scanned object

imageFileNames = dir(fullfile(scanned_object_images_path,'*.png'));
imageFileNames = fullfile({imageFileNames.folder}, {imageFileNames.name});
fprintf('%d images to process\n',length(imageFileNames));

N = length(imageFileNames);

if exist('angs','var')
    assert(length(angs)==N);
else
    deg = 5;
    fprintf('Using %d deg rotation\n',deg);
    angs = -5*pi/180*(0:N-1);
end

%%

pointcloud = cell(N,1);

figure
for i = 1:N
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % detect laser line
    I = imread(imageFileNames{i});
    if ~exist('mask','var')
    %%
    [~,rect] = imcrop(I);
    mask = zeros(size(I,1),size(I,2));
    rect = round(rect);
    mask(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)) = 1;
    figure, imagesc(mask)
    end
    [Lx,Ly] = detectLaser2(I,mask);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % triangulate
    points_c = triangulateLaserLine(laserPlane,[Lx,Ly]',KK,kc);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % transform from camera to turntable coords (world coordinate system)
    points = worldRotation' * (points_c - worldTranslation);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % turntable rotation angle
    ang  = angs(i);
    Rturn = [cos(ang) -sin(ang) 0
             sin(ang)  cos(ang) 0
             0         0        1
            ];    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % undo the turntable location to place the point in world coords
    points = Rturn' * points;

    pointcloud{i} = points;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% display
    if 1

    subplot(121)
    imshow(I)
    hold on;
    plot_axis_img(worldRotation,worldTranslation,KK,kc);
    plot(Lx,Ly,'g+');
    title(sprintf('%d/%d',i,N)) 
      
    subplot(122)
    scatter3(points(1,:),points(2,:),points(3,:),'+');
    title(sprintf('Rot %.2f',ang))
    xlabel('X'); ylabel('Y'); zlabel('Z');
    hold on; grid on;
    %view(-ang/pi*180+230,30)
    view(230,30)
    
    pause(0.1)
    end    
end

pts = [pointcloud{:}];

%
figure;
scatter3(pts(1,:),pts(2,:),pts(3,:),'.');
xlabel('X'); ylabel('Y'); zlabel('Z');
grid on; 
axis equal;
axis vis3d;
rotate3d on;
