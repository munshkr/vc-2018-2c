%% Load intrinsic params and image list

load(calib_resuls_filename,'KK', 'kc')

imageFileNames = dir(fullfile(images_path,'*.png'));
imageFileNames = fullfile({imageFileNames.folder}, {imageFileNames.name});
fprintf('%d images to process\n',length(imageFileNames));

%% Detect checkerboards in images

% Generate world coordinates of the corners of the squares
squareSize = 8.7; % small checkerboard is 8 x 10 (rows x cols) with 8.7 mm square size
%squareSize = 14.9; % big checkerboard is 8 x 10 with 14.9 mm square size

[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames);

fprintf('%d of %d images detected\n',sum(imagesUsed),length(imageFileNames));

assert(length(imageFileNames)==sum(imagesUsed));
%
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

N = length(imageFileNames);
iPt = cell(N,1);
wPt = cell(N,1);
for i = 1:N
    iPt{i} = imagePoints(:,:,i);
    wPt{i} = worldPoints(:,[2 1]);
end

%% Detect CALTag patterns
% https://www.cs.ubc.ca/labs/imager/tr/2010/Atcheson_VMV2010_CALTag/
%!git clone https://github.com/brada/caltag
caltag_path = [pwd '/caltag'];
addpath(caltag_path);

cd([caltag_path '/GeneratePattern']);
!python ./generate_pattern.py -r 9 -c 7
cd('../../');
%
N = length(imageFileNames);
iPt = cell(N,1);
wPt = cell(N,1);

for i=1:length(imageFileNames)
    I = imread(imageFileNames{i});
    fprintf('%s\n',imageFileNames{i});
    [ps,us] = caltag(I, [caltag_path '/GeneratePattern/output.mat'], false );
    wPt{i} = ps(:,[2 1]);
    iPt{i} = us(:,[2 1]);
end

%% check coorner coordinates 
figure;

i = 1;
us = iPt{i};
ps = wPt{i};
I = imread(imageFileNames{i});
imagesc(I); colormap(gray);
hold on;
plot(us(:,1),us(:,2),'g*')

for i=1:length(us)
    text(us(i,1),us(i,2),sprintf('(%.0f,%.0f)',ps(i,1),ps(i,2)),'Color','green','FontSize',14);
end
% 

%% estimate each checkboard pose (R,t)
N = length(imageFileNames);
Rt = cell(N,1);
Tt = cell(N,1);

for i = 1:N
    us = iPt{i};
    ps = wPt{i};
    
    [R,T] = estimateExtrinsics(us,ps,KK,kc);
 
    Rt{i} = R;
    Tt{i} = T;
end

save([images_path '/ChessboardCorners.mat'],'iPt','wPt','Rt','Tt');
