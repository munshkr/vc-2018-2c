%% Read color images from disk
imageFileNames = dir(fullfile(laser_color_images_path,'*.png'));
imageFileNames = fullfile({imageFileNames.folder}, {imageFileNames.name});
fprintf('%d images to process\n',length(imageFileNames));

%% detect laser line pixels

i = 1; % test on several images
I = imread(imageFileNames{i});
figure, imshow(I)
    
if ~exist('mask','var')
    [~,rect] = imcrop(I);
    title('Select Laser Plane ROI');
    mask = zeros(size(I,1),size(I,2));
    rect = round(rect);
    mask(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)) = 1;
    figure; imagesc(mask); 
end
%
[Lx,Ly] = detectLaser2(I,mask);

%% display results
LI = zeros(size(I,1),size(I,2),1);
LI(sub2ind(size(I),Ly,Lx)) = 1;

figure;
subplot(121)
imagesc(I)
subplot(122)
imagesc(LI)
title(num2str(i))




