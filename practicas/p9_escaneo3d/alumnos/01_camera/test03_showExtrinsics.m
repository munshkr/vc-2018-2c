%% Load calibration from file

set_paths;

% intrinsic parameteres
KK = struct2array(load(calib_resuls_filename,'KK'));
kc = struct2array(load(calib_resuls_filename,'kc'));

n_ima = struct2array(load(calib_resuls_filename,'n_ima'));

Rc = cell(n_ima,1);
Tc = cell(n_ima,1);
iPt = cell(n_ima,1);
wPt = cell(n_ima,1);
for i=1:n_ima
    Rc{i} = struct2array(load(calib_resuls_filename,['Rc_' num2str(i)]));
    Tc{i} = struct2array(load(calib_resuls_filename,['Tc_' num2str(i)]));
    iPt{i} = struct2array(load(calib_resuls_filename,['x_' num2str(i)]));
    wPt{i} = struct2array(load(calib_resuls_filename,['X_' num2str(i)]));
end


%% Show calibration 

images = dir(fullfile(calib_images_path,'*.png'));
images = fullfile({images.folder}', {images.name}');

assert(length(images) == n_ima);
%
figure;

for i=1:length(images)
    %%
    us = iPt{i};
    I = imread(images{i});    
    imshow(I);
    title(sprintf('Image %d of %d',i,n_ima))
    
    hold on;
    plot(us(1,:),us(2,:),'g+')
    
    pause;
end

figure;
montage(images)

%%
load(calib_resuls_filename)

XX = cell(n_ima,1);
Rc = cell(n_ima,1);
Tc = cell(n_ima,1);
for kk=1:n_ima
    eval(['XX{' num2str(kk) '} = X_' num2str(kk) ';']);
    eval(['Rc{' num2str(kk) '} = Rc_' num2str(kk) ';']);
    eval(['Tc{' num2str(kk) '} = Tc_' num2str(kk) ';']);
end

show_camera = 1;
%alpha_c = 0;
n_sq_x = n_sq_x_1;
n_sq_y = n_sq_y_1;

show_ext_calib;
%show_ext_calib2;

