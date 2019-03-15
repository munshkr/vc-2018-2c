%% Load calibration from file
set_paths;
load(fullfile(turntable_images_path,'CameraPose.mat'),'worldRotation','worldTranslation');
load(calib_resuls_filename,'KK', 'kc')
load([turntable_images_path '/../TurntableCalib.mat'],'angs');
load([laser_color_images_path '/../LaserPlane_Results.mat'],'laserPlane');

%% Display all the cameras, and world frame coordinates

