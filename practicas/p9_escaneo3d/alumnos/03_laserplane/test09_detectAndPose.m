%%
% For each chessboard which intersects the laserplane, use the function  
% 02_turntable/test04_detectAndPose.m and save the data in LaserChessboardCorners.mat

% save([laser_color_images_path '/../LaserChessboardCorners.mat'],'iPt','wPt','Rl','Tl');

%%
% If you used the same images for calibration (with Bouguet's toolbox), run
% this code
n_ima = struct2array(load(calib_resuls_filename,'n_ima'));

Rl = cell(n_ima,1);
Tl = cell(n_ima,1);
iPt = cell(n_ima,1);
wPt = cell(n_ima,1);
for i=1:n_ima
    Rl{i} = struct2array(load(calib_resuls_filename,['Rc_' num2str(i)]));
    Tl{i} = struct2array(load(calib_resuls_filename,['Tc_' num2str(i)]));
    iPt{i} = struct2array(load(calib_resuls_filename,['x_' num2str(i)]));
    wPt{i} = struct2array(load(calib_resuls_filename,['X_' num2str(i)]));
end


save([laser_color_images_path '/../LaserChessboardCorners.mat'],'iPt','wPt','Rl','Tl');