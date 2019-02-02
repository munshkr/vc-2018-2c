%% Perform camera calibration using Bouguet Calibration Toolbox
% http://www.vision.caltech.edu/bouguetj/calib_doc/
% http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/example.html

!git clone https://github.com/fragofer/TOOLBOX_calib

set_paths;

cd(calib_images_path);

addpath('TOOLBOX_calib-3DP');

calib



% Click on the Image names button
% Click on the Extract grid corners button
% Size dX of each square along the X direction = 20mm (check in CalibResults.mat)
% Size dY of each square along the X direction = 20mm (check in CalibResults.mat)
% After corner extraction, click on the button Calibration
% Click on Reproject on images in the Camera calibration tool to show the reprojections of the grids onto the original images
% Click on Show Extrinsic in the Camera calibration tool
% Click on Analyse error to view the new reprojection error 
% Click on Save

