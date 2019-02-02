% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly executed under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 819.407052569545840 ; 819.949325826004838 ];

%-- Principal point:
cc = [ 590.565014688615065 ; 353.699302751974642 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ 0.098476656142129 ; -0.183130526770840 ; -0.007835675123168 ; 0.000203284611414 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 0.965580147353912 ; 0.723790977627717 ];

%-- Principal point uncertainty:
cc_error = [ 1.177212016294000 ; 1.303641697859439 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.003068475197987 ; 0.009753683326875 ; 0.000462089087403 ; 0.000588413424775 ; 0.000000000000000 ];

%-- Image size:
nx = 1152;
ny = 768;


%-- Various other variables (may be ignored if you do not use the Matlab Calibration Toolbox):
%-- Those variables are used to control which intrinsic parameters should be optimized

n_ima = 14;						% Number of calibration images
est_fc = [ 1 ; 1 ];					% Estimation indicator of the two focal variables
est_aspect_ratio = 1;				% Estimation indicator of the aspect ratio fc(2)/fc(1)
center_optim = 1;					% Estimation indicator of the principal point
est_alpha = 0;						% Estimation indicator of the skew coefficient
est_dist = [ 1 ; 1 ; 1 ; 1 ; 0 ];	% Estimation indicator of the distortion coefficients


%-- Extrinsic parameters:
%-- The rotation (omc_kk) and the translation (Tc_kk) vectors for every calibration image and their uncertainties

%-- Image #1:
omc_1 = [ -1.964523e+00 ; -2.056160e+00 ; -1.010945e+00 ];
Tc_1  = [ -3.396876e+02 ; -1.746089e+02 ; 9.437478e+02 ];
omc_error_1 = [ 7.837118e-04 ; 1.293044e-03 ; 2.505540e-03 ];
Tc_error_1  = [ 1.370925e+00 ; 1.576730e+00 ; 1.273967e+00 ];

%-- Image #2:
omc_2 = [ -1.893674e+00 ; -2.065086e+00 ; -8.363306e-01 ];
Tc_2  = [ -4.153264e+02 ; -2.132997e+02 ; 9.864925e+02 ];
omc_error_2 = [ 7.950154e-04 ; 1.217063e-03 ; 2.352616e-03 ];
Tc_error_2  = [ 1.436302e+00 ; 1.681401e+00 ; 1.390934e+00 ];

%-- Image #3:
omc_3 = [ -1.808044e+00 ; -2.063341e+00 ; -6.463607e-01 ];
Tc_3  = [ -4.899782e+02 ; -2.618520e+02 ; 1.045854e+03 ];
omc_error_3 = [ 8.092776e-04 ; 1.191088e-03 ; 2.198155e-03 ];
Tc_error_3  = [ 1.523180e+00 ; 1.814644e+00 ; 1.542192e+00 ];

%-- Image #4:
omc_4 = [ -1.679271e+00 ; -2.042060e+00 ; -3.910003e-01 ];
Tc_4  = [ -5.724300e+02 ; -3.370982e+02 ; 1.147119e+03 ];
omc_error_4 = [ 9.099704e-04 ; 1.197647e-03 ; 2.047677e-03 ];
Tc_error_4  = [ 1.664424e+00 ; 1.994900e+00 ; 1.643292e+00 ];

%-- Image #5:
omc_5 = [ -1.534024e+00 ; -1.997258e+00 ; -1.356817e-01 ];
Tc_5  = [ -6.268488e+02 ; -4.214526e+02 ; 1.271298e+03 ];
omc_error_5 = [ 1.131619e-03 ; 1.233590e-03 ; 1.878121e-03 ];
Tc_error_5  = [ 1.848571e+00 ; 2.164735e+00 ; 1.582423e+00 ];

%-- Image #6:
omc_6 = [ -1.369159e+00 ; -1.927100e+00 ; 1.191385e-01 ];
Tc_6  = [ -6.434952e+02 ; -5.111058e+02 ; 1.413820e+03 ];
omc_error_6 = [ 1.306580e-03 ; 1.258269e-03 ; 1.695133e-03 ];
Tc_error_6  = [ 2.083239e+00 ; 2.340360e+00 ; 1.515572e+00 ];

%-- Image #7:
omc_7 = [ -1.098816e+00 ; -1.783624e+00 ; 4.744184e-01 ];
Tc_7  = [ 1.381987e+02 ; -7.344497e+02 ; 1.983165e+03 ];
omc_error_7 = [ 1.153991e-03 ; 1.321167e-03 ; 1.817612e-03 ];
Tc_error_7  = [ 2.957896e+00 ; 3.106173e+00 ; 1.586058e+00 ];

%-- Image #8:
omc_8 = [ -1.210231e+00 ; -1.848326e+00 ; 3.328496e-01 ];
Tc_8  = [ 3.468153e+01 ; -7.538588e+02 ; 1.985924e+03 ];
omc_error_8 = [ 1.208345e-03 ; 1.355350e-03 ; 1.846783e-03 ];
Tc_error_8  = [ 2.947643e+00 ; 3.135288e+00 ; 1.632916e+00 ];

%-- Image #9:
omc_9 = [ -1.362171e+00 ; -1.926590e+00 ; 1.229862e-01 ];
Tc_9  = [ -1.105363e+02 ; -7.659911e+02 ; 1.965055e+03 ];
omc_error_9 = [ 1.279219e-03 ; 1.435781e-03 ; 1.938826e-03 ];
Tc_error_9  = [ 2.905862e+00 ; 3.179006e+00 ; 1.764137e+00 ];

%-- Image #10:
omc_10 = [ -1.509205e+00 ; -1.988872e+00 ; -1.023195e-01 ];
Tc_10  = [ -2.503066e+02 ; -7.599575e+02 ; 1.916248e+03 ];
omc_error_10 = [ 1.287835e-03 ; 1.501574e-03 ; 2.170804e-03 ];
Tc_error_10  = [ 2.833997e+00 ; 3.232735e+00 ; 2.037323e+00 ];

%-- Image #11:
omc_11 = [ -1.659390e+00 ; -2.036039e+00 ; -3.588430e-01 ];
Tc_11  = [ -3.849024e+02 ; -7.337505e+02 ; 1.836368e+03 ];
omc_error_11 = [ 1.229766e-03 ; 1.530204e-03 ; 2.517666e-03 ];
Tc_error_11  = [ 2.729781e+00 ; 3.266504e+00 ; 2.408742e+00 ];

%-- Image #12:
omc_12 = [ -1.770501e+00 ; -2.056881e+00 ; -5.713583e-01 ];
Tc_12  = [ -4.745545e+02 ; -6.999307e+02 ; 1.756694e+03 ];
omc_error_12 = [ 1.266341e-03 ; 1.502284e-03 ; 2.679018e-03 ];
Tc_error_12  = [ 2.632596e+00 ; 3.198629e+00 ; 2.519524e+00 ];

%-- Image #13:
omc_13 = [ -1.858165e+00 ; -2.062990e+00 ; -7.551042e-01 ];
Tc_13  = [ -5.356599e+02 ; -6.640313e+02 ; 1.681784e+03 ];
omc_error_13 = [ 1.306924e-03 ; 1.416055e-03 ; 2.731797e-03 ];
Tc_error_13  = [ 2.541572e+00 ; 3.051926e+00 ; 2.429379e+00 ];

%-- Image #14:
omc_14 = [ -1.923697e+00 ; -2.060231e+00 ; -9.032369e-01 ];
Tc_14  = [ -5.739694e+02 ; -6.316855e+02 ; 1.618987e+03 ];
omc_error_14 = [ 1.289258e-03 ; 1.355999e-03 ; 2.760888e-03 ];
Tc_error_14  = [ 2.465190e+00 ; 2.902679e+00 ; 2.308748e+00 ];

