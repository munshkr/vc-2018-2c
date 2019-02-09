% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly executed under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 820.413204284246035 ; 819.051132395789523 ];

%-- Principal point:
cc = [ 595.716210500974171 ; 350.091829424128605 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ 0.097893002824471 ; -0.177968664992772 ; -0.008889928482164 ; 0.001773848395631 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 4.021278127610942 ; 3.001746478509199 ];

%-- Principal point uncertainty:
cc_error = [ 4.924104950231461 ; 5.362381268811578 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.012665911964545 ; 0.040245152071501 ; 0.001923310596509 ; 0.002488586962824 ; 0.000000000000000 ];

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
omc_1 = [ -2.530296e+00 ; -5.298217e-02 ; -5.120599e-01 ];
Tc_1  = [ -4.346659e+02 ; 3.374887e+02 ; 1.257002e+03 ];
omc_error_1 = [ 6.071778e-03 ; 2.882003e-03 ; 8.735744e-03 ];
Tc_error_1  = [ 7.568289e+00 ; 7.888237e+00 ; 6.451455e+00 ];

%-- Image #2:
omc_2 = [ -2.559879e+00 ; -1.135298e-01 ; -3.417188e-01 ];
Tc_2  = [ -5.116564e+02 ; 2.980449e+02 ; 1.293839e+03 ];
omc_error_2 = [ 5.746343e-03 ; 2.583529e-03 ; 8.734436e-03 ];
Tc_error_2  = [ 7.799635e+00 ; 8.164234e+00 ; 6.747076e+00 ];

%-- Image #3:
omc_3 = [ -2.584936e+00 ; -1.725443e-01 ; -1.311138e-01 ];
Tc_3  = [ -5.861983e+02 ; 2.496818e+02 ; 1.353866e+03 ];
omc_error_3 = [ 5.552912e-03 ; 2.397363e-03 ; 8.655897e-03 ];
Tc_error_3  = [ 8.172890e+00 ; 8.641567e+00 ; 7.190551e+00 ];

%-- Image #4:
omc_4 = [ -2.594486e+00 ; -2.541586e-01 ; 1.676791e-01 ];
Tc_4  = [ -6.682476e+02 ; 1.748593e+02 ; 1.455476e+03 ];
omc_error_4 = [ 5.486084e-03 ; 2.575287e-03 ; 8.777404e-03 ];
Tc_error_4  = [ 8.784737e+00 ; 9.505219e+00 ; 7.578020e+00 ];

%-- Image #5:
omc_5 = [ -2.571836e+00 ; -3.376278e-01 ; 4.845451e-01 ];
Tc_5  = [ -7.217557e+02 ; 9.122179e+01 ; 1.579312e+03 ];
omc_error_5 = [ 5.805664e-03 ; 3.114564e-03 ; 8.874466e-03 ];
Tc_error_5  = [ 9.554477e+00 ; 1.047175e+01 ; 7.374883e+00 ];

%-- Image #6:
omc_6 = [ -2.509932e+00 ; -4.239426e-01 ; 8.205414e-01 ];
Tc_6  = [ -7.376999e+02 ; 2.282063e+00 ; 1.721637e+03 ];
omc_error_6 = [ 6.315611e-03 ; 3.864706e-03 ; 8.513730e-03 ];
Tc_error_6  = [ 1.049748e+01 ; 1.140500e+01 ; 6.906940e+00 ];

%-- Image #7:
omc_7 = [ -2.339345e+00 ; -5.540150e-01 ; 1.323929e+00 ];
Tc_7  = [ 3.905718e+01 ; -2.196081e+02 ; 2.293733e+03 ];
omc_error_7 = [ 7.159805e-03 ; 3.783583e-03 ; 8.103051e-03 ];
Tc_error_7  = [ 1.383150e+01 ; 1.467813e+01 ; 6.736277e+00 ];

%-- Image #8:
omc_8 = [ -2.419425e+00 ; -5.037658e-01 ; 1.119751e+00 ];
Tc_8  = [ -6.461814e+01 ; -2.391320e+02 ; 2.296438e+03 ];
omc_error_8 = [ 7.379472e-03 ; 3.542541e-03 ; 8.226506e-03 ];
Tc_error_8  = [ 1.378622e+01 ; 1.477885e+01 ; 7.131418e+00 ];

%-- Image #9:
omc_9 = [ -2.506359e+00 ; -4.301246e-01 ; 8.284420e-01 ];
Tc_9  = [ -2.101529e+02 ; -2.514937e+02 ; 2.275605e+03 ];
omc_error_9 = [ 7.535089e-03 ; 3.280482e-03 ; 8.875314e-03 ];
Tc_error_9  = [ 1.362191e+01 ; 1.485752e+01 ; 7.977365e+00 ];

%-- Image #10:
omc_10 = [ -2.563870e+00 ; -3.523341e-01 ; 5.310663e-01 ];
Tc_10  = [ -3.501346e+02 ; -2.458586e+02 ; 2.227191e+03 ];
omc_error_10 = [ 7.470369e-03 ; 3.023465e-03 ; 9.909081e-03 ];
Tc_error_10  = [ 1.332986e+01 ; 1.482326e+01 ; 9.150935e+00 ];

%-- Image #11:
omc_11 = [ -2.593166e+00 ; -2.655166e-01 ; 2.093809e-01 ];
Tc_11  = [ -4.849811e+02 ; -2.198868e+02 ; 2.147205e+03 ];
omc_error_11 = [ 7.649907e-03 ; 2.565038e-03 ; 1.063199e-02 ];
Tc_error_11  = [ 1.288140e+01 ; 1.450324e+01 ; 1.025905e+01 ];

%-- Image #12:
omc_12 = [ -2.591129e+00 ; -1.951145e-01 ; -4.354787e-02 ];
Tc_12  = [ -5.744497e+02 ; -1.861991e+02 ; 2.066861e+03 ];
omc_error_12 = [ 7.888445e-03 ; 2.185981e-03 ; 1.068748e-02 ];
Tc_error_12  = [ 1.243173e+01 ; 1.396823e+01 ; 1.046980e+01 ];

%-- Image #13:
omc_13 = [ -2.572948e+00 ; -1.356285e-01 ; -2.527241e-01 ];
Tc_13  = [ -6.350622e+02 ; -1.501613e+02 ; 1.990583e+03 ];
omc_error_13 = [ 7.669168e-03 ; 2.261378e-03 ; 1.055447e-02 ];
Tc_error_13  = [ 1.199871e+01 ; 1.340568e+01 ; 1.023457e+01 ];

%-- Image #14:
omc_14 = [ -2.548923e+00 ; -8.866909e-02 ; -4.148384e-01 ];
Tc_14  = [ -6.726987e+02 ; -1.174912e+02 ; 1.926288e+03 ];
omc_error_14 = [ 7.416502e-03 ; 2.527470e-03 ; 1.025766e-02 ];
Tc_error_14  = [ 1.164353e+01 ; 1.294775e+01 ; 9.959788e+00 ];

