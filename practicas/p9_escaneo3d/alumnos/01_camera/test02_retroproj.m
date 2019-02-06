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

%% Reproject calibration corners to images

Rms = zeros(n_ima,1);
for i=1:n_ima
    %%    
    us = iPt{i};
    ps = wPt{i};
    R = Rc{i};
    T = Tc{i};

    [~, n] = size(ps);
    Xc = R*ps + repmat(T,1,n);
    Xc = Xc./Xc(3,:);
    Xi = KK * Xc;
    r = sqrt(sum(dot(us - Xi(1:2,:),us - Xi(1:2,:)))/n/2);
    Rms(i) = r;
end
%
mean_rms = mean(Rms);
std_rms = std(Rms);
fprintf('Mean RMS %f\n',mean_rms);
fprintf('Std  RMS %f\n',std_rms);


