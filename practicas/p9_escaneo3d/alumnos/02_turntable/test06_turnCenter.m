%%
%load([turntable_images_path '/ChessboardCorners.mat'],'Rt', 'Tt');
%load('/Users/pachi/workspace/posdoc/3DP2018/data/ChessboardCorners.mat','R', 'T');
load('/Users/pachi/workspace/posdoc/3DP2018/data/ChessboardCorners.mat');
%Rt = R; Tt = T; clear R T;
load(calib_resuls_filename,'KK', 'kc')
%%
N = length(iPt);
A = []; b = [];

for i=1:N
    %R = Rt{i};
    %T = Tt{i};
    us = iPt{i};
    ps = wPt{i};
    [R,T] = estimateExtrinsics(us,ps,KK,kc);
    
    if i==1
        R0 = R;
    end
        
    
    % In order to fix qz = 0, we skip the third column in the rotation matrices reducing A to 5 columns
    A = [A;  R(:,1:2) -eye(3)];
    b = [b; -T];
end

%[~,~,V] = svd(A|b);
%X = V(:,9);
X = A \ b;

%
% Remember that the world rotation is the rotation corresponding to the first image,
% and the world translation is the location of the turntable center in 3D with respect
% to the camera coordinate system.
% 
centerCoord = [X(1:2); 0]; % q
centerCoord3D = X(3:5); % Q

worldRotation = R0 %Rt{1}
worldTranslation = centerCoord3D
turntableCenter = centerCoord

save(fullfile(turntable_images_path,'CameraPose.mat'),'worldRotation','worldTranslation','turntableCenter');
