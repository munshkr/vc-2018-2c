%%
load([images_path '/ChessboardCorners.mat'],'Rt','Tt','iPt','wPt');
load([images_path '/CameraPose.mat'],'worldRotation', 'worldTranslation');
load(calib_resuls_filename,'KK', 'kc')

%% retroprojection
N = length(Rt);
Rms = zeros(N,1);
reprojectedPoints = [];
worldPoints = [];
for i=1:N
    %%
    us = iPt{i}; % image points
    ps = wPt{i}; % checkerboard points
    R = Rt{i};
    T = Tt{i};

    ps = [ps'; zeros(1,length(ps))];
        
    %pw = worldRotation * ps + worldTranslation;
    
    pc = R * ps + T;
    
    %pw = worldRotation' * pc - worldTranslation;
    
    worldPoints = [worldPoints pw];
    
    %% TODO: check this
    % project 'ps' to the image plane
    pc = R * ps + T;
    
    % apply lens distortion
    if length(kc) > 2
        x = pc(1,:) ./ pc(3,:);
        y = pc(2,:) ./ pc(3,:);
        %
        r2 = x.*x + y.*y;
        cte = 1 + kc(1) .* r2 + kc(2) .* r2 .* r2 + kc(5) .* r2 .*r2 .* r2; % kc(4) == 0
        dx1 = 2.*kc(3).*x.*y + kc(4).*(r2 + 2.*x.*x);
        dx2 = kc(3).*(r2 + 2.*y.*y) + 2.*kc(4).*x.*y;
        u = [];
        u(1,:) = cte.*x + dx1;
        u(2,:) = cte.*y + dx2;
        u(3,:) = 1;
    else
        u = pc;
        u(1,:) = u(1,:) ./ u(3,:);
        u(2,:) = u(2,:) ./ u(3,:);
        u(3,:) = 1;
    end
    
    %%
    % convert 'u' to pixel coords
    u = KK * u;
    
    % save result
    reprojectedPoints = [reprojectedPoints u];

    %%
    % compare with the original image point 'v'
    n = length(us);
    v = us' - u(1:2,:);
    r = sqrt(sum(dot(v,v))/n/2);
    Rms(i) = r;
end
%

mean_rms = mean(Rms);
std_rms = std(Rms);
fprintf('Mean RMS %f\n',mean_rms);
fprintf('Std  RMS %f\n',std_rms);


%%
figure;
scatter3(worldPoints(1,:),worldPoints(2,:),worldPoints(3,:),'+');
xlabel('X'); ylabel('Y'); zlabel('Z');
