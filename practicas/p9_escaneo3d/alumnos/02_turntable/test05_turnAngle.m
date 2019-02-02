%%
load([turntable_images_path '/ChessboardCorners.mat'],'Rt', 'Tt');

%%
N = length(Rt);
angs = zeros(N,1);
angs_raw = zeros(N,1);
diff_angs = zeros(N-1,1);

for i = 1:N

    R = Rt{i};
    T = Tt{i};
    
    if i==1
        R0 = R;
        T0 = T;
    end
    
    Ra = R0' * R;
    ca = Ra(1,1); sa = Ra(2,1);
    a = atan2(sa,ca);
    angs(i) = a;
    angs_raw(i) = a;
    
    if i > 1
        da = abs(angs(i) - angs(i-1));
        
        while da > pi/4
            angs(i) = -pi + angs(i);   
            da = abs(angs(i) - angs(i-1));
        end
        
        diff_angs(i) = da;
    end
end

angs(1) = 0;
diff_angs(1) = [];

%
figure
plot(angs/pi*180,'b.-')
grid on
title('angs')

figure
plot(angs_raw/pi*180,'b.-')
grid on
title('angs raw')

figure
plot(diff_angs/pi*180,'b.-')
grid on
title('diffs')

save([turntable_images_path '/../TurntableCalib.mat'],'angs');
