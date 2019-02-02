function plot_axis_img(R,T,KK,Kc,length)

if ~exist('length','var')
    length = 30;
end

v1 = R(:,1); v2 = R(:,2);
%v3 = cross(v1,v2);
v3 = R(:,3);

v1 = length*v1/norm(v1);
v2 = length*v2/norm(v2);
v3 = length*v3/norm(v3);

O = reshape(T,3,1);

width = 2; 
% plot X in red
P = [O O+v1];
P = KK*P;
P = P ./ P(3,:);
plot(P(1,:),P(2,:),'r','LineWidth', width)

% plot Y in green
P = [O O+v2];
P = KK*P;
P = P ./ P(3,:);
plot(P(1,:),P(2,:),'g','LineWidth', width)

% plot Z in blue
P = [O O+v3];
P = KK*P;
P = P ./ P(3,:);
plot(P(1,:),P(2,:),'b','LineWidth', width)


% plot center in red
P = O;
P = KK*P;
P = P ./ P(3,:);
plot(P(1,:),P(2,:),'ko','MarkerFaceColor', 'red')

end