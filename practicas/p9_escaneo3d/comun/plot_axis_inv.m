function plot_axis_inv(R,T,length,ix,iy,iz)

if nargin < 3
    length = 2;
    ix = 1; iy = 2; iz = 3;
end
if nargin < 4
    ix = 1; iy = 2; iz = 3;
end

T = reshape(T,3,1);

% v1 = R'*((c+e1)-T); % origin is (0,0,0), e1 = (1,0,0)

R = R';
v1 = R(:,1);
v2 = R(:,2);
v3 = R(:,3);

v1 = length*v1/norm(v1);
v2 = length*v2/norm(v2);
v3 = length*v3/norm(v3);

% O = R'*(c-T); % origin is (0,0,0)
O = R*-T;

line_width = 2; 

% plot X in blue
P = [O O+v1]';
plot3(P(:,ix),P(:,iy),P(:,iz),'b','LineWidth', line_width)

% plot Y in green
P = [O O+v2]';
plot3(P(:,ix),P(:,iy),P(:,iz),'g','LineWidth', line_width)

% plot Z in red
P = [O O+v3]';
plot3(P(:,ix),P(:,iy),P(:,iz),'r','LineWidth', line_width)

% plot center in red
%P = O';
%plot3(P(:,ix),P(:,iy),P(:,iz),'ko','MarkerFaceColor', 'red')

% 'MarkerEdgeColor','red','
% 'MarkerFaceColor','black



end