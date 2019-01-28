function res = rotateX(degs)
   rads = deg2rad(degs);
   res = [ 1 0 0;
           0 cos(rads) (-sin(rads));
           0 sin(rads) cos(rads) ];
end