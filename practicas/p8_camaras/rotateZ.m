function res = rotateZ(degs)
   rads = deg2rad(degs);
   res = [ cos(rads) (-sin(rads)) 0;
           sin(rads) cos(rads) 0;
           0 0 1 ];
end