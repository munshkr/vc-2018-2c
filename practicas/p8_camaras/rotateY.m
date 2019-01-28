function res = rotateY(degs)
   rads = deg2rad(degs);
   res = [ cos(rads) 0 sin(rads);
           0 1 0;
           (-sin(rads)) 0 cos(rads) ];
end