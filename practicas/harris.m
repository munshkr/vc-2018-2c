limiteR = 1000000;
tamanio_gauss = 5;

Gx = [1   2   1;0   0   0;-1  -2  -1];
Gy = [-1   0   1;-2   0   2; -1   0   1];
Gxy = Gx .* Gy;
gauss = fspecial("gaussian", [tamanio_gauss tamanio_gauss]);


I = imread('img/test.png');

filas = size(I, 1);
columnas = size(I, 2);

% 1) Calculamos derivadas de I en x y en y
Ix = conv2(Gx, I);
Iy = conv2(Gy, I);


% 2) Generamos Ix2, Iy3, Ixy
Ix2 = Ix .^ 2;
Iy2 = Iy .^ 2;
Ixy = Ix .* Iy;

% 3) Convolucionamos con una gaussiana
Sx2 = conv2(Ix2, gauss);
Sy2 = conv2(Iy2, gauss);
Sxy = conv2(Ixy, gauss);

output = zeros(filas, columnas);
for x=1:filas,
   for y=1:columnas,
       % 4) Calculamos la matriz M
       H = [Sx2(x, y) Sxy(x, y); Sxy(x, y) Sy2(x, y)];
       
       % 5) A partir de M calculamos R
       R = det(H) - k * (trace(H) ^ 2);
       
       % 6) Verificamos si R supera el limite
       if (R > limiteR)
          output(x, y) = R; 
       end
   end
end

figure, imshow(I);
figure, imshow(output);
imwrite(output,'img/testHarris.jpg');
