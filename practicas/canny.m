lena = imread('img/lena256.png');
FilterRoberts1 = [1 0 0; 0 -1 0; 0 0 0];
FilterRoberts2 = [0 1 0; -1 0 0; 0 0 0];
Ix = conv2(lena, FilterRoberts1, "same");
Iy = conv2(lena, FilterRoberts2, "same");
intensidad_gradiente = sqrt(Ix .^ 2 + Iy .^ 2);
orientacion_gradiente = atan2(Ix ,Iy);
Ig = intensidad_gradiente;
Io = orientacion_gradiente;
In=zeros(size(Ig));
[filas, columnas] = size(Ig);
for i = 2 : filas-1
    if (0 == mod(i,100))
        txt = sprintf('Fila %f', i)
    end
    for j = 2 : columnas-1
        anguloEnDeg = Io(i,j) * 180/pi;
        if (anguloEnDeg < 45 && anguloEnDeg > -45)
            if Ig(i,j)>max(Ig(i-1,j),Ig(i+1,j))
                In(i,j)=Ig(i,j);
            end
        end
        if (anguloEnDeg > 45 && anguloEnDeg < 135)
            if Ig(i,j)>max(Ig(i+1,j-1),Ig(i-1,j+1))
                In(i,j)=Ig(i,j);
            end
        end
        if (anguloEnDeg > 135 || anguloEnDeg < -135)
            if Ig(i,j)>max(Ig(i,j-1),Ig(i,j+1))
                In(i,j)=Ig(i,j);
            end
        end
        if (anguloEnDeg > -135 && anguloEnDeg < -45)
            if Ig(i,j)>max(Ig(i-1,j-1),Ig(i+1,j+1))
                In(i,j)=Ig(i,j);
            end
        end
    end
end
imshow(In)