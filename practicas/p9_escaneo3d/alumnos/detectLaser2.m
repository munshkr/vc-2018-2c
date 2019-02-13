function [Lx,Ly] = detectLaser2(inputImage,mascara)
    % Dada una imagen a color inputImage detecta los píxeles del láser
    % y devuelve sus coordenadas en los vectores [Lx,Ly]
    % Opcionalmente, puede tomar una imagen binaria mask que indica
    % la región de búsqueda
    
    
    
    %%% Segun el curso los pasos a seguir son:
    % 1. Smooth the input image to remove noise with the function 'filterImage' from the previous section.
    % 2. Isolate the red color by comparing the color channels as: v = r - (g+b)/2,
    %    v will have a high value for red pixels.
    % 3. Search the average maximum value of rows: avgMax = (max(row[0]) + max(row[1]) + ... + max(row(N)))/N
    % 4. Apply a threshold to all values much lower than the average maximum: e.g. if (pixel<0.8*avgMax) pixel = 0;
    % 5. Search the maximum of each row and set that pixel equal to 255
    
    n = size(inputImage);
   
    % 1. Smooth the input image to remove noise with the function 'filterImage' from the previous section.
    % aplicamos mascara si es que pasaron algo por la variable
    inputImage = double(inputImage);
    if exist('mascara','var')
        for i = 1:3
            inputImage(:,:,i) = inputImage(:,:,i) .* mascara;
        end
    end
       
    
    % blureamos la imagen para remover ruido
    inputImageBlureada = zeros(n);
    kernel_gaussiano =[1,4,6,4,1; 4,16,24,16,4; 6,24,36,24,6; 4,16,24,16,4; 1,4,6,4,1]/256;
    for i = 1:3
        inputImageBlureada(:,:,i) = imfilter(inputImage(:,:,i), kernel_gaussiano, 'same');
    end

    % 2. Isolate the red color by comparing the color channels as: v = r - (g+b)/2,
    %    v will have a high value for red pixels.
    v = inputImageBlureada(:,:,1) -(inputImageBlureada(:,:,2)+inputImageBlureada(:,:,3))/2;

    % 3. Search the average maximum value of rows: avgMax = (max(row[0]) + max(row[1]) + ... + max(row(N)))/N
    avgRowMax = mean(max(v'));

    % 4. Apply a threshold to all values much lower than the average maximum: e.g. if (pixel<0.8*avgMax) pixel = 0;
    v(v < 1.5*avgRowMax) = 0;
    
    % 5. Search the maximum of each row and set that pixel equal to 255
    % (este paso en realidad es encontrar que valores son distintos de
    % cero)
    [Ly, Lx]=find(v>0);
end