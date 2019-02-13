function [X,L] = estimateLaserPlane(planePoints)
    % Estima los parámetros del plano del láser X = [nt; d] para un conjunto
    % de puntos 3D planePoints representados por un vector tipo celda, donde
    % para cada i planePoints{i} contiene los puntos que pertenecen al plano
    % de luz láser proyectado en un tablero de ajedrez i.
    % Además devuelve L el conjunto de todos los puntos planePoints

    % Concatenamos todos los puntos en 
    todosLosPuntos=[];
    for i = 1:length(planePoints)
        puntosImagenesi = transpose([planePoints{i}; ones(1,length(planePoints{i}))]);
        todosLosPuntos = [ todosLosPuntos;  puntosImagenesi];
    end
    
    % Buscamos el plano que mejor aproxima los puntos
    % La normal de este plano va a indicar la dimension de menor
    % varianza de los datos
    % O sea, x es la normal del plano <=> arg min x (||A x|) con ||x|| = 1
    % O sea, queremos el ultimo vector de la base ortonormal que nos da SVD
    [~,~,Vt] = svd(todosLosPuntos);
    % la normalizamos
    X = Vt(:, 4)/norm(Vt(1:3,4));
    L = transpose(todosLosPuntos);
end