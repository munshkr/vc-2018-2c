function checkerboardPlane = estimateCheckerboardPlane(R,T)
    % Devuelve un vector checkerboardPlane = [nt; d] en coordenadas
    % de la imagen (R,t) para el tablero de ajedrez en el mundo

    % calculamos la normal al checkerboard
    nt = cross(R(:,1),R(:,2));
    
    % calculamos un punto del checkerboard
    d = -(nt'*(R*[1;1;0]+T));
    
    % el plano del checkerboard se ve representado por estos dos valores
    checkerboardPlane = [nt; d];
    return;
end