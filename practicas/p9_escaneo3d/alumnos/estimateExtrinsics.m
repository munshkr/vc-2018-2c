function [R,T] = estimateExtrinsics(us,ps,KK,kc)
% Estima la rotación R y la traslación T que mapea las coordenadas ps
% del tablero de ajedrez en el mundo a sus proyecciones us en el plano imagen.

%%% PARAMETROS
% ps = coordenadas del tablero de ajedrez en el mundo | worldCoord - input
% us = proyecciones en el plano de la imagen de los puntos ps | imageCoord - input
% KK = matriz de calibración | Matrix3d K - intrinsic
% kc = parámetros de distorsión | Vector5d kc - intrinsic

%%%% En el curso es equivalente a:
%void hw3::estimateExtrinsics(Matrix3d const& K, /* intrinsic */
%                           Vector5d const& kc, /* intrinsic */
%                           QVector const& worldCoord, /* input */
%                           QVector const& imageCoord, /* input */
%                           Matrix3d& R, /* output */
%                           Vector3d& T  /* output */);

% en el curso:
%      p = ps (coordenadas en el mundo)
%      u = us (proyeccion de p)
%      K = KK
p = ps;
u = [us ones(size(us,1),1)];
K = KK;
n = size(us,1)

% Armamos uMoño
uMonio=[];
for i = 1:n
    uMonio = [uMonio; transpose(inv(K)*transpose(u(i)))];
end


uMonio
% A es de 2nx9
A = [];
for i = 1:n
    nuevaFila1 = [0, -p(i,1), uMonio(i,2) * p(i,1), 0, -p(i,2), uMonio(i,2)*p(i,2), 0, -1, uMonio(i,2)];
    nuevaFila2 = [p(i,1), 0, -uMonio(i,1)*p(i,1), p(i,2), 0, -uMonio(i,1)*p(i,2), 1, 0, -uMonio(i,1)];
    A = [A; nuevaFila1; nuevaFila2];
end

% X sombrerito = arg min x (||A x|) con ||x|| = 1
% Por lo tanto, X sombrerito es el vector de norma 1 que menor reduce a ||Ax||
% Pensandolo en terminos de PCA, queremos buscar la direccion de menor
% varianza de los datos
% O sea, queremos el ultimo vector de la base ortonormal que nos da SVD
[U,Sigma,Vt] = svd(A);
    
XSombrerito = Vt(:,end); % El ultimo vector de la base ortonormal de V

%XSombrerito = transpose([r11,r21,r31, r12, r22, r32, Tx, Ty, Tz])
% Ya teniendo XSombrerito, definimos Rsombrerito
% RSombrerito = [transpose([r11, r21, r31]), transpose([r12, r22, r32])]
% TODO: Preguntar, en RSombrerito hay un error en el curso? Lo define como [r21,
% r22, r23], pero en X no estan esos valores

RSombrerito = [XSombrerito(1), XSombrerito(2), XSombrerito(3);
               XSombrerito(4), XSombrerito(5), XSombrerito(6)];
           
TSombrerito = [XSombrerito(7),XSombrerito(8),XSombrerito(9)]; 

s = 2 / (norm(RSombrerito(1))+norm(RSombrerito(2)));

% Calculamos R a partir de S y RSombrerito
R1 = s*RSombrerito(1,:);
R3 = cross(R1, s * RSombrerito(2,:));
R2 = cross(R3, R1);
           

R = [R1 R2 R3];
T = s*TSombrerito;

return;

end