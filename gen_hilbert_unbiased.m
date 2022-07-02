function [] = gen_hilbert_unbiased(N, iter, folder)

% Generador de estados puros de 'n' qubits aleatorios.
%
%  |Psi> = sum_[ijkl...={0,1}] C_{ijk...}|i>|j>|k>...
%
% Y cálculo de las matrices densidad, descomp. y rango de Schmidt
% SVD/Schmidt Decomp. Sólo funciona para n = 2
% ---------------------------------------------------------------

n = 2;                        % Qubits
pc_test = 0.5;                % Porcentaje de test
N_test = floor(pc_test*N);    % Numero de estados a generar para test
N_train = N-N_test;           % Numero de estados a generar para train

N_not = N_test+N_train;       % N. no entrelazados. La mitad en proporción de cada uno

% Init
schmidt_rank_train = zeros(N_train,1);
schmidt_rank_test = zeros(N_test,1);
schmidt_rank_not = zeros(N_test,1);

% Generación

C_train = nsphere(2^(n+1)-1, N_train);
C_train = transpose(C_train(1:2^n,:)+1i*C_train((2^n+1):end,:));

for i = 1:N_train       % Iter. para cada estado
    
    M = [C_train(i,1), C_train(i,2); C_train(i,3), C_train(i,4)];
    sv_train(i,:) = svd(M)'.^2;
    schmidt_rank_train(i) = sum(sv_train(i,:) > eps);
    
end

C_test = nsphere(2^(n+1)-1, N_test);
C_test= transpose(C_test(1:2^n,:)+1i*C_test((2^n+1):end,:));

for i = 1:N_test       % Iter. para cada estado
    
    M = [C_test(i,1), C_test(i,2); C_test(i,3), C_test(i,4)];
    sv_test(i,:) = svd(M)'.^2;
    schmidt_rank_test(i) = sum(sv_test(i,:) > eps);
    
end
%%
C_not = zeros(N_not, 2^n);

%Creamos los estados no entrelazados de forma independiente
for i = 1:N_not
    
    magnitude_1 = rand([2,1],"double");  % Un estado de 1 qubit
    phase_1 = 2*pi*rand([2,1],"double");
    norm_1 = sqrt(sum(magnitude_1.^2));
    magnitude_1 = magnitude_1./norm_1;
    
    magnitude_2 = rand([2,1],"double");  % El otro estado de 1 qubit
    phase_2 = 2*pi*rand([2,1],"double");
    norm_2 = sqrt(sum(magnitude_2.^2));
    magnitude_2 = magnitude_2./norm_2;
    
    C_1 = magnitude_1.*exp(1i*phase_1);
    C_2 = magnitude_2.*exp(1i*phase_2);
    
    C_not(i,:) = kron(C_1, C_2);       % Guarda en un vector los coef complejos
    % de cada estado.
    
    M = [C_not(i,1), C_not(i,2); C_not(i,3), C_not(i,4)];
    sv_not(i,:) = svd(M)'.^2;
    schmidt_rank_not(i) = sum(sv_not(i,:) > eps);
end

% Se añaden los no entrelazados de forma aleatoria a los train y test.
% Hasta que no sepa como le voy a introducir los datos a la red, esto no
% puedo hacerlo

% Se unen los entrelazados y no entrelazados en una misma matriz y se
% permuta aleatoriamente

C_test = [C_test; C_not(1:N_test,:)];
C_train = [C_train; C_not((N_test+1):end,:)];
schmidt_rank_test = [schmidt_rank_test; schmidt_rank_not(1:N_test)];
schmidt_rank_train = [schmidt_rank_train; schmidt_rank_not((N_test+1):end)];
sv_test = [sv_test; sv_not(1:N_test,:)];
sv_train = [sv_train; sv_not((N_test+1):end,:)];

%Se genera la permutación de las filas que permutar

P_test = randperm(2*N_test);
P_train = randperm(2*N_train);

C_test = C_test(P_test,:);
C_train = C_train(P_train,:);
schmidt_rank_test = schmidt_rank_test(P_test);
schmidt_rank_train = schmidt_rank_train(P_train);
sv_test = sv_test(P_test,:);
sv_train = sv_train(P_train,:);


% Guarda

csvwrite([folder,'test_',int2str(iter),'.csv'], [real(C_test), imag(C_test)]);
csvwrite([folder,'test_sol_',int2str(iter),'.csv'], schmidt_rank_test-1);
csvwrite([folder,'test_sv_',int2str(iter),'.csv'], sv_test);

csvwrite([folder,'train_',int2str(iter),'.csv'], [real(C_train), imag(C_train)]);
csvwrite([folder,'train_sol_',int2str(iter),'.csv'], schmidt_rank_train-1);
csvwrite([folder,'train_sv_',int2str(iter),'.csv'], sv_train);

end