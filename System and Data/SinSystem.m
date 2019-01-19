function [X_danach,Y] = SinSystem(X0,test_observation,Varianz_syst,Varianz_meassurement)
%   SinSystem  [X_t+1,Y] = SinSystem(a,b,T,Q,R)
%   Meassures X_t+1 = 3*3*sin(3*X_t) according Y = X_t+1 
%   with given the domain (a,b) and 
%   take T messureamets of it. 
%   with Normal distributed Noise of Varianz
%   Q for the System and R for
%   meassurement
%

%% Definitionsbereich desde Systems

%% Noise 
% Noise from system F
varianz_epsilon = Varianz_syst;

% Noise in Antwort system G
varianz_v = Varianz_meassurement;

%% Systembereich
X_danach = zeros(size(X0,1),test_observation);
X_danach(1) = X0;
Y = zeros(1,test_observation);
Y(1) = X0 + sqrt(varianz_v)*(-1+2*rand(1));
for i = 1:test_observation-1
    epsilon = sqrt(varianz_epsilon)*(-1+2*rand(size(X0)));
    X_danach(i+1) = 3*sin(3*X_danach(i))+epsilon;
    v = sqrt(varianz_v)*(-1+2*rand(1));
    Y(i+1) = X_danach(i+1) + v;
end


% % Noise from system F
% varianz_epsilon = Q;
% epsilon = sqrt(varianz_epsilon)*rand(size(X_vorher));
% % Noise in Antwort system G
% varianz_v = R;
% v = sqrt(varianz_v)*rand(size(X_vorher));
% % System Function
% X_danach = 3*sin(3*X_vorher)+epsilon;
% % System Antwort
% Y = X_danach + v;


% In case you want to save the Data to Workspace (uncoment this part)
% save('TrainingData','Y','X_vorher')
end