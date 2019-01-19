function [mean_S_T_vector,cov_s_T_vector,R_s_2toT_vector] = Backward(mean_s_T_vector,cov_s_T_vector)
% BACKWARD Computes the  posterior distribution of the hidden state on all
% observations it is done in the following steps:
% First the "Backward inference" calculate the the conditional
% P(X_t-1|x_t,y_1:t-1)

% InitBackward get P(x_T|Y) = N(mean_s_T,cov_s_T)
Mean_s_T  = mean_s_T_vector(:,end);
cov_s_T = cov_s_T_vector(:,:,end);
for i = (T-1):-1:1
    %% Backwar inference
    % Backward inference approximates the conditional by first
    % approximating the joint distribution P(X_t-1,x_t|,y_1:t-1)
    
    % Calcualtion of mean_e_t-1, cov_e_t-1, mean_p_t, cov_p_t, cov_ep
    calculate:
    mean_e_t-1
    cov_e_t-1
    mean_p_t
    cov_p_t
    cov_ep
    % Claculate J_t-1 = C_ep*(C_p_t)^-1
    J_t-1 = cov_ep*(cov_p_t)^-1;
    % desired distribution P(X_t-1,x_t|,y_1:t-1) = N(x_t-1|m,S)
    m = mean_e_t-1 + J_t-1*(x_t-mean_p_t);
    S = cov_e_t-1 - J_t-1*cov_ep'*(Transpose);
    %% Smoothed state distribution
    % Computes the intregral (7) von turner S-S_ILGP
    % p(x_t-1|y_1:T)= gamma(x_t-1) = N(x_t-1|mean_s_t-1,cov_s_t-1)
    mean_s_t-1 = mean_e_t-1 + J_t-1 * (mean_s_t - mean_p_t)
    cov_s_t-1 = cov_e_t-1 + J_t-1 * (cov_s_t-cov_p_t) * J_t-1'(Transpose)

    %% Cross-covariance for learning 
    % cov(x_t-1,x_t|Y) = J_t-1*C_s_t
    croscov(:,:,__t-1_t) = J_t-1*C_s_t;
end
outputArg1 = inputArg1;
outputArg2 = inputArg2;
end

