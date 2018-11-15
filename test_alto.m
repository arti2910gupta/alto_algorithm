% load foursquare data and evaluate the forecasting error
clc
clear
close all

addpath(genpath('./'))
load 'foursquare.mat' % 56* 1200 * 15

nLag = 2;

[X,Y] = series_to_samples(series_obs, nLag);

train_ratio = 0.8;
num_train = floor(length(X) * train_ratio);

X_train = X(1:num_train);
Y_train = Y(1:num_train);
X_test = X(num_train+1:end);
Y_test = Y(num_train+1:end);
%% Laplacian regularizer
mu = 1e-2;  % local consistency parameter ( for Laplacian regularizer)
sim = sim_friend;
sim = sim/(max(sim(:)));       % The goal is to balance between two measures
H = chol(eye(size(sim)) + mu*sim);

nInit = 2;
sz_batch = 1;
nRank = 2;
[ W, objs, ts ] = online_tensor_learn(X_train,Y_train, H,nInit, sz_batch, nRank);
test_err_svd = least_square_error(W, X_test, Y_test);




