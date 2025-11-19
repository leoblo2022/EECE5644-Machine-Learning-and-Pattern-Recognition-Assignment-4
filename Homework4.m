%% Homework 4: Question 1: Support Vector Machines (SVM) and Multi-Layer Perceptrons (MLP)
clear all; close all;

%% Generate and visualize training and test data
% Parameters
Ntrain = 1000; Ntest  = 10000;       
r_neg  = 2; r_pos  = 4; sigma  = 1;       

% Generate training data
X_train_neg = (r_neg * [cos(rand(Ntrain/2,1)*2*pi - pi),  sin(rand(Ntrain/2,1)*2*pi - pi)]) + sigma * randn(Ntrain/2,2);
y_train_neg = 1 * ones(Ntrain/2,1);

X_train_pos = (r_pos * [cos(rand(Ntrain/2,1)*2*pi - pi),  sin(rand(Ntrain/2,1)*2*pi - pi)]) + sigma * randn(Ntrain/2,2);
y_train_pos = 2 * ones(Ntrain/2,1);

xTrain = [X_train_neg; X_train_pos]; yTrain = [y_train_neg; y_train_pos];
xTrain = xTrain'; yTrain = yTrain';

% Generate test data
X_test_neg  = (r_neg * [cos(rand(Ntest/2,1)*2*pi - pi),  sin(rand(Ntest/2,1)*2*pi - pi)]) + sigma * randn(Ntest/2,2);
y_test_neg  = 1 * ones(Ntest/2,1);

X_test_pos  = (r_pos * [cos(rand(Ntest/2,1)*2*pi - pi),  sin(rand(Ntest/2,1)*2*pi - pi)]) + sigma * randn(Ntest/2,2);
y_test_pos  = 2 * ones(Ntest/2,1);

xTest = [X_test_neg; X_test_pos]; yTest = [y_test_neg; y_test_pos];
xTest = xTest'; yTest = yTest';

N = size(xTrain, 2); idx = randperm(N); % random permutation       
xTrain = xTrain(:, idx);    % shuffle columns of xTrain
yTrain = yTrain(idx);       % shuffle labels the same way

N = size(xTest, 2); idx = randperm(N); % random permutation
xTest = xTest(:, idx);    % shuffle columns of xTrain
yTest = yTest(idx);       % shuffle labels the same way


% Plot training set
figure(1); hold on; 
scatter(X_train_neg(:,1), X_train_neg(:,2), 20, 'b', 'filled');
scatter(X_train_pos(:,1), X_train_pos(:,2), 20, 'r', 'filled');
xlabel('x_1'); ylabel('x_2');
title('Training Set Visualization');
legend('Class -1 (r = 2)', 'Class +1 (r = 4)');
axis equal;


%% MLP Structure
% mlp paramaters
inputDim = 2;     % number of input features (2D)
hiddenDim = 1;   % starting number of perceptrons, will be optimally selected later
outputDim = 2;    % number of classes
learningRate = 0.01;
numEpochs = 5000;

% Initialize network weights
rng(0);
W1 = 0.1 * randn(hiddenDim, inputDim);
b1 = zeros(hiddenDim, 1);
W2 = 0.1 * randn(outputDim, hiddenDim);
b2 = zeros(outputDim, 1);

%% Model Order Selection
Nfolds = 10; % 10-fold cross validation
perceptronList = [1, 2, 5, 10, 15, 20];

% 10-fold cross-validation indices
indices = makeFolds(length(yTrain), Nfolds); % used helper function

meanValError = zeros(1, length(5));

% Try several numbers-of-perceptrons
for M = 1:length(perceptronList)
        numPerceptrons = perceptronList(M);
        valErrors = zeros(1, Nfolds);

        for fold = 1:Nfolds
            % Split train/validation sets
            valIdx = (indices == fold);
            trainIdx = ~valIdx;
            xTr = xTrain(:, trainIdx);
            yTr = yTrain(:, trainIdx);
            xVal = xTrain(:, valIdx);
            yVal = yTrain(:, valIdx);

            %% Model Training 
            [W1, b1, W2, b2] = trainMLP(xTr, yTr, inputDim, numPerceptrons, outputDim, learningRate, numEpochs); % used helper function

            % Evaluate on validation set
            yPredVal = predictMLP(xVal, W1, b1, W2, b2); % used helper function
            valErrors(fold) = mean(yPredVal ~= yVal);
        end 
        meanValError(M) = mean(valErrors);
        fprintf('N=%d, Hidden=%d, Mean Val Error=%.3f\n', Ntrain, numPerceptrons, meanValError(M));
 end 
% Pick best model order
[~, bestIdx] = min(meanValError);
bestHidden = perceptronList(bestIdx);

 % Retrain on full data with best hiddenDim
[W1, b1, W2, b2] = trainMLP(xTrain, yTrain, inputDim, bestHidden, outputDim, learningRate, numEpochs);

 %% Performance Assessment 
% Test on large test set
yPred = predictMLP(xTest, W1, b1, W2, b2);
accuracy = mean(yPred == yTest);
testErrors = mean(yPred ~= yTest);

fprintf('=== Test Accuracy of mlp ===\n');
fprintf('parameters: N=%d, Best number of perceptrons in hidden layer =%d, Test Error=%.3f, Test Accuracy: %.2f%%\n', Ntrain, bestHidden, testErrors*100, accuracy * 100);
fprintf('\n')

%% Visualize Best Performance Results 
figure; hold on; 
xlabel('x_1'); ylabel('x_2')
title('MLP Classification Results');
view(45,25); axis equal;

% Define two distinct green shades
greenShades = [ 0.2 0.8 0.2; 0.0 0.6 0.0];

% Loop over classes and plot correct/incorrect
for k = 1:outputDim
    correctIdx = find(yTest == k & yPred == k);
    incorrectIdx = find(yTest == k & yPred ~= k);

    % Correct: shaded green
    scatter(xTest(1, correctIdx), xTest(2, correctIdx), 20, greenShades(k,:), 'filled');

    % Incorrect: red
    scatter(xTest(1, incorrectIdx), xTest(2, incorrectIdx), 20, 'r', 'filled');
end
legend({'Correct (Class 1)', 'Correct (Class 2)', 'Incorrect'}, 'Location', 'bestoutside');
axis equal;


%% Model Training function 
function [W1, b1, W2, b2] = trainMLP(xTrain, yTrain, inputDim, hiddenDim, outputDim, learningRate, numEpochs)
Y = full(ind2vec(yTrain, outputDim));
Ntrain = size(xTrain, 2);
%rng(0);
W1 = 0.1 * randn(hiddenDim, inputDim);
b1 = zeros(hiddenDim, 1);
W2 = 0.1 * randn(outputDim, hiddenDim);
b2 = zeros(outputDim, 1);

for epoch = 1:numEpochs

    % Forward pass  step
    Z1 = W1 * xTrain + b1;      % hidden pre-activation
    A1 = activationFunction(Z1); % sigmoid activation helper function
    Z2 = W2 * A1 + b2;          % output pre-activation
    
    % Softmax output
    expZ = exp(Z2 - max(Z2,[],1)); % for numerical stability
    A2 = expZ ./ sum(expZ,1);
    
    % compute Loss (cross-entropy) 
    %loss = -mean(sum(Y .* log(A2 + 1e-12), 1));
    
    % Backpropagation step
    dZ2 = A2 - Y;                        % output layer gradient
    dW2 = (dZ2 * A1') / Ntrain;
    db2 = mean(dZ2, 2);
    
    dA1 = W2' * dZ2;
   % dZ1 = dA1 .* A1 .* (1 - A1);         % derivative of sigmoid
    dZ1 = dA1 .* (2 .* Z1); % derivative of quadratic y = x^2
    dW1 = (dZ1 * xTrain') / Ntrain;
    db1 = mean(dZ1, 2);
    
    % Gradient descent step
    W1 = W1 - learningRate * dW1;
    b1 = b1 - learningRate * db1;
    W2 = W2 - learningRate * dW2;
    b2 = b2 - learningRate * db2;
    
    % display progress
    %if mod(epoch, 10) == 0
        %fprintf('Epoch %d/%d, Loss = %.4f\n', epoch, numEpochs, loss);
    %end
end
end 

%% Performance Evaluation function
function yPred = predictMLP(x, W1, b1, W2, b2)
    Z1 = W1*x + b1; 
   % A1 = 1./(1+exp(-Z1));
    A1 = Z1.^2;  % quadratic activation
    Z2 = W2*A1 + b2;
    expZ = exp(Z2 - max(Z2,[],1));
    A2 = expZ ./ sum(expZ,1);
    [~, yPred] = max(A2, [], 1);
end

%% k-fold cross validation function
function indices = makeFolds(N, K)
% makeFolds: manually create K roughly equal folds for N samples
% Returns an index vector of length N with fold numbers (1..K)

    indices = zeros(1, N);
    foldSizes = repmat(floor(N / K), 1, K);
    remainder = mod(N, K);
    foldSizes(1:remainder) = foldSizes(1:remainder) + 1;

    allIdx = randperm(N);  % randomize sample order
    startIdx = 1;

    for k = 1:K
        thisFold = allIdx(startIdx : startIdx + foldSizes(k) - 1);
        indices(thisFold) = k;
        startIdx = startIdx + foldSizes(k);
    end
end

% activation function
function out = activationFunction(in)
% Pick a shared nonlinearity for all perceptrons: sigmoid or ramp style...
% You can mix and match nonlinearities in the model.
% However, typically this is not done; identical nonlinearity functions
% are better suited for parallelization of the implementation.
out = in.^2; % Quadratic Polynomial
%out = 1./(1+exp(-in)); % Logistic function - sigmoid style nonlinearity
end