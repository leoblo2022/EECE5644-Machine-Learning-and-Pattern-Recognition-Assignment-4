%% Homework 4: Question 1: Support Vector Machines (SVM) Classifier
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
figure(19); hold on; 
scatter(X_train_neg(:,1), X_train_neg(:,2), 20, 'b', 'filled');
scatter(X_train_pos(:,1), X_train_pos(:,2), 20, 'r', 'filled');
xlabel('x_1'); ylabel('x_2');
title('Training Set Visualization');
legend('Class -1 (r = 2)', 'Class +1 (r = 4)');
axis equal;

%% SVM Classifier Training
% Train a Gaussian kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
K = 10;
max_correct = 0; best_boxConstraint = 0; best_kernelScale = 0;
dummy = ceil(linspace(0,Ntrain,K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; 
end
CList = 10.^linspace(-1,9,11); sigmaList = 10.^linspace(-2,3,13);
CList = logspace(-2,3,15); sigmaList = logspace(-2,2,15);
for sigmaCounter = 1:length(sigmaList)
    [sigmaCounter,length(sigmaList)];
    kernelScale = sigmaList(sigmaCounter);
    for CCounter = 1:length(CList)
        C = CList(CCounter);
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            xValidate = xTrain(:,indValidate); % Using folk k as validation set
            lValidate = yTrain(indValidate);
	    if k == 1
        	indTrain = [indPartitionLimits(k,2)+1:Ntrain];
	    elseif k == K
        	indTrain = [1:indPartitionLimits(k,1)-1];
	    else
        	indTrain = [1:indPartitionLimits(k,1)-1,indPartitionLimits(k,2)+1:Ntrain];
	    end
            % using all other folds as training set
            x_Train = xTrain(:,indTrain); 
            lTrain = yTrain(indTrain);
            SVMk = fitcsvm(x_Train',lTrain,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',kernelScale);
            dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
            indCORRECT = find(lValidate == dValidate); 
            Ncorrect(k)=length(indCORRECT);
            
        end 
        
        % Pick best model order
        if sum(Ncorrect) > max_correct
            max_correct = sum(Ncorrect);
            best_boxConstraint = C;
            best_kernelScale = kernelScale;         
        end 
        
        PCorrect(CCounter,sigmaCounter)= sum(Ncorrect)/Ntrain;
        Accuracy = PCorrect(CCounter,sigmaCounter)*100;
        numColors = 20;        % however many iterations you want
        colors = parula(numColors);
       
        figure(2);     % connecting line
        scatter(C, Accuracy, 50, colors(sigmaCounter,:), 'filled')
        hold on
         % Set the x-axis to a logarithmic scale
        set(gca, 'XScale', 'log');
        ylim([40 85])
        ylabel("Accuracy")
     	("Box Constraint")
        hold on 
        
        fprintf('=== Accuracy of SVM ===\n');
        fprintf('Box Constraint =%.2f, kernel width =%.2f, Training Accuracy: %.2f%%\n', C, kernelScale, PCorrect(CCounter,sigmaCounter)*100);
        fprintf('\n') 
        
    end 
    
    figure(17);     % connecting line
    scatter(kernelScale, Accuracy, 50, colors(sigmaCounter,:), 'filled')
    hold on
    % Set the x-axis to a logarithmic scale
    set(gca, 'XScale', 'log');
    ylabel("Accuracy")
    ("Box Constraint")
    hold on   
end

% Retrain on full training data with best hyperparameters
SVMfinal = fitcsvm(xTrain',yTrain,'BoxConstraint', best_boxConstraint,'KernelFunction','gaussian','KernelScale', best_kernelScale);

 %% Performance Assessment 
% Test on large test set
yPred = SVMfinal.predict(xTest')'; % Labels of validation data using the trained SVM
indCORRECT = find(yTest == yPred); 
Ncorrect =length(indCORRECT);
PCorrect= sum(Ncorrect)/Ntest;
test_accuracy = PCorrect*100;
test_error = 100 - test_accuracy; 

fprintf('=== Test Accuracy of mlp ===\n');
fprintf('Best Box Constraint =%d, Best kernel width =%d, Test Error=%.3f, Test Accuracy: %.2f%%\n', best_boxConstraint, best_kernelScale, test_error, test_accuracy);
fprintf('\n')

%% Visualize Best Performance Results 
figure(18); hold on;
xlabel('x_1'); ylabel('x_2');
title('SVM Classification Results');
axis equal;

% Two shades of green (for correct classifications)
greenShades = [0.2 0.8 0.2; 
               0.0 0.6 0.0];
outputDim = 2;   % Two classes

for k = 1:outputDim
    correctIdx = find(yTest == k & yPred == k);
    incorrectIdx = find(yTest == k & yPred ~= k);

    % Correct: shaded green
    scatter(xTest(1, correctIdx), xTest(2, correctIdx), ...
            20, greenShades(k,:), 'filled');

    % Incorrect: red
    scatter(xTest(1, incorrectIdx), xTest(2, incorrectIdx), ...
            20, 'r', 'filled');
end

legend({'Correct (Class 1)', 'Correct (Class 2)', 'Incorrect'}, ...
       'Location', 'bestoutside');
hold off;
