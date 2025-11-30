%% Homework 4: Image Segmentation
clear all; close all;
    
% Read in image
imdata = imread('buffalo.jpg'); 
figure(1)
imshow(imdata);
I = im2double(imread('buffalo.jpg'));   % Convert to [0,1]

%% Generate 5-D feature vector 
if length(size(imdata))==2 % grayscale image
    [R,C] = size(imdata); N = R*C; imdata = double(imdata); % overwriting, since I don't need the uint8 format anymore
    rowIndices = [1:R]'*ones(1,C); colIndices = ones(R,1)*[1:C];
    features = [rowIndices(:)';colIndices(:)';imdata(:)']; 
    minf = min(features,[],2); maxf = max(features,[],2); ranges = maxf-minf;
    x = diag(ranges.^(-1))*(features-repmat(minf,1,N)); % each feature normalized to the unit interval [0,1]
elseif length(size(imdata))==3 % color image with RGB color values
    [R,C,D] = size(imdata); N = R*C; imdata = double(imdata);
    rowIndices = [1:R]'*ones(1,C); colIndices = ones(R,1)*[1:C];
    features = [rowIndices(:)';colIndices(:)']; % initialize with row and column indices
    for d = 1:D
        imdatad = imdata(:,:,d); % pick one color at a time
        features = [features;imdatad(:)'];
    end
    minf = min(features,[],2); maxf = max(features,[],2);
    ranges = maxf-minf;
    x = diag(ranges.^(-1))*(features-repmat(minf,1,N)); % each feature normalized to the unit interval [0,1]
end
d = size(x,1); % feature dimensionality
    
 
%% Fit GMMs with model selection via K-fold cross validation 
max_components = 10;     % You can change this
K = 10;                  % 5-fold cross-validation
Nfolds = 10;
f = x'; % feature vector 

bestK = 0;
bestValLL = -inf;
bestGMM = [];
avg_list = zeros(1,0);


indices = makeFolds(size(f,1), Nfolds); % used helper function
nSamples = d*length(f(:,1));

for M = 1:10
    fprintf("Testing GMM with %d components...\n", M);
    foldLL = zeros(K,1);
    nParams(1,M) = (M-1) + d*M + M*(d+nchoosek(d,2)); % from prof

    for fold = 1:K
        testIdx = (indices == fold);
        trainIdx = ~testIdx;
        

        trainData = f(trainIdx,:);
        testData  = f(testIdx,:);

   
        gmm = fitgmdist(trainData, M, "RegularizationValue", 1e-5, "Options", statset("MaxIter", 500));
        %gmm = fitgmdist(trainData,M,'Replicates',2, "Options", statset("MaxIter", 1000)); 
      

        foldLL(fold) = mean(log(pdf(gmm, testData)));
    end

    avgLL = mean(foldLL);
    avg_list(end+1) = avgLL;

    % maximum average validation-log-likelihood  
    if avgLL > bestValLL
        bestValLL = avgLL;
        bestM = M;
        bestGMM = fitgmdist(f, M, "RegularizationValue", 1e-5, "Options", statset("MaxIter", 500));
        %bestGMM = fitgmdist(f,M,'Replicates',2, "Options", statset("MaxIter", 1000)); 
       
    end
end

fprintf("\nBest number of components = %d\n", bestM);

plot(1:10, avg_list)

%% Assign the most likely component label to each pixel by evaluating posterior probabilities for each feature vector 
labels = cluster_gmm(bestGMM, f);       % integers in [1..bestM]
labels_img = reshape(labels, length(imdata(:,1,1)), length(imdata(1,:,1)));

labels_gray = mat2gray(labels_img);

%% Visual Assessment: Original image and GMM-based segmentation side by side
figure;
subplot(1,2,1);
imshow(I);
title('Original Image');

subplot(1,2,2);
imshow(labels_gray);
title(sprintf('GMM Segmentation (%d components)', bestK));


%% Helper functions    
function labels = cluster_gmm(gmm, X)
% X is N×D
% Returns integer labels in [1..K]

    N = size(X,1);
    K = gmm.NumComponents;

    % Preallocate log-prob matrix
    logProb = zeros(N, K);

    for k = 1:K
        mu = gmm.mu(k, :);         % 1×D
        Sigma = gmm.Sigma(:, :, k);% D×D
        pi_k = gmm.ComponentProportion(k);

        % Compute log-likelihood under this Gaussian
        logPdf = log(mvnpdf(X, mu, Sigma) + eps);

        % Add log mixing weight
        logProb(:, k) = log(pi_k + eps) + logPdf;
    end

    % MAP assignment: choose the max posterior component
    [~, labels] = max(logProb, [], 2);
end
    
%% k-fold cross validation HELPER function
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
    

    
