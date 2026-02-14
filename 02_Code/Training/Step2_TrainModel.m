% SOCIAL SAFETY - STEP 2: TRAIN ADVANCED MODEL
% Features: Stacked BiLSTM, L2 Regularization, Gradient Clipping
clc; clear; close all;

% --- LOAD HIGH RES DATA ---
dataFile = "..\..\01_Data\FeatureData_HighRes.mat";

if ~isfile(dataFile)
    error('Error: FeatureData_HighRes.mat not found. Run Step 1 again!');
else
    load(dataFile); 
end

numFeatures = size(featuresTrain{1}, 1); 
numClasses = 2;

% --- ADVANCED ARCHITECTURE ---
layers = [
    sequenceInputLayer(numFeatures, 'Name', 'input')
    
    % Layer 1: Understands raw motion
    bilstmLayer(128, 'OutputMode', 'sequence', 'Name', 'bilstm1')
    dropoutLayer(0.6, 'Name', 'drop1') % Higher dropout (60%) to stop memorization
    
    % Layer 2: Understands "Patterns of Violence"
    bilstmLayer(100, 'OutputMode', 'last', 'Name', 'bilstm2')
    dropoutLayer(0.5, 'Name', 'drop2')
    
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% --- TRAINING OPTIONS ---
options = trainingOptions('adam', ...
    'MaxEpochs', 25, ...
    'MiniBatchSize', 24, ... % Smaller batch for better generalization
    'InitialLearnRate', 1e-4, ... % Slower learning rate (more careful)
    'L2Regularization', 0.02, ... % FORCE logic over memorization
    'GradientThreshold', 2, ...   % Prevents model from getting confused by noise
    'Shuffle', 'every-epoch', ...
    'ValidationData', {featuresVal, labelsVal}, ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% --- TRAIN ---
disp('Starting Advanced Training...');
[netLSTM, info] = trainNetwork(featuresTrain, labelsTrain, layers, options);

% --- SAVE ---
saveFolder = "..\..\03_Models";
save(fullfile(saveFolder, 'ViolenceModel_Advanced.mat'), 'netLSTM');
disp('SUCCESS: Advanced Model Saved.');