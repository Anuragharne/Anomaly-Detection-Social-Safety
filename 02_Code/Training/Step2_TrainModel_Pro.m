% SOCIAL SAFETY - STEP 2: PRO TRAINING (STACKED BiLSTM)
% ARCHITECTURE: ResNet50 Features -> BiLSTM(128) -> BiLSTM(64) -> Classify
clc; clear; close all;

% --- 1. LOAD DATA ---
dataFile = "..\..\01_Data\FeatureData_Pro_YOLOv4.mat";

if ~isfile(dataFile)
    error('Error: FeatureData_Pro_YOLOv4.mat not found. Run Step 1 first!');
else
    disp('Loading Feature Data...');
    load(dataFile); 
end

% Verify Data Dimensions
numFeatures = size(featuresTrain{1}, 1); % Should be 2048 for ResNet-50
numClasses = 2; 

disp(['Feature Vector Size: ' num2str(numFeatures)]);
disp(['Training Samples:    ' num2str(numel(featuresTrain))]);
disp(['Validation Samples:  ' num2str(numel(featuresVal))]);

% --- 2. DEFINE PRO ARCHITECTURE ---
layers = [
    sequenceInputLayer(numFeatures, 'Name', 'input')
    
    % LAYER 1: The "Context" Layer
    % Bi-Directional LSTM looks at the video forwards and backwards
    bilstmLayer(128, 'OutputMode', 'sequence', 'Name', 'bilstm_1')
    dropoutLayer(0.5, 'Name', 'drop_1') % 50% Dropout stops memorization
    
    % LAYER 2: The "Focus" Layer
    % Condenses the information into a decision
    bilstmLayer(64, 'OutputMode', 'last', 'Name', 'bilstm_2')
    dropoutLayer(0.5, 'Name', 'drop_2')
    
    % CLASSIFICATION HEAD
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% --- 3. OPTIMIZED TRAINING OPTIONS ---
% Key settings for >85% Accuracy:
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...          % Batch 32 fits well in 4050 VRAM
    'InitialLearnRate', 1e-4, ...     % Slower LR = Better Convergence
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 10, ...    % Drop LR every 10 epochs
    'LearnRateDropFactor', 0.1, ...
    'L2Regularization', 0.02, ...     % Strong penalty for overfitting
    'GradientThreshold', 1, ...       % Prevents "NaN" errors
    'Shuffle', 'every-epoch', ...
    'ValidationData', {featuresVal, labelsVal}, ...
    'ValidationFrequency', 20, ...    % Check accuracy frequently
    'Plots', 'training-progress', ...
    'Verbose', true);

% --- 4. START TRAINING ---
disp('------------------------------------------------');
disp('STARTING TRAINING RUN...');
disp('Goal: Watch the Black Dotted Line (Validation Accuracy).');
disp('It should climb steadily to >85%.');

[netLSTM, info] = trainNetwork(featuresTrain, labelsTrain, layers, options);

% --- 5. SAVE ---
saveFolder = "..\..\03_Models";
save(fullfile(saveFolder, 'ViolenceModel_Pro_ResNet50.mat'), 'netLSTM');

disp('SUCCESS: Model Trained & Saved.');