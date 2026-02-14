% SOCIAL SAFETY - STEP 1: PREPARE DATA (HIGH PRECISION VERSION)
% Changes: Sampling rate increased (Every 2nd frame instead of 5th)
clc; clear; close all;

% --- CONFIGURATION ---
baseDataFolder = "..\..\01_Data";
trainFolder = fullfile(baseDataFolder, "train");
valFolder   = fullfile(baseDataFolder, "val");
outputFile  = fullfile(baseDataFolder, "FeatureData_HighRes.mat"); % New file name

if ~isfolder(trainFolder)
    error('Error: Data folders not found.');
end

% --- LOAD RESNET-18 ---
net = resnet18;
layerName = 'pool5'; 
inputSize = net.Layers(1).InputSize(1:2);

% --- PROCESSING FUNCTION ---
disp('Phase 1: Processing TRAINING Data (Dense Sampling)...');
[featuresTrain, labelsTrain] = processFolder(trainFolder, net, layerName, inputSize);

disp('Phase 2: Processing VALIDATION Data (Dense Sampling)...');
[featuresVal, labelsVal] = processFolder(valFolder, net, layerName, inputSize);

% --- SAVE ---
save(outputFile, 'featuresTrain', 'labelsTrain', 'featuresVal', 'labelsVal');
disp('SUCCESS: High-Res Data Prepared.');

% --- HELPER FUNCTION ---
function [features, labels] = processFolder(mainFolder, net, layerName, inputSize)
    categories = {'Fight', 'NonFight'};
    features = {};
    labels = {};
    
    for c = 1:length(categories)
        className = categories{c};
        currentPath = fullfile(mainFolder, className);
        videos = dir(fullfile(currentPath, '*.avi'));
        
        numVideos = length(videos);
        fprintf('  > Class: %s (%d videos)\n', className, numVideos);
        
        for i = 1:numVideos
            vidPath = fullfile(videos(i).folder, videos(i).name);
            try
                vid = VideoReader(vidPath);
                numFrames = vid.NumFrames;
                
                % CHANGE: Read every 2nd frame (More detail!)
                frameIdx = 1:2:numFrames; 
                
                % Limit sequence length (LSTM hates super long sequences)
                % If video is too long, cut it to first 150 frames
                if length(frameIdx) > 150
                    frameIdx = frameIdx(1:150);
                end
                
                if isempty(frameIdx), continue; end
                
                frames = read(vid, [1 numFrames]);
                frames = frames(:,:,:,frameIdx);
                
                % Resize
                framesResized = zeros(inputSize(1), inputSize(2), 3, length(frameIdx), 'uint8');
                for f = 1:length(frameIdx)
                    framesResized(:,:,:,f) = imresize(frames(:,:,:,f), inputSize);
                end
                
                % Extract
                feats = activations(net, framesResized, layerName, 'OutputAs', 'columns');
                features{end+1, 1} = feats; 
                labels{end+1, 1} = className;
                
            catch
                % Skip bad files
            end
            
            if mod(i, 50) == 0, fprintf('    Processed %d / %d\n', i, numVideos); end
        end
    end
    labels = categorical(labels);
end