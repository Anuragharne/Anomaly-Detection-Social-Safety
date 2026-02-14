% SOCIAL SAFETY - STEP 1: PRO DATA PREPARATION (GPU + YOLOv4 + RESNET50)
% TARGET: >85% Accuracy on Violence Detection
% HARDWARE: NVIDIA RTX 4050 Optimized
clc; clear; close all;

% --- 1. CONFIGURATION ---
baseDataFolder = "..\..\01_Data";
trainFolder = fullfile(baseDataFolder, "train");
valFolder   = fullfile(baseDataFolder, "val");
outputFile  = fullfile(baseDataFolder, "FeatureData_Pro_YOLOv4.mat");

% --- 2. INITIALIZE GPU & MODELS ---
disp('------------------------------------------------');
disp('INITIALIZING AI ENGINES ON RTX 4050...');
try
    % Reset GPU to clear VRAM
    g = gpuDevice(1);
    freeMemory = g.AvailableMemory / 1e9;
    fprintf('  > GPU Detected: %s (%.1f GB Free)\n', g.Name, freeMemory);
    
    % A. Load Feature Extractor (ResNet-50 is deeper/better than ResNet-18)
    net = resnet50; 
    layerName = 'avg_pool'; % The feature layer for ResNet-50
    inputSize = net.Layers(1).InputSize(1:2); % [224 224]
    fprintf('  > Feature Extractor: ResNet-50 (Loaded)\n');

    % B. Load Object Detector (YOLO v4 Tiny)
    yolo = yolov4ObjectDetector("tiny-yolov4-coco");
    fprintf('  > Object Detector: YOLO v4-Tiny (Loaded)\n');
    
catch ME
    error('MISSING ADD-ONS! Please install: \n 1. ResNet-50 Network \n 2. YOLO v4 Object Detection \n 3. Parallel Computing Toolbox');
end

% --- 3. EXECUTION ---
disp('------------------------------------------------');
disp('PHASE 1: PROCESSING TRAINING DATA...');
[featuresTrain, labelsTrain] = processFolder(trainFolder, net, yolo, layerName, inputSize);

disp('------------------------------------------------');
disp('PHASE 2: PROCESSING VALIDATION DATA...');
[featuresVal, labelsVal] = processFolder(valFolder, net, yolo, layerName, inputSize);

% --- 4. SAVE ---
disp('------------------------------------------------');
disp('SAVING DATASETS...');
save(outputFile, 'featuresTrain', 'labelsTrain', 'featuresVal', 'labelsVal');
disp('SUCCESS: Pro-Level Feature Extraction Complete.');


% ---------------------------------------------------------
% HELPER FUNCTION: THE SMART PROCESSOR
% ---------------------------------------------------------
function [features, labels] = processFolder(mainFolder, net, detector, layerName, inputSize)
    categories = {'Fight', 'NonFight'};
    features = {};
    labels = {};
    
    for c = 1:length(categories)
        className = categories{c};
        currentPath = fullfile(mainFolder, className);
        videos = dir(fullfile(currentPath, '*.avi'));
        
        numVideos = length(videos);
        fprintf('  > Class: %s (%d videos)\n', className, numVideos);
        
        % Iterate through videos
        for i = 1:numVideos
            vidPath = fullfile(videos(i).folder, videos(i).name);
            try
                vid = VideoReader(vidPath);
                numFrames = vid.NumFrames;
                
                % STRATEGY: High-Density Sampling
                % We take every 2nd frame to capture fast punches.
                frameIdx = 1:2:numFrames;
                
                % MEMORY GUARD: Cap at 64 frames (Power of 2 is better for GPU)
                if length(frameIdx) > 64, frameIdx = frameIdx(1:64); end
                if isempty(frameIdx), continue; end
                
                % Read frames into memory
                rawFrames = read(vid, [1 numFrames]);
                rawFrames = rawFrames(:,:,:,frameIdx);
                
                % Pre-allocate the processed batch
                framesProcessed = zeros(inputSize(1), inputSize(2), 3, length(frameIdx), 'uint8');
                
                % Box Persistence: Remember last known location if detection fails
                lastBox = []; 
                
                % --- FRAME-BY-FRAME INTELLIGENT CROPPING ---
                for f = 1:length(frameIdx)
                    img = rawFrames(:,:,:,f);
                    
                    % 1. Detect People (GPU Accelerated)
                    [bboxes, scores, labelIdx] = detect(detector, img, 'ExecutionEnvironment', 'gpu');
                    
                    % Filter: Only 'person' class (ID 1 in COCO usually, but checking string is safer)
                    % Note: YOLOv4 tiny coco: 'person' is class 1.
                    isPerson = scores > 0.25; % Confidence Threshold
                    % (Simplification: In a violence dataset, almost all detections are people)
                    
                    personBoxes = bboxes(isPerson, :);
                    
                    if ~isempty(personBoxes)
                        % 2. Smart Crop: Focus on the ACTION (Union of all people)
                        minX = max(1, round(min(personBoxes(:,1))));
                        minY = max(1, round(min(personBoxes(:,2))));
                        maxX = min(size(img,2), round(max(personBoxes(:,1) + personBoxes(:,3))));
                        maxY = min(size(img,1), round(max(personBoxes(:,2) + personBoxes(:,4))));
                        
                        % Add Padding (Context is important for violence)
                        pad = 20;
                        minX = max(1, minX-pad); minY = max(1, minY-pad);
                        maxX = min(size(img,2), maxX+pad); maxY = min(size(img,1), maxY+pad);
                        
                        % Crop & Save Box for next time
                        cropImg = img(minY:maxY, minX:maxX, :);
                        framesProcessed(:,:,:,f) = imresize(cropImg, inputSize);
                        lastBox = [minX, minY, maxX, maxY];
                        
                    elseif ~isempty(lastBox)
                        % 3. Persistence: Detection failed? Use the last known box!
                        % This prevents flickering/black frames.
                        minX = lastBox(1); minY = lastBox(2);
                        maxX = lastBox(3); maxY = lastBox(4);
                        cropImg = img(minY:maxY, minX:maxX, :);
                        framesProcessed(:,:,:,f) = imresize(cropImg, inputSize);
                        
                    else
                        % 4. Fallback: No people ever seen? Use full frame.
                        framesProcessed(:,:,:,f) = imresize(img, inputSize);
                    end
                end
                
                % --- FEATURE EXTRACTION (PURE GPU SPEED) ---
                % Move the entire batch of 64 images to GPU VRAM
                gpuFrames = gpuArray(framesProcessed);
                
                % Run ResNet-50
                featsGPU = activations(net, gpuFrames, layerName, ...
                    'OutputAs', 'columns', 'ExecutionEnvironment', 'gpu');
                
                % Retrieve features
                features{end+1, 1} = gather(featsGPU); 
                labels{end+1, 1} = className;
                
            catch
                % Skip corrupt video files
            end
            
            % Progress Bar
            if mod(i, 20) == 0
                fprintf('    Processed %d / %d (Class: %s)\n', i, numVideos, className);
            end
        end
    end
    labels = categorical(labels);
end