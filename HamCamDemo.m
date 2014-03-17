close all;
clear;
clc;

dataDir = './data';
resultsDir = 'Results';

infileName = 'joanneafterBetter';
LEDX = 141;
LEDY = 531;
validationMinPeakDist = 8;

rangeRadius = 15;
expected = 132.6;

flow0 = (expected-rangeRadius)/60;
fhigh0 =(expected+rangeRadius)/60;

pulse_range = [50 180];

inFile = fullfile(dataDir,strcat(infileName,'.avi'));
outfile2 = fullfile(resultsDir,strcat(infileName,'Crop'));

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();
bigEyePairDetector = vision.CascadeObjectDetector('EyePairSmall');
bigEyePairDetector.MergeThreshold = 1;

% Read a video frame and run the face detector.
videoFileReader = vision.VideoFileReader(inFile);
videoFrame      = step(videoFileReader);
faceBox         = step(faceDetector, videoFrame);

numFaces = size(faceBox,1);

ExtrapolatedPoint = zeros(numFaces,3,2);

% Convert the first box to a polygon.
% This is needed to be able to visualize the rotation of the object.

for i = 1:numFaces
    faceX = faceBox(i, 1); 
    faceY = faceBox(i, 2); 
    faceW = faceBox(i, 3); 
    faceH = faceBox(i, 4);
    faceBoxPolygon = [faceX, faceY, faceX+faceW, faceY, faceX+faceW, faceY+faceH, faceX, faceY+faceH];
    
    faceCrops = imcrop(videoFrame, [faceX faceY faceW faceH]);
    % Draw the returned bounding box around the detected face.
    videoFrame = insertShape(videoFrame, 'Polygon', faceBoxPolygon);
    
    %only look for eye pair within detected face region
    pairEyeBoxBig = step(bigEyePairDetector, faceCrops);
    
    thresh = 1;
    while size(pairEyeBoxBig,1) > 1
        %if more than one pair of eyes is detected on the face, increase
        %the threshold until only one set of eyes is identified
        thresh = thresh+1;
        bigEyePairDetector.MergeThreshold = thresh;
        pairEyeBoxBig = step(bigEyePairDetector, faceCrops);
    end
    %translate coords in facecrop to coords in whole image
    pairEyeBoxBig(1,1) = pairEyeBoxBig(1,1)+faceX;
    pairEyeBoxBig(1,2) = pairEyeBoxBig(1,2)+faceY;
    
    
    eyePairBigX = pairEyeBoxBig(1, 1); 
    eyePairBigY = pairEyeBoxBig(1, 2); 
    eyePairBigW = pairEyeBoxBig(1, 3); 
    eyePairBigH = pairEyeBoxBig(1, 4);
    pairEyeBoxBigPoly = [eyePairBigX, eyePairBigY, eyePairBigX+eyePairBigW, eyePairBigY, eyePairBigX+eyePairBigW, eyePairBigY+eyePairBigH, eyePairBigX, eyePairBigY+eyePairBigH];
    videoFrame = insertShape(videoFrame, 'Polygon', pairEyeBoxBigPoly, 'Color', [1,0,1]);
    
    ExtrapolatedPoint(i,1:3,1:2) = [eyePairBigX + eyePairBigW/5, eyePairBigY + eyePairBigH*1.7; ...
        eyePairBigX + 4*eyePairBigW/5, eyePairBigY + eyePairBigH*1.7; ...
        eyePairBigX + eyePairBigW/2, eyePairBigY - eyePairBigH/2];
    videoFrame = insertMarker(videoFrame, [ExtrapolatedPoint(i,1,1) ExtrapolatedPoint(i,1,2)] , '+', 'Color', 'red');
    videoFrame = insertMarker(videoFrame, [ExtrapolatedPoint(i,2,1) ExtrapolatedPoint(i,2,2)], '+', 'Color', 'red');
    videoFrame = insertMarker(videoFrame, [ExtrapolatedPoint(i,3,1) ExtrapolatedPoint(i,3,2)], '+', 'Color', 'red');
end

numPoints = size(ExtrapolatedPoint,2);
%ExtrapolatedPoint now contains 3 points for each face

% (1,1,1) = face1, point1, x
% (1,1,2) = face1, point1, y
% (1,2,1) = face1, point2, x
% (1,2,2) = face1, point2, y
% ...
% (3,3,1) = face3, point3, x

figure; imshow(videoFrame); title('Chosen Points');

for k = 1:numFaces
    if k > 1
        videoFileReader = vision.VideoFileReader(inFile);
        videoFrame = step(videoFileReader);
    end
    
    % Detect feature points in the face region.
    points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', faceBox(k,:));
    
    minDist = zeros(size(points,1),size(ExtrapolatedPoint,2));
    for i = 1:size(points,1)
        currentLoc = points(i).Location; %loops through each of the detected points
        minDist(i,1) = pdist([currentLoc; ExtrapolatedPoint(k,1,1) ExtrapolatedPoint(k,1,2)]); %[x-current, y-current; extrapolated-x, extrapolated-y], ; separates the row
        minDist(i,2) = pdist([currentLoc; ExtrapolatedPoint(k,2,1) ExtrapolatedPoint(k,2,2)]);
        minDist(i,3) = pdist([currentLoc; ExtrapolatedPoint(k,3,1) ExtrapolatedPoint(k,3,2)]);   
    end

    [Loc, ind] = min(minDist);
    points = points(ind);
    
    %save vectors for distance between corner point and desired
    %extrapolated point
    
    %cornerpoint fun 
    p1= points(1).Location;%points is a cornerPoints object, so you must use this syntax to get the coordinates
    p2= points(2).Location;
    p3= points(3).Location;
    
    cropvectors(1,:) = [ExtrapolatedPoint(k,1,1),ExtrapolatedPoint(k,1,2)]-p1; %calculate the vector distance between ideal point and tracked cornerPoint
    cropvectors(2,:) = [ExtrapolatedPoint(k,2,1),ExtrapolatedPoint(k,2,2)]-p2;
    cropvectors(3,:) = [ExtrapolatedPoint(k,3,1),ExtrapolatedPoint(k,3,2)]-p3; 

    % Create a point tracker and enable the bidirectional error constraint to
    % make it more robust in the presence of noise and clutter.
    pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

    % Initialize the tracker with the initial point locations and the initial
    % video frame.
    points = points.Location;
    initialize(pointTracker, points, videoFrame);

    videoPlayer  = vision.VideoPlayer('Position',...
        [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30]);

    % Make a copy of the points to be used for computing the geometric
    % transformation between the points in the previous and the current frames
    oldPoints = points;

    frame = 1;
    numSamples = size(points,1);

    for i = 1:numSamples
       Crop(i) = VideoWriter(strcat(outfile2,num2str(i)));
       open(Crop(i));
    end
    
    while ~isDone(videoFileReader)
        % get the next frame
        videoFrame = step(videoFileReader);
        frame = frame+1;

        % Track the points. Note that some points may be lost.
        [points, isFound] = step(pointTracker, videoFrame);

        %sub-array of isFound to figure out point losses for each tracked
        %point's index.
        currentNumTrackedPoints = size(points,1);
        lossSoFar = 0;
        updatedTrackedIndicies = zeros(numSamples,1);

        for i = 1:size(isFound,1)
            if isFound(i) == 0
                fprintf('point lost: %i \n',i);
            end
        end

        visiblePoints = points(isFound, :);
        oldInliers = oldPoints(isFound, :);
        MyPoints = visiblePoints;

        if size(visiblePoints, 1) >= 2 % need at least 2 points

            % Estimate the geometric transformation between the old points
            % and the new points and eliminate outliers
            [~, ~, ~, ~] = estimateGeometricTransform_MOD(...
                oldInliers, visiblePoints, 1:size(points,1), 'similarity', 'MaxDistance', 4);

            %Crop the frame around the tracked points
            for j = 1:numSamples
                CropFrame = imcrop(videoFrame, [(MyPoints(j,1)-cropvectors(j,1)-5) (MyPoints(j,2)-cropvectors(j,2)-5) 11 11]); %crop patch
                writeVideo(Crop(j), CropFrame);
            end
            redLED(frame-1) = mean(videoFrame(LEDY, LEDX, :));
            videoFrame = insertMarker(videoFrame, [LEDX LEDY], '+', 'Color', 'red');
            videoFrame = insertMarker(videoFrame, MyPoints, '+', 'Color', 'blue');
            
            % Reset the points
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end
        % Display the annotated video frame using the video player object
        step(videoPlayer, videoFrame);
    end
    numFrames = frame;

    % Clean up
    release(videoPlayer);
    release(pointTracker);
    for i = 1:numSamples
        close(Crop(i));
    end
    
    % Only run this if using the Red LED
    [LEDpeaks, LEDlocs] = findpeaks(double(redLED),'MINPEAKDISTANCE',validationMinPeakDist, 'MINPEAKHEIGHT', 0.5);
    actualPeaks = size(LEDpeaks,2);
    Actual_BPM_from_EasyPulse = 60*actualPeaks*30/numFrames
    
    for i = 1:numSamples
        % Run MIT code on cropped video
        filename = strcat(strcat(strcat(infileName,'Crop'),num2str(i)),'.avi');
        inFile_Sample = fullfile(resultsDir,filename);
        fprintf('MIT processing face %i of %i, sample %i of %i...\n', k, numFaces, i, numSamples);
        amplify_spatial_Gdown_temporal_ideal(inFile_Sample,resultsDir,50,2,flow0,fhigh0,30, 0);
    end
    
    mean_r = zeros(numSamples,numFrames);
    mean_g = zeros(numSamples,numFrames);
    mean_b = zeros(numSamples,numFrames);
    for i = 1:numSamples
        rangeString = strcat(num2str(flow0),'-to-',num2str(fhigh0));
        filename = strcat(infileName, 'Crop', num2str(i), '-ideal-from-',rangeString,'-alpha-50-level-2-chromAtn-0.avi');
        inFile_Processed = fullfile(resultsDir,filename);

        videoFileReader = vision.VideoFileReader(inFile_Processed);
        videoFrame = step(videoFileReader);
        frame = 1;

        mean_r(i,frame) = mean(mean(videoFrame(:,:,1)));
        mean_g(i,frame) = mean(mean(videoFrame(:,:,2)));
        mean_b(i,frame) = mean(mean(videoFrame(:,:,3)));

        while ~isDone(videoFileReader)
            videoFrame = step(videoFileReader);
            frame = frame+1;
            
            mean_r(i,frame) = mean(mean(videoFrame(:,:,1)));
            mean_g(i,frame) = mean(mean(videoFrame(:,:,2)));
            mean_b(i,frame) = mean(mean(videoFrame(:,:,3)));
        end
    end
    mean_r = mean_r(1:numSamples,:);
    mean_g = mean_g(1:numSamples,:);
    mean_b = mean_b(1:numSamples,:);
    
    % ICA (assume 9 signals)
    addpath('./FastICA_2.5');
    sig = [mean_r(1,:); mean_r(2,:); mean_r(3,:); mean_g(1,:); mean_g(2,:); mean_g(3,:); mean_b(1,:); mean_b(2,:); mean_b(3,:)];
    [decomp] = fastica(sig,'verbose', 'off');
    F = fft(decomp,[],2);
    
    figure;
    plot(1:(size(F,2)-1),abs(F(:,2:end)));
    
    fps = 30;
    f = fps/size(F,2) * (0:floor(size(F,2)/2)-1);
    
    [amp,freq] = max(abs(F(:,2:end)),[],2);
    ICA_post = f(freq+1)*60;
    
    range1 = find(ICA_post(:) > pulse_range(1));
    range2 = find(ICA_post(:) < pulse_range(2));
    range = intersect(range1, range2);
    
    pulse_mask = zeros(size(amp));
    pulse_mask(range) = 1;
    
    [~,pulseInd] = max(amp.*pulse_mask);
    Calculated_Pulse = f(freq(pulseInd)+1)*60
end
