close all;
clear;
clc;

dataDir = './data';
resultsDir = 'Results';
mkdir(resultsDir);

% infileName = 'more_still_small';
% inFile = fullfile(dataDir,strcat(infileName,'.mp4'));
% infileName = 'JoanneAudreyMultiFace4';
% inFile = fullfile(dataDir,strcat(infileName,'.mp4'));
infileName = 'JoanneSmall';
inFile = fullfile(dataDir,strcat(infileName,'.avi'));
outfile2 = fullfile(resultsDir,strcat(infileName,'Crop'));
outfile3 = fullfile(resultsDir,strcat(infileName,'Demo'));

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();
bigEyePairDetector = vision.CascadeObjectDetector('EyePairBig');
% bigEyePairDetector = vision.CascadeObjectDetector('EyePairSmall');
bigEyePairDetector.MergeThreshold = 1;
% smallEyePairDetector.MergeThreshold = 1;
% mouthDetector = vision.CascadeObjectDetector('Mouth'); 


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
    imshow(videoFrame);
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
    
    ExtrapolatedPoint(i,1:3,1:2) = [eyePairBigX + eyePairBigW/4, eyePairBigY + eyePairBigH*2; ...
        eyePairBigX + 3*eyePairBigW/4, eyePairBigY + eyePairBigH*2; ...
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
 
% videoFrame = insertMarker(videoFrame, [274 898], '+', 'Color', 'green');

figure; imshow(videoFrame); title('Detected Stuff');

for k = 1:numFaces
    if k > 1
        inFile = fullfile(dataDir,strcat(infileName,'.avi'));
        videoFileReader = vision.VideoFileReader(inFile);
        videoFrame = step(videoFileReader);
        imshow(videoFrame);
    end
    
    % Detect feature points in the face region.
    points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', faceBox(k,:));
    
    minDist = zeros(size(points,1),size(ExtrapolatedPoint,2));
    
    for i = 1:size(points,1)
        currentLoc = points(i).Location;
        minDist(i,1) = pdist([currentLoc; ExtrapolatedPoint(k,1,1) ExtrapolatedPoint(k,1,2)]);
        minDist(i,2) = pdist([currentLoc; ExtrapolatedPoint(k,2,1) ExtrapolatedPoint(k,2,2)]);
        minDist(i,3) = pdist([currentLoc; ExtrapolatedPoint(k,3,1) ExtrapolatedPoint(k,3,2)]);   
    end

    [Loc, ind] = min(minDist);

    points = points(ind);

    %Display the detected points.
    figure; imshow(videoFrame), hold on, title('Detected features');
    plot(points);



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

    % Crop = VideoWriter(outfile2);
    % open(Crop);

    frame = 1;
    % TrackedIndecies = 120;
    % TrackedIndecies = 543;
    % numSamples = 5;  %THIS is the parameter to change how many points are used
    numSamples = size(points,1);
    % TrackedIndecies = zeros(numSamples,1);
    % [numPoints, xy] = size(points);

    % interval = round(numPoints/numSamples);

    for i = 1:numSamples+1
    %    TrackedIndecies(i) = round(interval*(i-0.5)+1);
       Crop(i) = VideoWriter(strcat(outfile2,num2str(i)));
       open(Crop(i));
    end
    
    DemoVid = VideoWriter(outfile3);
    open(DemoVid);
    
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
    %         [xform, oldInliers, NvisiblePoints, newTrackedIndecies] = estimateGeometricTransform_MOD(...
    %             oldInliers, visiblePoints, TrackedIndecies, 'similarity', 'MaxDistance', 4);
            [xform, oldInliers, NvisiblePoints, newTrackedIndecies] = estimateGeometricTransform_MOD(...
                oldInliers, visiblePoints, 1:size(points,1), 'similarity', 'MaxDistance', 4);

    %         visiblePoints = NvisiblePoints;
    %         TrackedIndecies = newTrackedIndecies;
    %         MyPoints = visiblePoints(TrackedIndecies, :);
            % Apply the transformation to the bounding box
    %         [faceBoxPolygon(1:2:end), faceBoxPolygon(2:2:end)] ...
    %             = transformPointsForward(xform, faceBoxPolygon(1:2:end), faceBoxPolygon(2:2:end));

            %Crop the frame around the tracked points
            for j = 1:numSamples
                original(j,frame-1) = mean(videoFrame(round(MyPoints(j,1)), round(MyPoints(j,2)), :));
                CropFrame = imcrop(videoFrame, [(MyPoints(j,1)-5) (MyPoints(j,2)-5) 11 11]);
                writeVideo(Crop(j), CropFrame);
            end
            CropFrame = imcrop(videoFrame, [5 5 11 11]);
            writeVideo(Crop(4), CropFrame);
            %[274 899] RED LED in JoanneSmall.avi
            redLED(frame-1) = mean(videoFrame(899, 274, :));

            videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
                'Color', 'white');
            videoFrame = insertMarker(videoFrame, MyPoints, '+', ...
                'Color', 'green');
            videoFrame = insertMarker(videoFrame, [10 10], '+', ...
                'Color', 'green');
            
            writeVideo(DemoVid, videoFrame);
            
            % Reset the points
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end

        % Display the annotated video frame using the video player object
        step(videoPlayer, videoFrame);
        %writeVideo(myVideo, videoFrame);
    end
    
    numFrames = frame;

    % fig1 = figure;
    % plot(1:frame,Pixels(:,1),'color',[1.0 0.0 0.0]);
    % 
    % fig2 = figure;
    % plot(1:frame,Pixels(:,2),'color',[0.0 1.0 0.0]);
    % 
    % fig3 = figure;
    % plot(1:frame,Pixels(:,3),'color',[0.0 0.0 1.0]);

    % Clean up
    release(videoPlayer);
    release(pointTracker);
    for i = 1:numSamples+1
        close(Crop(i));
    end
    close(DemoVid);
    %close(myVideo);

    for i = 1:numSamples+1
        %% Run MIT code on cropped video
        %inFile = fullfile(dataDir,'JoanneSmallCrop.avi');
        
        filename = strcat(strcat(strcat(infileName,'Crop'),num2str(i)),'.avi');
        inFile = fullfile(resultsDir,filename);
        fprintf('face %i, sample %i, filter %i of %i/n', k, i, 1, 8);
        amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,50,1,40/60,50/60,30, 0);
        fprintf('face %i, sample %i, filter %i of %i/n', k, i, 2, 8);
        amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,50,1,50/60,60/60,30, 0);
        fprintf('face %i, sample %i, filter %i of %i/n', k, i, 3, 8);
        amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,50,1,60/60,70/60,30, 0);
        fprintf('face %i, sample %i, filter %i of %i/n', k, i, 4, 8);
        amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,50,1,70/60,80/60,30, 0);
        fprintf('face %i, sample %i, filter %i of %i/n', k, i, 5, 8);
        amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,50,1,80/60,90/60,30, 0);
        fprintf('face %i, sample %i, filter %i of %i/n', k, i, 6, 8);
        amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,50,1,90/60,100/60,30, 0);
        fprintf('face %i, sample %i, filter %i of %i/n', k, i, 7, 8);
        amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,50,1,100/60,110/60,30, 0);
        fprintf('face %i, sample %i, filter %i of %i/n', k, i, 8, 8);
        amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,50,1,110/60,120/60,30, 0);
    %     amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,50,3,120/60,240/60,30, 0);
    end
    
    for i = 1:numSamples+1
        fig1 = figure('name',strcat('Processed heartbeat from sample ', num2str(i), ' for face', num2str(k)));
        G = zeros(8,numFrames);
        modResponse = zeros(8,numFrames);
        responseMean = zeros(8, numFrames);
        for j = 1:8 % 7 = number of bands
            %% Graph data
            switch j
                case 1
                    rangeString = '0.66667-to-0.83333';
                    alpha = 1;
                    colArray = [1 0 0];
                case 2
                    rangeString = '0.83333-to-1';
                    alpha = 1.18;
                    colArray = [1 0.5 0];
                case 3
                    rangeString = '1-to-1.1667';
                    alpha = 1.35;
                    colArray = [1 1 0];
                case 4
                    rangeString = '1.1667-to-1.3333';
                    alpha = 1.49;
                    colArray = [0 1 0];
                case 5
                    rangeString = '1.3333-to-1.5';
                    alpha = 1.60;
                    colArray = [0 1 1];
                case 6
                    rangeString = '1.5-to-1.6667';
                    alpha = 1.7;
                    colArray = [0 0 1];
                case 7
                    rangeString = '1.6667-to-1.8333';
                    alpha = 1.77;
                    colArray = [1 0 1];
                case 8
                    rangeString = '1.8333-to-2';
                    alpha = 1.87;
                    colArray = [0.1 0.1 0.1];
            end
            filename = strcat(strcat(strcat(infileName,'Crop'),num2str(i)),'-ideal-from-',rangeString,'-alpha-50-level-1-chromAtn-0.avi');
        %     filename = strcat(strcat(strcat(infileName,'Crop'),num2str(i)),'-ideal-from-2-to-4-alpha-50-level-2-chromAtn-0.avi');
            inFile = fullfile(resultsDir,filename);

            videoFileReader = vision.VideoFileReader(inFile);
            videoFrame = step(videoFileReader);
            frame = 1;
        %     M = mean(mean(mean(videoFrame)));
            M = mean(videoFrame(5,:,1));
            G(j, frame) = M;

            while ~isDone(videoFileReader)
                videoFrame = step(videoFileReader);
                frame = frame+1;
        %         M = mean(mean(mean(videoFrame)));
                M = mean(videoFrame(5,:,1));
                G(j,frame) = M;
            end
%             plot(1:frame,G(:),'color',colArray);
            responseMean = zeros(1, size(G,2));
            for x = 25:(size(G,2)-25)
                responseMean(x) = mean(G(j,x-24:x+24));
                diff = G(j,x) - responseMean(x);
                modResponse(j, x) = responseMean(x) + alpha*diff;
            end
            plot(1:frame,modResponse(:),'color',colArray);
            hold on;
            ylim([0, 1]);
        end
        legend('40 to 50','50 to 60', '60 to 70', '70 to 80', '80 to 90', '90 to 100', '100 to 110', '110 to 120');
        figure('name', 'Hearbeat');
        for y = 50:numFrames-50
%             response55 = 
        end
    end
    figure;
    plot(redLED);
    ylim([0, 1]);
    xlim([0 500]);
    for j = 1:numSamples
       figure('name',strcat('Unaltered intensity for point ', num2str(j)));
       plot(original(j,:));
       ylim([0, 1]);
       xlim([0 500]);
    end
end
