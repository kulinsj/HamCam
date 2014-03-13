close all;
clear;
clc;
figure;
dataDir = './data';
resultsDir = 'Results';
mkdir(resultsDir);

% infileName = 'more_still_small';
% infileName = 'JoanneAudreyMultiFace4';
% infileName = 'JoanneSmall';
infileName = 'face';
% inFile = fullfile(dataDir,strcat(infileName,'.avi'));
inFile = fullfile(dataDir,strcat(infileName,'.mp4'));
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

% Convert the first box to a polygon.
% This is needed to be able to visualize the rotation of the object.

pairEyeBoxBig = step(bigEyePairDetector, videoFrame);
eyePairBigX = round(pairEyeBoxBig(1, 1)); 
eyePairBigY = round(pairEyeBoxBig(1, 2)); 
eyePairBigW = round(pairEyeBoxBig(1, 3)); 
eyePairBigH = round(pairEyeBoxBig(1, 4));
pairEyeBoxBigPoly = [eyePairBigX, eyePairBigY, eyePairBigX+eyePairBigW, eyePairBigY, eyePairBigX+eyePairBigW, eyePairBigY+eyePairBigH, eyePairBigX, eyePairBigY+eyePairBigH];
videoFrame = insertShape(videoFrame, 'Polygon', pairEyeBoxBigPoly, 'Color', [1,0,1]);

numFaces = size(pairEyeBoxBig,1);
ExtrapolatedPoint = zeros(numFaces,3,2);
for i = 1:numFaces
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

for k = 1:1
    if k > 1
        videoFileReader = vision.VideoFileReader(inFile);
        videoFrame = step(videoFileReader);
        imshow(videoFrame);
    end
    
    % Detect feature points in the face region.
    points = detectMinEigenFeatures(rgb2gray(videoFrame));
    
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

%     videoPlayer  = vision.VideoPlayer('Position',...
%         [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30]);

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
    
%     DemoVid = VideoWriter(outfile3);
%     open(DemoVid);
    
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

%         for i = 1:size(isFound,1)
%             if isFound(i) == 0
%                 fprintf('point lost: %i \n',i);
%             end
%         end

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
%             redLED(frame-1) = mean(videoFrame(899, 274, :));

            videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
                'Color', 'white');
            videoFrame = insertMarker(videoFrame, MyPoints, '+', ...
                'Color', 'green');
            videoFrame = insertMarker(videoFrame, [10 10], '+', ...
                'Color', 'green');
            
%             writeVideo(DemoVid, videoFrame);
            
            % Reset the points
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end

        % Display the annotated video frame using the video player object
%         step(videoPlayer, videoFrame);
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
%     release(videoPlayer);
    release(pointTracker);
    for i = 1:numSamples+1
        close(Crop(i));
    end
%     close(DemoVid);
    %close(myVideo);

    for i = 1:numSamples+1
        %% Run MIT code on cropped video
        %inFile = fullfile(dataDir,'JoanneSmallCrop.avi');
        
        filename = strcat(strcat(strcat(infileName,'Crop'),num2str(i)),'.avi');
        inFileM = fullfile(resultsDir,filename);
        fprintf('face %i, sample %i, filter %i of %i\n', k, i, 1, 9);
        amplify_spatial_Gdown_temporal_ideal(inFileM,resultsDir,50,1,0.01,0.06,30, 0);
        fprintf('face %i, sample %i, filter %i of %i\n', k, i, 2, 9);
        amplify_spatial_Gdown_temporal_ideal(inFileM,resultsDir,50,1,0.05,0.1,30, 0);
        fprintf('face %i, sample %i, filter %i of %i\n', k, i, 3, 9);
        amplify_spatial_Gdown_temporal_ideal(inFileM,resultsDir,50,1,0.1,0.15,30, 0);
        fprintf('face %i, sample %i, filter %i of %i\n', k, i, 4, 9);
        amplify_spatial_Gdown_temporal_ideal(inFileM,resultsDir,50,1,0.2,0.25,30, 0);
        fprintf('face %i, sample %i, filter %i of %i\n', k, i, 5, 9);
        amplify_spatial_Gdown_temporal_ideal(inFileM,resultsDir,50,1,0.3,0.35,30, 0);
        fprintf('face %i, sample %i, filter %i of %i\n', k, i, 6, 9);
        amplify_spatial_Gdown_temporal_ideal(inFileM,resultsDir,50,1,0.833,1,30, 0);
        fprintf('face %i, sample %i, filter %i of %i\n', k, i, 7, 9);
        amplify_spatial_Gdown_temporal_ideal(inFileM,resultsDir,50,1,1,1.05,30, 0);
        fprintf('face %i, sample %i, filter %i of %i\n', k, i, 8, 9);
        amplify_spatial_Gdown_temporal_ideal(inFileM,resultsDir,50,1,1.2,1.25,30, 0);
        fprintf('face %i, sample %i, filter %i of %i\n', k, i, 9, 9);
        amplify_spatial_Gdown_temporal_ideal(inFileM,resultsDir,50,1,2,2.5,30, 0);
    end
    
    for i = 1:numSamples+1
        G = zeros(9,numFrames);
        for j = 1:9 % 9 = number of bands
            %% Build the data
            switch j
                case 1
                    rangeString = '0.01-to-0.06';
                case 2
                    rangeString = '0.05-to-0.1';
                case 3
                    rangeString = '0.1-to-0.15';
                case 4
                    rangeString = '0.2-to-0.25';
                case 5
                    rangeString = '0.3-to-0.35';
                case 6
                    rangeString = '0.833-to-1';
                case 7
                    rangeString = '1-to-1.05';
                case 8
                    rangeString = '1.2-to-1.25';
                case 9
                    rangeString = '2-to-2.5';
            end
            filename = strcat(strcat(strcat(infileName,'Crop'),num2str(i)),'-ideal-from-',rangeString,'-alpha-50-level-1-chromAtn-0.avi');
            inFileG = fullfile(resultsDir,filename);

            videoFileReader = vision.VideoFileReader(inFileG);
            videoFrame = step(videoFileReader);
            frame = 1;
            G(j, frame) = mean(videoFrame(5,:,1));

            while ~isDone(videoFileReader)
                videoFrame = step(videoFileReader);
                frame = frame + 1;
                G(j,frame) = mean(videoFrame(5,:,1));
            end
        end
        
        fig1 = figure('name',strcat('Processed heartbeat from sample ', num2str(i), ' for face', num2str(k)));
        for j = 1:9  %plot the original gathered data
            switch j
                case 1
                    colArray = [1 0 0];
                case 2
                    colArray = [1 0.5 0];
                case 3
                    colArray = [1 1 0];
                case 4
                    colArray = [0 1 0];
                case 5
                    colArray = [0 1 1];
                case 6
                    colArray = [0 0 1];
                case 7
                    colArray = [1 0 1];
                case 8
                    colArray = [1 0.5 1];
                case 9
                    colArray = [0.1 0.1 0.1];
            end
            plot(1:numFrames,G(j,:),'color',colArray);
            hold on;
        end
        ylim([0, 1]);
        legend('0.01-to-0.05','0.05-to-0.1', '0.1-to-0.15', '0.2-to-0.25', '0.3-to-0.35', '0.833-to-1', '1-to-1.05', '1.2-to-1.25', '2-to-2.5');
        
        
        modelfun = @(b,x)(b(1)+b(2)*sin(b(3)+b(4)*x));
        beta0 = [0.5;0.15;0.2;0.01]; %Guess at initial params of sine functions
        B = zeros(9,4);
        
        div = 50;
        figure('name',strcat('Fit curves for sample ', num2str(i), ' for face', num2str(k)));
        for j = 1:9  %plot the fit curves
            b1 = mean(G(j,:));
            b2 = var(G(j,:));
            switch j
                case 1
                    colArray = [1 0 0];
                    beta0 = [b1;b2;1;1/div];
                case 2
                    colArray = [1 0.5 0];
                    beta0 = [b1;b2;1;1/div];
                case 3
                    colArray = [1 1 0];
                    beta0 = [b1;b2;1;2/div];
                case 4
                    colArray = [0 1 0];
                    beta0 = [b1;b2;1;4/div];
                case 5
                    colArray = [0 1 1];
                    beta0 = [b1;b2;1;5/div];
                case 6
                    colArray = [0 0 1];
                    beta0 = [b1;b2;1;11/div];
                case 7
                    colArray = [1 0 1];
                    beta0 = [b1;b2;1;26/div];
                case 8
                    colArray = [1 0.5 1];
                    beta0 = [b1;b2;1;29/div];
                case 9
                    colArray = [0.1 0.1 0.1];
                    beta0 = [b1;b2;1;38/div];
            end
            Y = G(j,find(G(j,:),1,'first'):find(G(j,:),1,'last'));
            numPoints = size(Y,2);
            X = 1:numPoints;
            [beta, R, J, CovB, MSE, ErrorModelInfo] = nlinfit(X,Y,modelfun,beta0);
            MeanSE(j) = MSE;
            for b=1:4
                B(j,b) = beta(b);
            end
            toPlot = (beta(1)+beta(2)*sin(beta(3)+beta(4)*X));
            plot(toPlot, 'color',colArray);
            hold on;
        end
        legend(num2str(B(1,2)),num2str(B(2,2)),num2str(B(3,2)),num2str(B(4,2)),num2str(B(5,2)),num2str(B(6,2)),num2str(B(7,2)),num2str(B(8,2)),num2str(B(9,2)));
        ylim([0, 1]);
        B
        MeanSE
%         figure('name', 'Hearbeat');
%         for y = 50:numFrames-50
%              response55 = 
%         end
    end
%     figure;
%     plot(redLED);
%     ylim([0, 1]);
%     xlim([0 500]);
    for j = 1:numSamples
       figure('name',strcat('Unaltered intensity for point ', num2str(j)));
       plot(original(j,:));
       ylim([0, 1]);
       xlim([0 500]);
    end
end
