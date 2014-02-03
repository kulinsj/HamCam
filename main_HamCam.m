close all;
clear;
clc;

dataDir = './data';
resultsDir = 'Results';
mkdir(resultsDir);

% infileName = 'more_still_small';
% inFile = fullfile(dataDir,strcat(infileName,'.mp4'));
infileName = 'JoanneSmall';
inFile = fullfile(dataDir,strcat(infileName,'.avi'));
outfile2 = fullfile(resultsDir,strcat(infileName,'Crop'));

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();
leftEyeDetector = vision.CascadeObjectDetector('LeftEye');
rightEyeDetector = vision.CascadeObjectDetector('RightEye');
bigEyePairDetector = vision.CascadeObjectDetector('EyePairBig');
smallEyePairDetector = vision.CascadeObjectDetector('EyePairSmall');
mouthDetector = vision.CascadeObjectDetector('Mouth'); 


% Read a video frame and run the face detector.
videoFileReader = vision.VideoFileReader(inFile);
videoFrame      = step(videoFileReader);
faceBox         = step(faceDetector, videoFrame);
leftEyeBox      = step(leftEyeDetector, videoFrame);
rightEyeBox     = step(rightEyeDetector, videoFrame);
pairEyeBoxBig   = step(bigEyePairDetector, videoFrame);
pairEyeBoxSmall = step(smallEyePairDetector, videoFrame);
mouthBox        = step(mouthDetector, videoFrame);

% Convert the first box to a polygon.
% This is needed to be able to visualize the rotation of the object.

faceX = faceBox(1, 1); 
faceY = faceBox(1, 2); 
faceW = faceBox(1, 3); 
faceH = faceBox(1, 4);
faceBoxPolygon = [faceX, faceY, faceX+faceW, faceY, faceX+faceW, faceY+faceH, faceX, faceY+faceH];
% Draw the returned bounding box around the detected face.
videoFrame = insertShape(videoFrame, 'Polygon', faceBoxPolygon);

ReyeX = rightEyeBox(1, 1); 
ReyeY = rightEyeBox(1, 2); 
ReyeW = rightEyeBox(1, 3); 
ReyeH = rightEyeBox(1, 4);
ReyeBoxPolygon = [ReyeX, ReyeY, ReyeX+ReyeW, ReyeY, ReyeX+ReyeW, ReyeY+ReyeH, ReyeX, ReyeY+ReyeH];
videoFrame = insertShape(videoFrame, 'Polygon', ReyeBoxPolygon, 'Color', [1,0,0]);

LeyeX = leftEyeBox(1, 1); 
LeyeY = leftEyeBox(1, 2); 
LeyeW = leftEyeBox(1, 3); 
LeyeH = leftEyeBox(1, 4);
LeyeBoxPolygon = [LeyeX, LeyeY, LeyeX+LeyeW, LeyeY, LeyeX+LeyeW, LeyeY+LeyeH, LeyeX, LeyeY+LeyeH];
videoFrame = insertShape(videoFrame, 'Polygon', LeyeBoxPolygon, 'Color', [1,1,0]);

eyePairBigX = pairEyeBoxBig(1, 1); 
eyePairBigY = pairEyeBoxBig(1, 2); 
eyePairBigW = pairEyeBoxBig(1, 3); 
eyePairBigH = pairEyeBoxBig(1, 4);
eyePairBigBoxPolygon = [eyePairBigX, eyePairBigY, eyePairBigX+eyePairBigW, eyePairBigY, eyePairBigX+eyePairBigW, eyePairBigY+eyePairBigH, eyePairBigX, eyePairBigY+eyePairBigH];
videoFrame = insertShape(videoFrame, 'Polygon', eyePairBigBoxPolygon, 'Color', [1,0,1]);

ExtrapolatedPoint = [eyePairBigX + eyePairBigW/4, eyePairBigY + eyePairBigH*2; ...
    eyePairBigX + 3*eyePairBigW/4, eyePairBigY + eyePairBigH*2; ...
    eyePairBigX + eyePairBigW/2, eyePairBigY - eyePairBigH/2];
videoFrame = insertMarker(videoFrame, ExtrapolatedPoint, '+', 'Color', 'red');

% eyePairSmallX = pairEyeBoxSmall(1, 1); 
% eyePairSmallY = pairEyeBoxSmall(1, 2); 
% eyePairSmallW = pairEyeBoxSmall(1, 3); 
% eyePairSmallH = pairEyeBoxSmall(1, 4);
% eyePairSmallBoxPolygon = [eyePairSmallX, eyePairSmallY, eyePairSmallX+eyePairSmallW, eyePairSmallY, eyePairSmallX+eyePairSmallW, eyePairSmallY+eyePairSmallH, eyePairSmallX, eyePairSmallY+eyePairSmallH];
% videoFrame = insertShape(videoFrame, 'Polygon', eyePairSmallBoxPolygon, 'Color', [0,1,0]);

mouthX = mouthBox(1, 1); 
mouthY = mouthBox(1, 2); 
mouthW = mouthBox(1, 3); 
mouthH = mouthBox(1, 4);
mouthBoxPolygon = [mouthX, mouthY, mouthX+mouthW, mouthY, mouthX+mouthW, mouthY+mouthH, mouthX, mouthY+mouthH];
videoFrame = insertShape(videoFrame, 'Polygon', mouthBoxPolygon, 'Color', [0,0,1]);


% figure; imshow(videoFrame); title('Detected Stuff');

% Detect feature points in the face region.
points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', faceBox);

minDist = zeros(size(points,1),size(ExtrapolatedPoint,1));

for i = 1:size(points,1)
    currentLoc = points(i).Location;
    minDist(i,1) = pdist([currentLoc; ExtrapolatedPoint(1,:)]);
    minDist(i,2) = pdist([currentLoc; ExtrapolatedPoint(2,:)]);
    minDist(i,3) = pdist([currentLoc; ExtrapolatedPoint(3,:)]);   
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

for i = 1:numSamples
%    TrackedIndecies(i) = round(interval*(i-0.5)+1);
   Crop(i) = VideoWriter(strcat(outfile2,num2str(i)));
   open(Crop(i));
end

% while ~isDone(videoFileReader)
%     % get the next frame
%     videoFrame = step(videoFileReader);
%     frame = frame+1;
%     
%     % Track the points. Note that some points may be lost.
%     [points, isFound] = step(pointTracker, videoFrame);
%     
%     %sub-array of isFound to figure out point losses for each tracked
%     %point's index.
%     currentNumTrackedPoints = size(TrackedIndecies);
%     lossSoFar = 0;
%     updatedTrackedIndicies = zeros(numSamples,1);
%     
%     for i = 1:currentNumTrackedPoints(1)
%         %create sub-array of "isFound" between each tracked point index
%         %pair
%         if i == 1
%             iFT = isFound(1:TrackedIndecies(1),1);
%         else
%             iFT = isFound((TrackedIndecies(i-1)+1):TrackedIndecies(i),1);
%         end
%         %calc number of lost points up to the current index
%         thisIFTloss = sum(iFT(:)==0);
%         lossSoFar = lossSoFar + thisIFTloss;
%         if isFound(TrackedIndecies(i),1) == 0
%             fprintf('point lost: %i \n', TrackedIndecies(i));
%             updatedTrackedIndicies(i) = - 1;
%         else
%             %save updated index to new array
%             updatedTrackedIndicies(i) = TrackedIndecies(i) - lossSoFar;
%         end
%     end
%     
%     TrackedIndecies = updatedTrackedIndicies;
%     
% %     iFT = isFound(1:TrackedIndecies, 1);
% %     if isFound(TrackedIndecies,1) == 0
% %        fprintf('point lost\n'); 
% %     end
% %     loss = sum(iFT(:)==0);
% %     TrackedIndecies = TrackedIndecies - loss;
%     
%     visiblePoints = points(isFound, :);
%     oldInliers = oldPoints(isFound, :);
%     MyPoints = visiblePoints(TrackedIndecies(:), :);
%     %MyPoints = [275 902];
%     
%     if size(visiblePoints, 1) >= 2 % need at least 2 points
% 
%         % Estimate the geometric transformation between the old points
%         % and the new points and eliminate outliers
%         [xform, oldInliers, NvisiblePoints, newTrackedIndecies] = estimateGeometricTransform_MOD(...
%             oldInliers, visiblePoints, TrackedIndecies, 'similarity', 'MaxDistance', 4);
%         
%         visiblePoints = NvisiblePoints;
%         TrackedIndecies = newTrackedIndecies;
%         MyPoints = visiblePoints(TrackedIndecies, :);
%         %MyPoints
%         % Apply the transformation to the bounding box
%         [faceBoxPolygon(1:2:end), faceBoxPolygon(2:2:end)] ...
%             = transformPointsForward(xform, faceBoxPolygon(1:2:end), faceBoxPolygon(2:2:end));
%         
%         %Crop the frame around the tracked points
%         for j = 1:numSamples
%             CropFrame = imcrop(videoFrame, [(MyPoints(j,1)-5) (MyPoints(j,2)-5) 11 11]);
%             writeVideo(Crop(j), CropFrame);
%         end
%         
%         videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
%             'Color', 'white');
%         videoFrame = insertMarker(videoFrame, MyPoints, '+', ...
%             'Color', 'green');
%         % Reset the points
%         oldPoints = visiblePoints;
%         setPoints(pointTracker, oldPoints);
%     end
% 
%     % Display the annotated video frame using the video player object
%     step(videoPlayer, videoFrame);
%     %writeVideo(myVideo, videoFrame);
% end
% 
% % fig1 = figure;
% % plot(1:frame,Pixels(:,1),'color',[1.0 0.0 0.0]);
% % 
% % fig2 = figure;
% % plot(1:frame,Pixels(:,2),'color',[0.0 1.0 0.0]);
% % 
% % fig3 = figure;
% % plot(1:frame,Pixels(:,3),'color',[0.0 0.0 1.0]);
% 
% % Clean up
% release(videoPlayer);
% release(pointTracker);
% for i = 1:numSamples
%     close(Crop(i));
% end
% %close(myVideo);
% 
% for i = 1:numSamples
%     %% Run MIT code on cropped video
%     %inFile = fullfile(dataDir,'JoanneSmallCrop.avi');
%     filename = strcat(strcat(strcat(infileName,'Crop'),num2str(i)),'.avi');
%     inFile = fullfile(resultsDir,filename);
%     amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,50,2,45/60,100/60,30, 0);
% end
% 
% for i = 1:numSamples
%     %% Graph data
%     filename = strcat(strcat(strcat(infileName,'Crop'),num2str(i)),'-ideal-from-0.75-to-1.6667-alpha-50-level-2-chromAtn-0.avi');
%     inFile = fullfile(resultsDir,filename);
% 
%     videoFileReader = vision.VideoFileReader(inFile);
%     videoFrame = step(videoFileReader);
%     frame = 1;
%     M = mean(mean(mean(videoFrame)));
% 
%     G = [];
%     G(frame) = M;
% 
%     while ~isDone(videoFileReader)
%         videoFrame = step(videoFileReader);
%         frame = frame+1;
%         M = mean(mean(mean(videoFrame)));
%         G(frame) = M;
%     end
% 
%     fig1 = figure('name',strcat('Processed heartbeat from sample ', num2str(i)));
%     plot(1:frame,G(:),'color',[1.0 0.0 0.0]);
% end
