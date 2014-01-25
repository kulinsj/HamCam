clear all

dataDir = './data';
resultsDir = 'ResultsSIGGRAPH2012';
infileName = 'JoanneSmall';
inFile = fullfile(dataDir,strcat(infileName,'.avi'));
outfile = fullfile(resultsDir,strcat(infileName,'KLT.avi'));
outfile2 = fullfile(resultsDir,strcat(infileName,'Crop'));
%outfile3 = fullfile(resultsDir,strcat(infileName,'CropRot'));
%outfile4 = fullfile(resultsDir,strcat(infileName,'CropRotCrop'));

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video frame and run the face detector.
videoFileReader = vision.VideoFileReader(inFile);
videoFrame      = step(videoFileReader);
bbox            = step(faceDetector, videoFrame);

% Convert the first box to a polygon.
% This is needed to be able to visualize the rotation of the object.
x = bbox(1, 1); y = bbox(1, 2); w = bbox(1, 3); h = bbox(1, 4);
bboxPolygon = [x, y, x+w, y, x+w, y+h, x, y+h];

a = 0.15;
cropagon = [x-a*w, y-a*h, x+w*(1+a), y-a*h, x+w*(1+a), y+h*(1+a), x-a*w, y+h*(1+a)];
Point = [y x] +0.5*w+0.5*h;
Pixels = [];
Pixels(1, :) = videoFrame (floor(Point(1)), floor(Point(2)), :);
% Draw the returned bounding box around the detected face.
videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon);
videoFrame = insertShape(videoFrame, 'Polygon', cropagon);

figure; imshow(videoFrame); title('Detected face');

% Detect feature points in the face region.
points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox);

% Display the detected points.
figure, imshow(videoFrame), hold on, title('Detected features');
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

NormalizeH = abs(cropagon(2) - cropagon(8));
NormalizeW = abs(cropagon(1) - cropagon(3));

myVideo = VideoWriter(outfile);
 Crop = VideoWriter(outfile2);
% CropRot = VideoWriter(outfile3);
% CropRotCrop = VideoWriter(outfile4);
open(myVideo);
 open(Crop);
% open(CropRot);
% open(CropRotCrop);
frame = 1;
% TrackedIndex = 120;
TrackedIndex = 543;
while ~isDone(videoFileReader)
    % get the next frame
    videoFrame = step(videoFileReader);
    frame = frame+1;
    
    % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame);
    
    iFT = isFound(1:TrackedIndex, 1);
    if isFound(TrackedIndex,1) == 0
       fprintf('point lost\n'); 
    end
    loss = sum(iFT(:)==0);
    
    TrackedIndex = TrackedIndex - loss;
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);
    
    MyPoint = visiblePoints(TrackedIndex, :);
    %MyPoint = [275 902];
    
    
%     fprintf('TI= %i, loss= %i, sizeP= (%i x %i), sizeVP= (%i x %i), sizeOI= (%i, %i)\n', TrackedIndex, loss, size(points), size(visiblePoints), size(oldInliers)); 
    if size(visiblePoints, 1) >= 2 % need at least 2 points

        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [xform, oldInliers, NvisiblePoints, newTrackedIndex] = estimateGeometricTransform_MOD(...
            oldInliers, visiblePoints, TrackedIndex, 'similarity', 'MaxDistance', 4);
        if newTrackedIndex == -1
            fprintf('point lost\n'); 
        end
        yes = size(NvisiblePoints,1) - size(visiblePoints,1);
%         if yes < 0
%             yes
%             TrackedIndex
%             newTrackedIndex
%         end
        visiblePoints = NvisiblePoints;
        TrackedIndex = newTrackedIndex;
        % Apply the transformation to the bounding box
        [bboxPolygon(1:2:end), bboxPolygon(2:2:end)] ...
            = transformPointsForward(xform, bboxPolygon(1:2:end), bboxPolygon(2:2:end));
         CropFrame = imcrop(videoFrame, [(MyPoint(1)-5) (MyPoint(2)-5) 11 11]);
        Pixels(frame, :) = videoFrame (floor(MyPoint(2)), floor(MyPoint(1)), :);
        
         writeVideo(Crop, CropFrame);
%         writeVideo(CropRot, CropRotNormal);
%         writeVideo(CropRotCrop, CropRotCropNormal);
        % Display tracked points
        videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
            'Color', 'white');
        videoFrame = insertMarker(videoFrame, MyPoint, '+', ...
            'Color', 'green');
        % Reset the points
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);
    end

    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame);
    writeVideo(myVideo, videoFrame);
end

fig1 = figure;
plot(1:frame,Pixels(:,1),'color',[1.0 0.0 0.0]);

fig2 = figure;
plot(1:frame,Pixels(:,2),'color',[0.0 1.0 0.0]);

fig3 = figure;
plot(1:frame,Pixels(:,3),'color',[0.0 0.0 1.0]);

% Clean up
release(videoFileReader);
release(videoPlayer);
release(pointTracker);
close(myVideo);
%close(Crop);
%close(CropRot);
%close(CropRotCrop);