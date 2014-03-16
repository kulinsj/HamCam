close all;
clear;
clc;

dataDir = './data';
resultsDir = 'Results';
mkdir(resultsDir);

infileName = 'audreybeforeBetter';
LEDX = 64;
LEDY = 360;
validationMinPeakDist = 10;
inFile = fullfile(dataDir,strcat(infileName,'.avi'));
% inFile = fullfile(dataDir,strcat(infileName,'.mp4'));

outfile2 = fullfile(resultsDir,strcat(infileName,'Crop'));

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();
% bigEyePairDetector = vision.CascadeObjectDetector('EyePairBig');
bigEyePairDetector = vision.CascadeObjectDetector('EyePairSmall');
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
    
    ExtrapolatedPoint(i,1:3,1:2) = [eyePairBigX + eyePairBigW/4, eyePairBigY + eyePairBigH*1.2; ...
        eyePairBigX + 3*eyePairBigW/4, eyePairBigY + eyePairBigH*1.2; ...
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

figure; imshow(videoFrame); title('Detected Stuff');

for k = 1:numFaces
    if k > 1
        videoFileReader = vision.VideoFileReader(inFile);
        videoFrame = step(videoFileReader);
        imshow(videoFrame);
    end
    
    % Detect feature points in the face region.
    points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', faceBox(k,:));
    
%     figure; 
%     p = points(38).Location;
%     videoFrame = insertMarker(videoFrame, p, 'o','Color', 'white');
%     imshow(videoFrame), hold on, title('Detected features');
%     plot(points);
    
    minDist = zeros(size(points,1),size(ExtrapolatedPoint,2));
    
    for i = 1:size(points,1)
        currentLoc = points(i).Location; %loops through each of the detected points
        minDist(i,1) = pdist([currentLoc; ExtrapolatedPoint(k,1,1) ExtrapolatedPoint(k,1,2)]); %[x-current, y-current; extrapolated-x, extrapolated-y], ; separates the row
        minDist(i,2) = pdist([currentLoc; ExtrapolatedPoint(k,2,1) ExtrapolatedPoint(k,2,2)]);
        minDist(i,3) = pdist([currentLoc; ExtrapolatedPoint(k,3,1) ExtrapolatedPoint(k,3,2)]);   
    end

    [Loc, ind] = min(minDist);
%     ind(3) = 37;
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

    frame = 1;
    numSamples = size(points,1);

    for i = 1:numSamples+1
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
                CropFrame = imcrop(videoFrame, [(MyPoints(j,1)-cropvectors(j,1)-5) (MyPoints(j,2)-cropvectors(j,2)-5) 11 11]); %crop patch
                writeVideo(Crop(j), CropFrame);
            end
            CropFrame = imcrop(videoFrame, [5 5 11 11]);
            writeVideo(Crop(4), CropFrame);
            %[274 899] RED LED in JoanneSmall.avi
            redLED(frame-1) = mean(videoFrame(LEDY, LEDX, :));
            videoFrame = insertMarker(videoFrame, [LEDX LEDY], '+', 'Color', 'red');

            videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
                'Color', 'white');
            videoFrame = insertMarker(videoFrame, MyPoints, '+', ...
                'Color', 'green');
            videoFrame = insertMarker(videoFrame, [10 10], '+', ...
                'Color', 'green');
            
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
    for i = 1:numSamples+1
        close(Crop(i));
    end
    
    %% Only run this if using the Red LED
    [LEDpeaks, LEDlocs] = findpeaks(double(redLED),'MINPEAKDISTANCE',validationMinPeakDist, 'MINPEAKHEIGHT', 0.5);
    figure('name','Red LED signal');
    actualPeaks = size(LEDpeaks,2);
    actualBPM = 60*actualPeaks*30/numFrames
    plot(redLED);
    hold on;
    scatter(LEDlocs, redLED(LEDlocs));
    ylim([0, 1]);

%     actualPeaks = 25;
    
    %%Initialize Optimization
    flow0 = 65/60;
    fhigh0 = 80/60;
    
    flow = flow0;
    fhigh = fhigh0;
    %%Uncomment to optimize for the high-low frequnecies
%     x0 = [ flow0; fhigh0];
%     A = [ 1 0;
%         -1 0;
%         0 1;
%         0 -1;
%         1 -1];
%     b = [3; -0.4; 40; -0.62; -0.05];
%     
%     options = optimoptions('fmincon');
% %     options.MaxFunEvals = 40;
%     options.DiffMinChange = 0.1;
%     options.MaxIter = 2;
%     options.Display = 'iter';
%     
%     
%     X = fmincon(@(x) objectiveFunction(x, LEDlocs), x0, A, b, [], [],[],[],[],options);
%     X
%     alpha = 50;
%     flow = X(1);
%     fhigh = X(2);
    
    alpha = 50;
    for i = 1:numSamples+1
        %% Run MIT code on cropped video
        filename = strcat(strcat(strcat(infileName,'Crop'),num2str(i)),'.avi');
        inFile_Sample = fullfile(resultsDir,filename);
        fprintf('face %i of %i, sample %i of %i, filter %i of %i\n', k, numFaces, i, numSamples+1 , 1, 1);
        amplify_spatial_Gdown_temporal_ideal(inFile_Sample,resultsDir,alpha,2,flow,fhigh,30, 0);
    end
    
    numPeaksG = zeros(numFaces, numSamples+1);
    pulseG = zeros(numFaces, numSamples+1);
    mean_r = zeros(numSamples+1,numFrames);
    mean_g = zeros(numSamples+1,numFrames);
    mean_b = zeros(numSamples+1,numFrames);
    for i = 1:numSamples+1
%         fig1 = figure('name',strcat('Processed heartbeat from sample ', num2str(i), ' for face', num2str(k)));
        G = zeros(1,numFrames);
        rangeString = strcat(num2str(flow),'-to-',num2str(fhigh));
        filename = strcat(infileName, 'Crop', num2str(i), '-ideal-from-',rangeString,'-alpha-',num2str(alpha),'-level-2-chromAtn-0.avi');
        inFile_Processed = fullfile(resultsDir,filename);

        videoFileReader = vision.VideoFileReader(inFile_Processed);
        videoFrame = step(videoFileReader);
        frame = 1;

        G(frame) = mean(mean(videoFrame(:,:,1)));
        
        mean_r(i,frame) = mean(mean(videoFrame(:,:,1)));
        mean_g(i,frame) = mean(mean(videoFrame(:,:,2)));
        mean_b(i,frame) = mean(mean(videoFrame(:,:,3)));

        while ~isDone(videoFileReader)
            videoFrame = step(videoFileReader);
            frame = frame+1;
            G(frame) = mean(mean(videoFrame(:,:,1)));
            
            mean_r(i,frame) = mean(mean(videoFrame(:,:,1)));
            mean_g(i,frame) = mean(mean(videoFrame(:,:,2)));
            mean_b(i,frame) = mean(mean(videoFrame(:,:,3)));
        end
        G = G(1:find(G,1,'last')); %trim zeros
        G = filter(ones(1,7)/7,1,G); %smooth
        
        fig1 = figure('name',strcat('Processed heartbeat from sample ', num2str(i), ' for face', num2str(k)));
        plot(1:size(G,2),G(:),'color',[1 0 0]);
        hold on;
        ylim([0, 1]);
        legend(strcat(num2str(flow*60),' to ', num2str(fhigh*60)));
        [peaks, locs] = findpeaks(G,'MINPEAKDISTANCE',10);
        scatter(locs, G(locs));
        numPeaksG(k,i) = size(peaks,2);
        pulseG(k,i) = size(peaks,2)*60*30/size(G,2);
    end
    mean_r = mean_r(1:numSamples,:);
    mean_g = mean_g(1:numSamples,:);
    mean_b = mean_b(1:numSamples,:);
    
    % ICA (assume 9 signals)
    addpath('./FastICA_2.5');
    sig = [mean_r(1,:); mean_r(2,:); mean_r(3,:); mean_g(1,:); mean_g(2,:); mean_g(3,:); mean_b(1,:); mean_b(2,:); mean_b(3,:)];
%     sig = [mean_r(2,:); mean_r(3,:); mean_g(2,:); mean_g(3,:); mean_b(2,:); mean_b(3,:)];
%     sig = [mean_r(1,:); mean_g(1,:); mean_b(1,:)];
    [decomp] = fastica(sig);
%     [decomp] = jadeR(sig,3);
%     decomp = decomp*sig;
    
    F = fft(decomp,[],2);
    
    figure;
    plot(1:(size(F,2)-1),abs(F(:,2:end)));
    
    fps = 30;
    f = fps/size(F,2) * (0:floor(size(F,2)/2)-1);
    
    [amp,freq] = max(abs(F(:,2:end)),[],2);
    ICA_post = f(freq+1)*60
    
    [~,pulseInd] = max(amp);
    pulse = f(freq(pulseInd)+1)*60
    
%     ICA_post1 = f(freq+1)*60;
%     
%     
%     
%     sig = [mean_r(2,:); mean_g(2,:); mean_b(2,:)];
%     [decomp] = fastica(sig);
%     
%     F = fft(decomp,[],2);
%     
%     figure;
%     plot(1:(size(F,2)-1),abs(F(:,2:end)));
%     
%     fps = 30;
%     f = fps/size(F,2) * (0:floor(size(F,2)/2)-1);
%     
%     [~,freq] = max(abs(F(:,2:end)),[],2);
%     ICA_post2 = f(freq+1)*60;
%     
%     
%     
%     sig = [mean_r(3,:); mean_g(3,:); mean_b(3,:)];
%     [decomp] = fastica(sig);
%     
%     F = fft(decomp,[],2);
%     
%     figure;
%     plot(1:(size(F,2)-1),abs(F(:,2:end)));
%     
%     fps = 30;
%     f = fps/size(F,2) * (0:floor(size(F,2)/2)-1);
%     
%     [~,freq] = max(abs(F(:,2:end)),[],2);
%     ICA_post3 = f(freq+1)*60;
% 
%     ICA_post = [ICA_post1; ICA_post2; ICA_post3]
    

    numPeaksG
    pulseG
%     figure;
%     plot(redLED);
%     ylim([0, 1]);
%     xlim([0 500]);
%     for j = 1:numSamples
%        figure('name',strcat('Unaltered intensity for point ', num2str(j)));
%        plot(original(j,:));
%        ylim([0, 1]);
%        xlim([0 500]);
%     end
end
