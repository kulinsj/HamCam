function error = objectiveFunction(x, LEDLocations)
    % x(1) = flow
    % x(2) = fhigh
    % x(3) = numSamples
    % x(4) = actualPeaks
    x
    resultsDir = 'Results';
    infileName = 'JoanneSmall';
    
    alpha = 50;
    flow = x(1);
    fhigh = x(2);
    numSamples = 3;
    actualPeaks = size(LEDLocations,2);
    
    numPeaksG = zeros(1, numSamples+1);
    pulseG = zeros(1, numSamples+1);
    
    error = 0;

    for i = 1:numSamples+1
        % Run MIT code on cropped video
        filename = strcat(strcat(strcat(infileName,'Crop'),num2str(i)),'.avi');
        inFile_Sample = fullfile(resultsDir,filename);
        amplify_spatial_Gdown_temporal_ideal(inFile_Sample,resultsDir,alpha,2,flow,fhigh,30, 0);
        
        rangeString = strcat(num2str(flow),'-to-',num2str(fhigh));
        filename = strcat(infileName, 'Crop', num2str(i), '-ideal-from-',rangeString,'-alpha-',num2str(alpha),'-level-2-chromAtn-0.avi');
        inFile_Processed = fullfile(resultsDir,filename);

        videoFileReader = vision.VideoFileReader(inFile_Processed);
        videoFrame = step(videoFileReader);
        frame = 1;
        G(frame) = mean(mean(videoFrame(:,:,1)));
        
        while ~isDone(videoFileReader)
            videoFrame = step(videoFileReader);
            frame = frame+1;
            G(frame) = mean(mean(videoFrame(:,:,1)));
        end
        G = G(1:find(G,1,'last')); %trim zeros
        G = filter(ones(1,7)/7,1,G); %smooth
        
        [peaks, peakLocs] = findpeaks(double(G),'MINPEAKDISTANCE',10);
        numPeaksG(i) = size(peaks,2);
        pulseG(i) = size(peaks,2)*60*30/size(G,2);
%         LEDLocations'
%         peakLocs'
        k = dsearchn(LEDLocations', peakLocs');
        for j = 1:size(peakLocs,2)
            error = error + (abs(LEDLocations(k(j))-peakLocs(j)))^2;
        end
        error = error + (abs(actualPeaks - numPeaksG(i)))^3;
%         error = error + (actualPeaks - numPeaksG(i))^2;
    end
    error
end