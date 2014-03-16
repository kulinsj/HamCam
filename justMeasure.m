%crop
infileName = 'lighttestBetter';
inFile = fullfile(dataDir,strcat(infileName,'.avi'));
% inFile = fullfile(dataDir,strcat(infileName,'.mp4'));

videoFileReader = vision.VideoFileReader(inFile);

frame = 1;
while ~isDone(videoFileReader)
    videoFrame = step(videoFileReader);
    G(frame) = mean(videoFrame(22,19));
    frame = frame +1;
end
[LEDpeaks, LEDlocs] = findpeaks(double(G),'MINPEAKDISTANCE',validationMinPeakDist, 'MINPEAKHEIGHT', 0.5);
plot(G);
hold on;
scatter(LEDlocs, G(LEDlocs));
release(videoFileReader);
disp('done');
peaks = size(LEDlocs,2)
bpm = peaks*60*30/frame