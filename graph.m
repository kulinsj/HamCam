dataDir = './data';
inFile = fullfile(dataDir,'JoanneSmallCrop-ideal-from-1.2-to-1.6333-alpha-50-level-2-chromAtn-0.avi');

videoFileReader = vision.VideoFileReader(inFile);
videoFrame      = step(videoFileReader);
frame = 1;
M = mean(mean(mean(videoFrame)));

G = [];
G(frame) = M;

while ~isDone(videoFileReader)
    videoFrame = step(videoFileReader);
    frame = frame+1;
    M = mean(mean(mean(videoFrame)));
    G(frame) = M;
end

fig1 = figure;
plot(1:frame,G(:),'color',[1.0 0.0 0.0]);