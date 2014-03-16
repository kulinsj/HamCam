%crop
infileName = 'audreybeforeBetter';
inFile = fullfile(dataDir,strcat(infileName,'.avi'));
% inFile = fullfile(dataDir,strcat(infileName,'.mp4'));
outfile = fullfile(dataDir,strcat(infileName,'Better'));

videoFileReader = vision.VideoFileReader(inFile);

%% Use this to find the coords for the crop
videoFrame = step(videoFileReader);
f=figure;
imshow(videoFrame);
set(f,'WindowButtonDownFcn',@mytestcallback)
% 
% figure;
% CropFrame = imcrop(videoFrame, [320 102 586-385 497-102]);
% imshow(CropFrame);

% Crop = VideoWriter(outfile);
% open(Crop);
% while ~isDone(videoFileReader)
%     videoFrame = step(videoFileReader);
%     CropFrame = imcrop(videoFrame, [320 112 586-385 497-102]);
%     writeVideo(Crop, CropFrame);
% end
% close(Crop);
% release(videoFileReader);
% disp('done');