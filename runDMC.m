dataDir = './data';
resultsDir = 'ResultsSIGGRAPH2012';

mkdir(resultsDir);


% inFile = fullfile(dataDir,'tiny.mp4');
inFile = fullfile(dataDir,'JoanneSmallCrop.avi');
fprintf('Processing %s\n\n', inFile);
%amplify_spatial_lpyr_temporal_ideal(inFile, resultsDir, 50, 5, 40/60, 100/60, 100, 0);
%fprintf('Processing %s\n', inFile);
amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,50,2,72/60,98/60,30, 0);

%butter(vidFile, outDir ,alpha, lambda_c, fl, fh ,samplingRate, chromAttenuation)

% amplify_spatial_lpyr_temporal_ideal(vidFile, outDir, alpha, lambda_c, wl, wh, samplingRate, chromAttenuation)