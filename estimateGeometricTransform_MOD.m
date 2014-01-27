function [tform, inlier_points1, inlier_points2, newTrackedPoints, status] ...
    = estimateGeometricTransform_MOD(matched_points1, matched_points2, ...
    TrackedPoints, transform_type, varargin)
%estimateGeometricTransform Estimate geometric transformation from matching point pairs.
%   TFORM = estimateGeometricTransform(MATCHED_POINTS1,MATCHED_POINTS2,
%   TRANSFORM_TYPE) returns a 2-D geometric transform which maps the
%   inliers in MATCHED_POINTS1 to the inliers in MATCHED_POINTS2.
%   MATCHED_POINTS1 and MATCHED_POINTS2 can be cornerPoints objects,
%   SURFPoints objects, MSERRegions objects, or M-by-2 matrices of [x,y]
%   coordinates. TRANSFORM_TYPE can be 'similarity', 'affine', or
%   'projective'. Outliers in MATCHED_POINTS1 and MATCHED_POINTS2 are
%   excluded by using the M-estimator SAmple Consensus (MSAC) algorithm.
%   The MSAC algorithm is a variant of the Random Sample Consensus (RANSAC)
%   algorithm. The returned TFORM is an affine2d object if TRANSFORM_TYPE
%   is set to 'similarity' or 'affine', and is a projective2d object
%   otherwise.
%
%   [TFORM,INLIER_POINTS1,INLIER_POINTS2] = estimateGeometricTransform(...)
%   additionally returns the corresponding inlier points in INLIER_POINTS1
%   and INLIER_POINTS2.
%
%   [TFORM,INLIER_POINTS1,INLIER_POINTS2,STATUS] =
%   estimateGeometricTransform(...) additionally returns a status code with
%   the following possible values:
% 
%     0: No error. 
%     1: MATCHED_POINTS1 and MATCHED_POINTS2 do not contain enough points.
%     2: Not enough inliers have been found.
%
%   When the STATUS output is not given, the function will throw an error
%   if MATCHED_POINTS1 and MATCHED_POINTS2 do not contain enough points or
%   if not enough inliers have been found.
%
%   [...] = estimateGeometricTransform(MATCHED_POINTS1,MATCHED_POINTS2, 
%   TRANSFORM_TYPE,Name,Value) specifies additional
%   name-value pair arguments described below:
%
%   'MaxNumTrials'        A positive integer scalar specifying the maximum
%                         number of random trials for finding the inliers.
%                         Increasing this value will improve the robustness
%                         of the output at the expense of additional
%                         computation.
% 
%                         Default value: 1000
%  
%   'Confidence'          A numeric scalar, C, 0 < C < 100, specifying the
%                         desired confidence (in percentage) for finding
%                         the maximum number of inliers. Increasing this
%                         value will improve the robustness of the output
%                         at the expense of additional computation.
%
%                         Default value: 99
% 
%   'MaxDistance'         A positive numeric scalar specifying the maximum
%                         distance in pixels that a point can differ from
%                         the projection location of its associated point.
% 
%                         Default value: 1.5
% 
%   Class Support
%   -------------
%   MATCHED_POINTS1 and MATCHED_POINTS2 must be cornerPoints objects,
%   SURFPoints objects, MSERRegions objects, or M-by-2 matrices of [x,y]
%   coordinates.
%
%   % EXAMPLE: Recover a transformed image using SURF feature points
%   % --------------------------------------------------------------
%   Iin  = imread('cameraman.tif'); imshow(Iin); title('Base image');
%   Iout = imresize(Iin, 0.7); Iout = imrotate(Iout, 31);
%   figure; imshow(Iout); title('Transformed image');
%  
%   % Detect and extract features from both images
%   ptsIn  = detectSURFFeatures(Iin);
%   ptsOut = detectSURFFeatures(Iout);
%   [featuresIn,   validPtsIn] = extractFeatures(Iin,  ptsIn);
%   [featuresOut, validPtsOut] = extractFeatures(Iout, ptsOut);
%  
%   % Match feature vectors
%   index_pairs = matchFeatures(featuresIn, featuresOut);
%   matchedPtsIn  = validPtsIn(index_pairs(:,1));
%   matchedPtsOut = validPtsOut(index_pairs(:,2));
%   figure; showMatchedFeatures(Iin,Iout,matchedPtsIn,matchedPtsOut);
%   title('Matched SURF points, including outliers');
%  
%   % Exclude the outliers and compute the transformation matrix
%   [tform,inlierPtsOut,inlierPtsIn] = estimateGeometricTransform(...
%        matchedPtsOut,matchedPtsIn,'similarity');
%   figure; showMatchedFeatures(Iin,Iout,inlierPtsIn,inlierPtsOut);
%   title('Matched inlier points');
%  
%   % Recover the original image Iin from Iout
%   outputView = imref2d(size(Iin));
%   Ir = imwarp(Iout, tform, 'OutputView', outputView);
%   figure; imshow(Ir); title('Recovered image');
%
% See also cp2tform, cornerPoints, detectMinEigenFeatures,
% detectFASTFeatures, detectSURFFeatures, detectMSERFeatures,
% extractFeatures, matchFeatures, imwarp

% References:
% [1] R. Hartley, A. Zisserman, "Multiple View Geometry in Computer
%     Vision," Cambridge University Press, 2003.
% [2] P. H. S. Torr and A. Zisserman, "MLESAC: A New Robust Estimator
%     with Application to Estimating Image Geometry," Computer Vision
%     and Image Understanding, 2000.

% Copyright  The MathWorks, Inc.
% $Revision: 1.1.6.2 $  $Date: 2012/11/15 14:56:27 $

%#codegen
%#ok<*EMCA>

% List of status code
statusCode = struct(...
    'NoError',           int32(0),...
    'NotEnoughPts',      int32(1),...
    'NotEnoughInliers',  int32(2));

% Parse and check inputs
[points1,points2,sampleSize,maxNumTrials,confidence,threshold,status] ...
    = parseInputs(statusCode, matched_points1, matched_points2, ...
        TrackedPoints, transform_type, varargin{:});
    
classToUse = getClassToUse(points1, points2);

% Compute the geometric transformation
if status == statusCode.NoError
    [isFound, tmatrix, inliers, outliers] = msac(sampleSize, maxNumTrials, ...
        confidence, threshold, points1, points2, classToUse);
    if ~isFound
        status = statusCode.NotEnoughInliers;
    end
else
    tmatrix = zeros([3,3], classToUse);
end

% Extract inlier points
if status == statusCode.NoError
    inlier_points1 = matched_points1(inliers, :);
    inlier_points2 = matched_points2(inliers, :);
    
    %adjust indecies of tracked points based on removed outliers
    currentNumTrackedPoints = size(TrackedPoints);
    lossSoFar = 0;
    newTrackedPoints = zeros(currentNumTrackedPoints(1),1);
    for i = 1:currentNumTrackedPoints(1)
        if i == 1
            os = outliers(1:TrackedPoints(1));
        else
            os = outliers((TrackedPoints(i-1)-1):TrackedPoints(i));
        end
        thisOSOutliers = sum(os(:));
        lossSoFar = lossSoFar+thisOSOutliers;
        if outliers(TrackedPoints(i))==1
            fprintf('point lost\n');
            newTrackedPoints(i) = -1;
        else
            newTrackedPoints(i) = TrackedPoints(i) - lossSoFar;
        end
    end
    
else
    inlier_points1 = matched_points1([]);
    inlier_points2 = matched_points2([]);
    tmatrix = zeros([3,3], classToUse);
end

% Report runtime error if the status output is not requested
reportError = (nargout ~= 4);
if reportError
    checkRuntimeStatus(statusCode, status);
end

if isTestingMode()
    % Return tform as a matrix for internal testing purposes
    tform = tmatrix;
else
    if sampleSize < 4  % similarity or affine
        tform = affine2d(tmatrix);
    else               % projective
        tform = projective2d(tmatrix);
    end
end

%==========================================================================
% Check runtime status and report error if there is one
%==========================================================================
function checkRuntimeStatus(statusCode, status)
coder.internal.errorIf(status==statusCode.NotEnoughPts, ...
    'vision:estimateGeometricTransform:notEnoughPts');

coder.internal.errorIf(status==statusCode.NotEnoughInliers, ...
    'vision:estimateGeometricTransform:notEnoughInliers');

%==========================================================================
% Parse and check inputs
%==========================================================================
function [points1, points2, sampleSize, maxNumTrials, confidence, ...
    maxDistance, status] ...
    = parseInputs(statusCode, matched_points1, matched_points2, ...
    TrackedPoints, transform_type, varargin)

isSimulationMode = isempty(coder.target);
if isSimulationMode
    % Instantiate an input parser
    parser = inputParser;
    parser.FunctionName = 'estimateGeometricTransform';
    parser.CaseSensitive = true;
    
    % Specify the optional parameters
    parser.addParamValue('MaxNumTrials', 1000);
    parser.addParamValue('Confidence',   99);
    parser.addParamValue('MaxDistance',  1.5);
    
    % Parse and check optional parameters
    parser.parse(varargin{:});
    r = parser.Results;
    
    maxNumTrials = r.MaxNumTrials;
    confidence   = r.Confidence;
    maxDistance  = r.MaxDistance;
    
else
    % Instantiate an input parser
    parms = struct( ...
        'MaxNumTrials',       uint32(0), ...
        'Confidence',         uint32(0), ...
        'MaxDistance',        uint32(0));
    
    popt = struct( ...
        'CaseSensitivity', true, ...
        'StructExpand',    true, ...
        'PartialMatching', false);
    
    % Specify the optional parameters
    optarg       = eml_parse_parameter_inputs(parms, popt,...
        varargin{:});
    maxNumTrials = eml_get_parameter_value(optarg.MaxNumTrials,...
        1000, varargin{:});
    confidence   = eml_get_parameter_value(optarg.Confidence,...
        99, varargin{:});
    maxDistance  = eml_get_parameter_value(optarg.MaxDistance,...
        1.5, varargin{:});
end

% Check required parameters
sampleSize = checkTransformType(transform_type);
points1 = checkAndConvertPoints(matched_points1);
points2 = checkAndConvertPoints(matched_points2);
status  = checkPointsSize(statusCode, sampleSize, points1, points2);

% Check optional parameters
checkMaxNumTrials(maxNumTrials);
checkConfidence(confidence);
checkMaxDistance(maxDistance);

%==========================================================================
function status = checkPointsSize(statusCode, sampleSize, points1, points2)

coder.internal.errorIf( size(points1,1) ~= size(points2,1), ...
    'vision:estimateGeometricTransform:numPtsMismatch');

coder.internal.errorIf( ~isequal(class(points1), class(points2)), ...
    'vision:estimateGeometricTransform:classPtsMismatch');

if size(points1,1) < sampleSize
    status = statusCode.NotEnoughPts;
else
    status = statusCode.NoError;
end

%==========================================================================
function points = checkAndConvertPoints(matched_points)
if isnumeric(matched_points)
    checkPointsAttributes(matched_points);
    points = matched_points;
elseif isa(matched_points, 'vision.internal.FeaturePoints')
    points = matched_points.Location;
else % MSERRegions
    points = matched_points.Centroid;
end

%==========================================================================
function checkPointsAttributes(value)
validateattributes(value, {'numeric'}, ...
    {'2d', 'nonsparse', 'real', 'size', [NaN, 2]},...
    'estimateGeometricTransform', 'MATCHED_POINTS');

%==========================================================================
function r = checkMaxNumTrials(value)
validateattributes(value, {'numeric'}, ...
    {'scalar', 'nonsparse', 'real', 'integer', 'positive', 'finite'},...
    'estimateGeometricTransform', 'MaxNumTrials');
r = 1;

%========================================================================== 
function r = checkConfidence(value)
validateattributes(value, {'numeric'}, ...
    {'scalar', 'nonsparse', 'real', 'positive', 'finite', '<', 100},...
    'estimateGeometricTransform', 'Confidence');
r = 1;

%==========================================================================
function r = checkMaxDistance(value)
validateattributes(value, {'numeric'}, ...
    {'scalar', 'nonsparse', 'real', 'positive', 'finite'},...
    'estimateGeometricTransform', 'MaxDistance');
r = 1;

%==========================================================================
function sampleSize = checkTransformType(value)
list = {'similarity', 'affine', 'projective'};
validatestring(value, list, 'estimateGeometricTransform', ...
    'TransformType');

switch(value(1))
    case 's'
        sampleSize = 2;
    case 'a'
        sampleSize = 3;
    otherwise
        sampleSize = 4;
end

%==========================================================================
function c = getClassToUse(points1, points2)
if isa(points1, 'double') || isa(points2, 'double')
    c = 'double';
else
    c = 'single';
end

%==========================================================================
function flag = isTestingMode
isSimulationMode = isempty(coder.target);
coder.extrinsic('vision.internal.testEstimateGeometricTransform');
if isSimulationMode
    flag = vision.internal.testEstimateGeometricTransform;
else
    flag = eml_const(vision.internal.testEstimateGeometricTransform);
end

%==========================================================================
% Algorithm for computing the fundamental matrix.
%==========================================================================
function T = computeSimilarity(points1, points2, classToUse)
numPts = size(points1, 1);
constraints = zeros(2*numPts, 5, classToUse);
constraints(1:2:2*numPts, :) = [-points1(:, 2), points1(:, 1), ...
    zeros(numPts, 1), -ones(numPts,1), points2(:,2)];
constraints(2:2:2*numPts, :) = [points1, ones(numPts,1), ...
    zeros(numPts, 1), -points2(:,1)];
[~, ~, V] = svd(constraints, 0);
h = V(:, end);
T = coder.nullcopy(zeros(3, classToUse));
T(:, 1:2) = [h(1:3), [-h(2); h(1); h(4)]] / h(5);
T(:, 3)   = [0; 0; 1];

%==========================================================================
function T = computeAffine(points1, points2, classToUse)
numPts = size(points1, 1);
constraints = zeros(2*numPts, 7, classToUse);
constraints(1:2:2*numPts, :) = [zeros(numPts, 3), -points1, ...
    -ones(numPts,1), points2(:,2)];
constraints(2:2:2*numPts, :) = [points1, ones(numPts,1), ...
    zeros(numPts, 3), -points2(:,1)];
[~, ~, V] = svd(constraints, 0);
h = V(:, end);
T = coder.nullcopy(zeros(3, classToUse));
T(:, 1:2) = reshape(h(1:6), [3,2]) / h(7);
T(:, 3)   = [0; 0; 1];

%==========================================================================
function T = computeProjective(points1, points2, classToUse)
numPts = size(points1, 1);
p1x = points1(:, 1);
p1y = points1(:, 2);
p2x = points2(:, 1);
p2y = points2(:, 2);
constraints = zeros(2*numPts, 9, classToUse);
constraints(1:2:2*numPts, :) = [zeros(numPts,3), -points1, ...
    -ones(numPts,1), p1x.*p2y, p1y.*p2y, p2y];
constraints(2:2:2*numPts, :) = [points1, ones(numPts,1), ...
    zeros(numPts,3), -p1x.*p2x, -p1y.*p2x, -p2x];
[~, ~, V] = svd(constraints, 0);
h = V(:, end);
T = reshape(h, [3,3]) / h(9);

%==========================================================================
function num = computeLoopNumber(sampleSize, confidence, pointNum, inlierNum)
num = int32(ceil(log10(1 - 0.01 * confidence)...
    / log10(1 - (inlierNum/pointNum)^sampleSize)));

%==========================================================================
function [ptsNorm, normMatrix] = normalizePts(pts, classToUse)
ptsNorm = cast(pts, classToUse);
cent = mean(ptsNorm, 1);
ptsNorm(:, 1) = ptsNorm(:, 1) - cent(1);
ptsNorm(:, 2) = ptsNorm(:, 2) - cent(2);

weight = std(ptsNorm(:),[],1);
if weight > 0
    weight = sqrt(2) / weight;
else
    weight = ones(1, classToUse);  % Just pick a value
end

ptsNorm(:, 1) = ptsNorm(:, 1) * weight;
ptsNorm(:, 2) = ptsNorm(:, 2) * weight;

normMatrix = [...
    1/weight,     0,            0;...
    0,            1/weight,     0;...
    cent(1),      cent(2),      1];

%==========================================================================
function tform = computeTForm(sampleSize, points1, points2, indices, classToUse)
[samples1, normMatrix1] = normalizePts(points1(indices, :), classToUse);
[samples2, normMatrix2] = normalizePts(points2(indices, :), classToUse);

switch(sampleSize)
    case 2
        tform = computeSimilarity(samples1, samples2, classToUse);
    case 3
        tform = computeAffine(samples1, samples2, classToUse);
    otherwise % 4
        tform = computeProjective(samples1, samples2, classToUse);
        tform = tform / tform(end);
end
tform = normMatrix1 \ (tform * normMatrix2);

%==========================================================================
function dis = evaluateTForm(sampleSize, threshold, tform, point1, point2, classToUse)
pt1 = cast(point1, classToUse);
pt2 = cast(point2, classToUse);
pt = pt1 * tform(1:2, 1:2) + tform(3, 1:2);
if sampleSize == 4
    denom = pt1 * tform(1:2, 3) + tform(3, 3);
    if abs(denom) > eps(classToUse)
        pt = pt ./ denom;
    else % Mark this point invalid by setting it to a location far away from pt2
        pt(:) = pt2 + threshold;
    end
end
dis = norm(pt - pt2);
dis = min(dis, threshold);

%==========================================================================
function [isFound, tform, inliers, outliers] = msac(sampleSize, maxNumTrials, ...
    confidence, threshold, points1, points2, classToUse)

numPts = size(points1, 1);
idxTrial = 1;
numTrials = int32(maxNumTrials);
maxDis = cast(threshold * numPts, classToUse);
bestDis = maxDis;
bestTForm = eye([3,3], classToUse);

% Create a random stream. It uses a fixed seed for the testing mode and a
% random seed for other mode.
if isTestingMode()
    rng('default');
end

while idxTrial <= numTrials
    indices = randperm(numPts, sampleSize);
    tform = computeTForm(sampleSize, points1, points2, indices, classToUse);
    
    accDis = zeros(1, classToUse);
    idxPt = 1;
    while accDis < bestDis && idxPt <= numPts
        dis = evaluateTForm(sampleSize, threshold, tform, ...
            points1(idxPt, :), points2(idxPt, :), classToUse);
        accDis = accDis + dis;
        idxPt = idxPt + 1;
    end
    
    if accDis < bestDis
        bestDis = accDis;
        bestTForm = tform;
        inlierNum = numPts - bestDis / threshold;
        num = computeLoopNumber(sampleSize, confidence, numPts, inlierNum);
        numTrials = min(numTrials, num);
    end
    idxTrial = idxTrial + 1;
end

distances = zeros([1, numPts], classToUse);
for idxPt = 1: numPts
    distances(idxPt) = evaluateTForm(sampleSize, threshold, bestTForm, ...
        points1(idxPt, :), points2(idxPt, :), classToUse);
end
inliers = (distances < threshold);
outliers = (distances >= threshold);
isFound = (sum(inliers) >= sampleSize);

if isFound
    tform = computeTForm(sampleSize, points1, points2, inliers, classToUse);
    tform = tform / tform(3,3);
else
    tform = zeros([3,3], classToUse);
end
