close all;
clear;
clc;

dataDir = './data';
resultsDir = 'Results';
hOffset = 1.7;

rangeRadius = 15;
Results = zeros(8,2);
for i = 1:8
   fprintf('----------------------');
   fprintf('TEST: on %i of 8 \n', i);
   switch i
       case 1
           infile = 'audreybeforeBetter';
           LEDX = 64;
           LEDY = 360;
           validationMinPeakDist = 10;
           expected = 70;
       case 2
           infile = 'audreyafterBetter';
           LEDX = 64;
           LEDY = 397;
           validationMinPeakDist = 7;
           expected = 156.5;
       case 3
           infile = 'janbeforeBetter';
           LEDX = 98;
           LEDY = 443;
           validationMinPeakDist = 15;
           expected = 59;
       case 4
           infile = 'janafterBetter';
           LEDX = 161;
           LEDY = 444;
           validationMinPeakDist = 7;
           expected = 133.7;
       case 5
           infile = 'joannebeforeBetter';
           LEDX = 160;
           LEDY = 532;
           validationMinPeakDist = 10;
           expected = 82;
       case 6
           infile = 'joanneafterBetter';
           LEDX = 141;
           LEDY = 531;
           validationMinPeakDist = 7;
           expected = 132.6;
       case 7
           infile = 'janbefore2Better';
           LEDX = 140;
           LEDY = 515;
           validationMinPeakDist = 15;
           expected = 52.4;
           hOffset = 1.6;
       case 8
           infile = 'janafter2Better';
           LEDX = 189;
           LEDY = 515;
           validationMinPeakDist = 7;
           expected = 124;
           hOffset = 1.7;
   end
   [Results(i,1), Results(i,2)] = HamCam(infile, LEDX, LEDY, validationMinPeakDist, expected, rangeRadius, hOffset);
end
Results
errorActual = zeros(1,8);
errorPercent = cell(1,8);
for i = 1:8
    errorActual(i) = Results(i,2) - Results(i,1);
    errorPercent{i} = strcat(num2str(round(10*abs(errorActual(i))*100/Results(i,1))/10),'%');
end
errorActual
errorPercent
TotalError = sum(errorActual)
ErrorVariance = var(errorActual)
