close all; 
clear all;
clc; 

epdata1 = [];
epdata2 = []; 
eptime1 = [];
eptime2 = []; 

for i=1:1:2
    fopen(strcat('mydata',int2str(i),'.txt'))
    fileId = fopen(strcat('mydata',int2str(i),'.txt'));
    epdata=fscanf(fileId,'%i');
    fclose(fileId);

    fopen(strcat('myTimedata',int2str(i) ,'.txt'))
    fileId = fopen(strcat('myTimedata',int2str(i) ,'.txt'));
    eptime=fscanf(fileId,'%i');
    fclose(fileId);
    
    if i==1
        eptime1 = eptime;
        epdata1 = epdata;
    end
    
    if i==2
        eptime2 = eptime;
        epdata2 = epdata;
    end
end


[pks1,locs1] = findpeaks(epdata1,'MINPEAKDISTANCE',25);
[pks2,locs2] = findpeaks(epdata2,'MINPEAKDISTANCE',25);

%calculate bpm1 
bts1 = size(pks1,1); %number of beats recorded
stime1 = eptime1(size(eptime1,1))/1000; %total time in seconds
bpm1 = round(bts1/stime1*60); 


%calculate bpm2 
bts2 = size(pks2,1); %number of beats recorded
stime2 = eptime2(size(eptime2,1))/1000; %total time in seconds
bpm2 = round(bts2/stime2*60); 

fig1 = figure;
str= sprintf('BPM = %d', bpm1);
plot(eptime1,epdata1,'b'); 
hold on;
plot(eptime1(locs1),pks1,'rv');
title(str);


fig2 = figure;
str= sprintf('BPM = %d', bpm2);
plot (eptime2, epdata2,'m');
hold on;
plot(eptime2(locs2),pks2,'gv');
title(str);



%easypulse = cat(2,epdata,eptime);
%Info on analogRead for arduino: http://arduino.cc/en/Reference/analogRead#.UyTd3vldWGc