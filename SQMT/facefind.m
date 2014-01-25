function [output,count,m,svec]=facefind(x,minf,maxf,dx,dy,sens);
%function [output,count,m,svec]=facefind(x,minf,maxf,dx,dy,sens);
%
%INPUT:
%x      - grayscale image, double format (typical in range 0-1)
%minf   - minimun face size to find [default 32]
%maxf   - maximum face size to find [default inf]
%dx     - number of pixel jump in x-direction [default 1]
%dy     - number of pixel jump in y-direction [default 1]
%sens   - sensitivity of detection on a scale from 1 to 10 [default 5]
%         sens = 1  -> misses fewer faces, more false accepted faces
%         sens = 10 -> misses more faces, fewer false accepted faces
%
%OUTPUT:
%output - matrix with positions and scale information for 
%         found faces
%         output(:,1) - [x1 x2 y1 y2].' 
%count  - number of patches evaluated for detection
%m      - true (recalculated) minf and maxf m=[minf maxf]
%svec   - scales of image used [x_s;y_s] (s - scale index)
%
%INFO:
%The input parameters minf, maxf, dx, dy and sens will get their 
%default value if an empty matrix is given as input. For example: 
%[output,count,m,svec]=facefind(x,[],[],[],[],7);
%will change the sensitivity value to 7 but keep the default 
%values for minf, maxf, dx and dy.
%
%EXAMPLE:
%x=imread('test01.jpg');%read the image to memory
%if (size(x,3)>1)%if RGB image make gray scale
%    try
%        x=rgb2gray(x);%image toolbox dependent
%    catch
%        x=sum(double(x),3)/3;%if no image toolbox do simple sum
%    end
%end
%x=double(x);%make sure the input is in double format
%[output,count,m,svec]=facefind(x);%run default face detection (full scan)
%imagesc(x), colormap(gray)%show image 
%plotbox(output)%show the detections as red squares
%plotsize(x,m)%draw the minimum and maximum face size boxes in top left