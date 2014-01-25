%Simple demonstration of one image face detection operation.

clear
close all

x=imread('test01.jpg');
if (size(x,3)>1)%if RGB image make gray scale
    try
        x=rgb2gray(x);%image toolbox dependent
    catch
        x=sum(double(x),3)/3;%if no image toolbox do simple sum
    end
end
x=double(x);%make sure the input is double format
[output,count,m,svec]=facefind(x);%full scan 
imagesc(x), colormap(gray)%show image
plotbox(output,[],8)%plot the detections as red squares
plotsize(x,m)%plot minmum and maximum face size to detect as green squares in top left corner