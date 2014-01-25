%Webcam demo with face detection. Require VFM for Matlab-webcam interface.
%Download VFM from: http://www2.cmp.uea.ac.uk/~fuzz/vfm/default.html
%Make sure vfm.dll is in the same catalog as facefind.dll for proper webcam
%operation.
%Adjust the image size with VFM to 352x288 (in Configure->Formar) for
%realtime operation (exact speed is dependent on the computer system).

clear%observe that clear all may cause error (link to webcam lost) if VFM is activated at run start
close all

%Do extra check to check that VFM works
try
    x=vfm('grab');
catch
    disp('VFM not found (vfm.dll) or webcam not connected.')
    disp('Download VFM from: http://www2.cmp.uea.ac.uk/~fuzz/vfm/default.html')
    disp('If you forgot to connect your webcam: restart Matlab and run again.')
    break;
end

disp('Press CTRL-C to break.')

while 1
    tic
    x=vfm('grab');
    try
        x=rgb2gray(x);%image toolbox dependent
    catch
        x=sum(double(x),3)/3;%if no image toolbox do simple sum
    end
    x=double(x);

    hold off
    clf
    cla
    imagesc(x);colormap(gray)
    ylabel('Press CTRL-C to break.')
    [output,count,m]=facefind(x,48,[],2,2);%speed up detection, jump 2 pixels and set minimum face to 48 pixels
    
    plotsize(x,m)
    plotbox(output)
    drawnow;drawnow;

    t=toc;
    %note that the FPS calculation is including grabbing the image and displaying the
    %image and detection
    title(['Frames Per Second (FPS): ' num2str(1/t) '  Number of patches analyzed: ' num2str(count)])
    drawnow;drawnow;
end