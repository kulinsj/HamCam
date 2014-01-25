function plotsize(x,m);
%function plotsize(x,m);
%
%INPUT:
%x - image
%m - min and max face size in vector, see 3rd output from facefind.m

minf=m(1);
maxf=m(2);

ex1=size(x,1)*0.01;
ex1e=size(x,1)*0.02;
ex2=size(x,1)*0.04;
ex2e=size(x,1)*0.05;
bx1=[0 maxf maxf 0];
by1=[ex1e ex1e ex1 ex1];
bx2=[0 minf minf 0];
by2=[ex2e ex2e ex2 ex2];

hold on
fill(bx1,by1,[0 1 0])
fill(bx2,by2,[0 1 0])
hold off