function plotbox(output,col,t);
%function plotbox(output,col,t);
%
%INPUT:
%output - output from face dection (see facefind.m)
%col    - color for detection boxes [default [1 0 0]]
%t      - thickness of lines [default 2]

if ((nargin<2) | isempty(col))
    col=[1 0 0];
end

if ((nargin<3) | isempty(t))
    t=2;
end

N=size(output,2);

if (N>0)
    for i=1:N
        x1=output(1,i);
        x2=output(2,i);
        y1=output(3,i);
        y2=output(4,i);
        boxv([x1 x2 y1 y2],col,t)
    end
end

function boxv(vec,col,t);
%function boxv(vec,col,t);
%
%INPUT:
%vec - vector with corner coordiantes ([x1 x2 y1 y2]) for a box
%col - color [default [1 0 0]]
%t   - thickness of lines [default 0.5]

if ((nargin<2) | isempty(col))
    col=[1 0 0];
end

if ((nargin<3) | isempty(t))
    t=0.5;
end

ind=find(isinf(vec));%special case if coordinate is Inf
a=200;%should be realmax, but bug in Matlab? (strange lines occur)
vec(ind)=sign(vec(ind))*a;

h1=line([vec(1) vec(2)],[vec(3) vec(3)]);
h2=line([vec(2) vec(2)],[vec(3) vec(4)]);
h3=line([vec(1) vec(2)],[vec(4) vec(4)]);
h4=line([vec(1) vec(1)],[vec(3) vec(4)]);

h=[h1 h2 h3 h4];
set(h,'Color',col);

set(h,'LineWidth',t)