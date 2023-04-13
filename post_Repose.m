clear all;close;clc

folder='build/DemoOutput_Repose/';
files=dir(folder);

file=files(end).name;

data=readtable([folder file]);

x=data.X;
y=data.Y;
data.Z=data.Z-min(data.Z);
data.Z=data.Z+data.r;

dx=0.0033*2;
path=(-0.40:dx:0.40)';

[X,Y] = meshgrid(path,path);
Z=X*0;
range=dx*2;
for i=1:numel(path)      
    for j=1:numel(path) 
    xLocal=X(i,j);    
    ylocal=Y(i,j);
    index=find(abs(x-xLocal)<range & abs(y-ylocal)<range);
    if ~isempty(index)
        temp=max(data.Z(index));
    Z(i,j)=temp;
    end
    end
end

figure(1)
surf(X,Y,Z)



X_idx = find(path >= 0, 1, 'first');  
Y_idx = find(path >= 0, 1, 'first');  

curve1_x=Y(:,X_idx);
curve1_z=sgolayfilt(Z(:,X_idx),3,15);
curve2_z=sgolayfilt(Z(Y_idx,:),3,15);

figure(2); hold on
plot(curve1_x,curve1_z)                            
plot(curve1_x,curve2_z)              
grid

figure(3); hold on

dx = mean(diff(curve1_x));                                 
dy1 = rad2deg(gradient(curve1_z,dx));                                
dy2 = rad2deg(gradient(curve2_z,dx));                                

plot(abs(dy1))
plot(abs(dy2))




