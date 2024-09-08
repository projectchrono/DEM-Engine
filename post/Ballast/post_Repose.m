clear;close all;clc

folder='../../build/DemoOutput_Repose_Cylinder/';
files=dir(folder);
vec={files.name};
index=contains(vec,'output');
vec=vec(index);

theta=45;
rot = [cos(theta) -sin(theta); sin(theta)  cos(theta)];

file=char(vec(end));
disp(file)

data=readtable([folder file]);

x=data.X;
y=data.Y;

data.Z=data.Z-min(data.Z);
z=data.Z+data.r;
meanValue=mean(z);
dev=std(z);
index=find(z<meanValue+3*dev);

x=x(index);
y=y(index);
z=z(index);



dx=0.020*1;
path=(-0.50:dx:0.50)';

[X,Y] = meshgrid(path,path);
Z=X*0;
range=dx*3;
for i=1:numel(path)      
    for j=1:numel(path) 
    xLocal=X(i,j);    
    ylocal=Y(i,j);
    index=find(abs(x-xLocal)<range & abs(y-ylocal)<range);
    if ~isempty(index)
        temp=abs(max(z(index)));
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

plot(curve1_x, abs(dy1))
plot(curve1_x, abs(dy2))




