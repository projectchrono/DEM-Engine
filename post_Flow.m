clear;close all;clc

folder='build/DemoOutput_Granular_Flow_1/';
files=dir(folder);


dx=0.0033*1;
pathX=(0.95:dx:1.00)';
pathY=(-0.04:dx:0.04)';

[X,Y] = meshgrid(pathX,pathY);
Z=X*0;


Y_idx = find(pathX >= 0.97, 1, 'first');  

curve1_y=Y(Y_idx,:);
time=(0:0.01:5)';
level_z=zeros(501,1);

for i=50:330
    file=['DEMdemo_output_' num2str(i,'%04i.csv')];
    disp(file)
    data=readtable([folder file]);
    x=data.X;
    y=data.Y;
    z=data.Z+data.r;

    index=find(x>0.96 & x<0.99);
    vec=z(index);
    x=x(index);
    y=y(index);
    z=z(index);
    if numel(index)>1
        % a=sort(vec,'descend');
        % value=mean(vec)+1.96*std(vec);
        % disp(value)
        range=dx*3;
        for j=1:numel(pathX)
            for k=1:numel(pathY)
                xLocal=X(k,j);
                ylocal=Y(k,j);
                index=find(abs(x-xLocal)<range & abs(y-ylocal)<range);
                if ~isempty(index)
                    temp=abs(max(z(index)));
                    Z(k,j)=temp;
                end
            end
        end
    end
    figure(1)
% surf(X,Y,Z)
 curve1_z=sgolayfilt(Z(:,Y_idx),3,15);
% plot(curve1_z)
disp(mean(curve1_z))
level_z(i)=mean(curve1_z);
end


figure(1)

plot(time,level_z)


