clear;close all;clc

%%% Experimental data from Sarno et al. 2018
exp=readtable("../../data/granularFlow/heightSarno2018.dat");


folder=['../' ...
    '../build/DemoOutput_Granular/Hopper_shute/1/'];
files=dir(folder);


dx=0.0033*1/2;
pathZ=(0.00:dx:0.06)';
pathY=(-0.04:dx:0.04)';

[X,Y] = meshgrid(pathY,pathZ);
Z=X*0;


% Y_idx = find(pathX >= 0.95, 1, 'first');  

% curve1_y=Y(Y_idx,:);
dt=0.01;
time=(0:dt:5)';
level_z=zeros(numel(time),1);

for i=200
    file=['DEMdemo_output_' num2str(i,'%04i.csv')];
    disp(file)
    data=readtable([folder file]);
    x=data.X;
    y=data.Y;
    z=data.Z;

    index=find(x>0.94 & x<0.96);
    
    x=x(index);
    y=y(index);
    z=z(index);

    %%open next file for velocity
    file=['DEMdemo_output_' num2str(i+1,'%04i.csv')];
    disp(file)
    data=readtable([folder file]);
    x1=data.X(index);
    y1=data.Y(index);
    z1=data.Z(index);

    vx=(x1-x)./dt;
    vy=(y1-y)./dt;
    vz=(z1-z)./dt;

    tempMin=0.0;
    if numel(index)>1
        % a=sort(vec,'descend');
        % value=mean(vec)+1.96*std(vec);
        % disp(value)
        range=dx*3;
        for j=1:numel(pathY)
            for k=1:numel(pathZ)
                yLocal=X(k,j);
                zlocal=Y(k,j);
                index=find(abs(y-yLocal)<range & abs(z-zlocal)<range);
                
                if ~isempty(index)
                    temp=(mean(vx(index)));
                    tempMin=abs(min(z(index)))-1*0.0033;
                    Z(k,j)=temp;
                end
            end
        end
    end
    
 surf(X,Y,Z)
 % curve1_z=sgolayfilt(Z(:,Y_idx),3,15);
  % plot(curve1_z)
  drawnow
 % disp([num2str(mean(curve1_z),'%1.3f') ' ' num2str(tempMin,'%1.3f')])
 % if numel(curve1_z)>0
 %     level_z(i)=mean(curve1_z);
 % end
end


figure(2); hold on

plot(time,level_z,'DisplayName','DEME')
plot(exp.Var1,exp.Var2,'DisplayName','Exp.')

axis([0 14 0 0.06])
