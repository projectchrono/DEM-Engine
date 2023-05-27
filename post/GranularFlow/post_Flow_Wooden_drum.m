clear;close all;clc

%%% Experimental data from Sarno et al. 2018
%exp=readtable("../../data/granularFlow/heightSarno2018.dat");


folder=['../../' ...
    'build/DemoOutput_Granular_WoodenCylindersDrum_1/'];
files=dir(folder);


dx=0.0029*1;
pathX=(-0.03:dx:0.03)';
pathY=(-0.08:dx:0.08)';

[X,Y] = meshgrid(pathX,pathY);
Z=X*0;


Y_idx = find(pathX >= 0.00, 1, 'first');

curve1_y=Y(Y_idx,:);
time=(0:0.01:3)';
level_z=zeros(numel(time),1);

points=(-0.04:0.01:0.05);
measureAngle=zeros(numel(50:99),3);

for p=1:3
    range=(p-1)*100+(50:99);
    
    counter=0;

    for k=range
        counter=counter+1;

        file=['DEMdemo_output_' num2str(k,'%04i.csv')];
        disp(file)
        data=readtable([folder file]);

        [X,Y] = meshgrid(pathX,pathY);
        Z=X*0;
        range=dx*3;

        x=data.X;
        y=data.Y;
        z=data.Z+data.r;
        z=z-min(z);


        for i=1:numel(pathY)
            for j=1:numel(pathX)
                xLocal=X(i,j);
                ylocal=Y(i,j);
                index=find(abs(x-xLocal)<range & abs(y-ylocal)<range);
                if ~isempty(index)
                    temp=abs(max(z(index)));
                    Z(i,j)=temp;
                end
            end
        end

        % figure(1)
        % surf(X,Y,Z)
        % drawnow

        X_idx = find(pathX >= 0, 1, 'first');

        curve1_x=Y(:,X_idx);
        curve1_z=sgolayfilt(Z(:,X_idx),3,15);
        results=interp1(curve1_x,curve1_z,points);

        figure(2); hold on
        plot(curve1_x,curve1_z)
        plot(points,results,'.')

        P = polyfit(points,results,1);
        yfit = polyval(P,points);

        plot(points,yfit,'r-.');
        % figure(3); hold on
        angle=abs(atand(P(1)));
        measureAngle(counter,p)=angle;



        % plot(curve1_x, abs(dy1))

    end


end
    figure(3); hold on
    boxplot(measureAngle,'PlotStyle','compact')