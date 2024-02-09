clear; close all; clc

% load reference solution F0

casefriction=0;
mass=(pi*0.1^3*4/3*1.0*1000);

M=[0 .1 .2 .3 .4 .5 .6 .7 .8 0.9 1 2 3 4 5 6 7 8 9 10 10 20 30 40 50 60 70 80 90 100 200]*mass;

folder='..//DemoOutput_Force3D_000/';

 %folder='./DemoOutput_Force3D_4_000_/';
 folder='./DemoOutput_Force3D_3_000_dt1e6/';

 plotOnly=[5 10 20 40 80];
 [~,index] = ismember(plotOnly,M/mass);

figure(1); hold on
for i=index
    % m=M(i+1);
    for j=casefriction:casefriction
        localFolder=[folder 'Test_' num2str(i) '/' num2str(j) '/'];
        A=readtable([localFolder 'Contact_pairs_0026.csv']);
        B=readtable([localFolder 'Contact_pairs_0099.csv']);

        radius=0.01;
        tolerance=0.01*radius;

        posZContactF0=A.Z;
        index=find(posZContactF0<min(posZContactF0+tolerance));

        F0=A.f_z(index);
        Fgravity=sum(F0);
        

        posZContactFext=B.Z;
        index=find(posZContactFext<min(posZContactFext+tolerance));

        Fz=B.f_z(index);
        Fext=sum(Fz)-Fgravity;
        
        
        y=(Fz-F0)/Fext;
        x=-numel(y)/2:numel(y)/2-1;
        x=x+0.5;
        plot(x,y,'.-');
        % axis([-inf inf -0.1 inf])
        string='';
        for j=1:numel(x)
            string=[string, sprintf('(%1.2f, %1.4e)', x(j), y(j))];
        end
    disp(string)

    end
end
