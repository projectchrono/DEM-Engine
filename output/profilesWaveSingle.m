clear; close all; clc

% load reference solution F0

casefriction=1;
mass=(pi*0.01^3*4/3*1.0*1000);
% mass=(pi*0.01^2*1000*0.05);

M=[0 .1 .2 .3 .4 .5 .6 .7 .8 0.9 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 400 800 1600]*mass;

folder='..//DemoOutput_Force3D_000/';

 folder='./DemoOutput_Force3D_4_000_/';
%   folder='./DemoOutput_Force3D_3_000_dt1e6/';
 folder='./DemoOutput_Force3D_4_0.0001/';
     folder='./DemoOutput_Force3D_4_0.20/';


 plotOnly=[0.5 1 2 4 6];
 plotOnly=[5 10 20 40  100 200];
 [~,indexCase] = ismember(plotOnly,M/mass);

figure(1); hold on
for i=indexCase
    m=M(i+1);
    f=m*9.81;
    for j=casefriction:casefriction
        localFolder=[folder 'Test_' num2str(i) '/'];
        A=readtable([localFolder 'Contact_pairs_0026.csv']);
        B=readtable([localFolder 'Contact_pairs_0099.csv']);

        radius=0.01;
        tolerance=0.02*radius;

        posZContactF0=A.Z;
        index=find(posZContactF0<min(posZContactF0+tolerance));

        F0=A.f_z(index);
        Fgravity=sum(F0);
        xpos=A.X(index);
        [~,b]=sort(xpos);
        F0=F0(b);

        pointA=A.A(index);
        pointB=A.B(index);

        posZContactFext=B.Z;
        index=find(posZContactFext<min(posZContactFext+tolerance));
        indexFz=zeros(numel(pointA),1);

        for k=1:numel(pointA)
            
            indexTempA=find(B.A==pointA(k));
            indexTempB=find(B.B(indexTempA)==pointB(k));

            indexFz(k)=indexTempA(indexTempB(1));
        end

        Fz=B.f_z(indexFz);

        xpos=B.X(indexFz);
        [a,b]=sort(xpos);
        Fz=Fz(b);

        Fext=sum(Fz)-Fgravity;

        

        % Fext=f;
        
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
