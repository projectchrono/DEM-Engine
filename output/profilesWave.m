clear; close all; clc

% load reference solution F0

casefriction=3;

folder='../build/DemoOutput_Force2D/';

folder='./DemoOutput_Force2D/';

figure(1); hold on
for i=0:7
    for j=casefriction:casefriction
        localFolder=[folder 'Test_' num2str(i) '/' num2str(j) '/'];
        A=readtable([localFolder 'Contact_pairs_0029.csv']);
        B=readtable([localFolder 'Contact_pairs_0060.csv']);

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


        plot((Fz-F0)/Fext);
        axis([-inf inf -0.1 inf])
    end
end
