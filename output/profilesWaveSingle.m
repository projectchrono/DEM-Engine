clear; close all; clc

% load reference solution F0

casefriction=0;

folder='..//DemoOutput_Force3D_000/';

 folder='./DemoOutput_Force3D_000/';

figure(1); hold on
for i=0:4
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
        
        y=(Fz-F0)/Fext;
        x=-numel(y)/2:numel(y)/2-1;
        plot(x,y,'.-');
        axis([-inf inf -0.1 inf])
        string='';
        for j=1:numel(x)
            string=[string, sprintf('(%2d, %1.3e)', x(j), y(j))];
       
    end
    disp(string)

    end
end
