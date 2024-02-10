clear; close all; clc

% load reference solution F0

casefriction=0;
mass=(pi*0.01^3*4/3*1.0*1000);

M=[0 .1 .2 .3 .4 .5 .6 .7 .8 0.9 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 400 800 1600]*mass;

folder='..//DemoOutput_Force3D_000/';

 folder='./DemoOutput_Force3D_000/';
   folder='./DemoOutput_Force3D_3_000_dt1e6/';
   folder='./DemoOutput_Force3D_4_0.20/';

figure(1); 
string='';
out=zeros(numel(M),1);
for i=1:numel(M)-3
    m=M(i);
    f=m*9.81;
    for j=casefriction:casefriction
        localFolder=[folder 'Test_' num2str(i) '/' num2str(j) '/'];
        localFolder=[folder 'Test_' num2str(i) '/'];
        A=readtable([localFolder 'Contact_pairs_0025.csv']);
        B=readtable([localFolder 'Contact_pairs_0099.csv']);

        radius=0.01;
        tolerance=0.01*radius;

        posZContactF0=A.Z;
        posXContactF0=A.X;
        indexZ=find(posZContactF0<min(posZContactF0+tolerance));
        indexX=find(abs(posXContactF0(indexZ))==min(abs(posXContactF0(indexZ))));

        partPosition=indexZ(indexX);
        contactPair=[A.A(partPosition) A.B(partPosition)];

        Fgravity=A.f_z(partPosition)/f;
        
        indexEnd=find(B.A==contactPair(1) & B.B==contactPair(2));

        Fz=B.f_z(indexEnd)/f;
 
   
        
        
        y=(Fz-Fgravity);
        Fext=m;
        % 
        % x=-numel(y)/2:numel(y)/2-1;
        % x=x+0.5;
        semilogx(m/mass,y,'.-');
        hold on
        axis([-inf inf -0.1 0.3])
        
        string=[string, sprintf('(%1.2f, %1.4e)', m/mass, y)];
        
    end
end 
disp(string)
grid on
