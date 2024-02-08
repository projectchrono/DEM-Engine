clear; close all; clc

% load reference solution F0

casefriction=0;
mass=(pi*0.01^3*4/3*1.0*1000);

M=([1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 2 3 4 5 6 7 8 9 10 11 21 31 41 51 61 71 81 91 101]-1)*mass;

folder='..//DemoOutput_Force3D_000/';

 folder='./DemoOutput_Force3D_000/';
  folder='./DemoOutput_Force3D_3_000_dt1e6/';

figure(1); 


for i=10:26
    m=M(i+1);
    for j=casefriction:casefriction
        localFolder=[folder 'Test_' num2str(i) '/' num2str(j) '/'];
        A=readtable([localFolder 'Contact_pairs_0020.csv']);
        B=readtable([localFolder 'Contact_pairs_0070.csv']);

        radius=0.01;
        tolerance=0.01*radius;

        posZContactF0=A.Z;
        posXContactF0=A.X;
        indexZ=find(posZContactF0<min(posZContactF0+tolerance));
        indexX=find(abs(posXContactF0(indexZ))==min(abs(posXContactF0(indexZ))));

        partPosition=indexZ(indexX);
        contactPair=[A.A(partPosition) A.B(partPosition)];

        Fgravity=A.f_z(partPosition)/m;
        
        indexEnd=find(B.A==contactPair(1) & B.B==contactPair(2));

        Fz=B.f_z(indexEnd)/m;
 
   
        
        
        y=(Fz-Fgravity);
        Fext=m;
        % 
        % x=-numel(y)/2:numel(y)/2-1;
        % x=x+0.5;
        semilogx(m,y,'.-');
        hold on
        % axis([-inf inf -0.1 inf])
    %     string='';
    %     for j=1:numel(x)
    %         string=[string, sprintf('(%1.2f, %1.4e)', x(j), y(j))];
    %     end
    % disp(string)

    end
end
