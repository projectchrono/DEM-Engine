clear;close all;clc
figure(); hold on
A=readtable("../../output/data_R0.001200_Int0.700000.csv");

strain=abs(A.Var1(1:end))*100;
strain=strain-strain(1);
stress=A.Var2(1:end);
stress=stress-stress(1);

plot(strain,stress)



A=readtable("../../output/data_R0.001200_Int0.900000.csv");

strain=abs(A.Var1(1:50))*100;
strain=strain-strain(1);
stress=A.Var2(1:50);
stress=stress-stress(1);

 plot(strain,stress)
x=strain;
y=stress/1e6;

A=readtable("../../output/data_R0.001000_Int0.900000.csv");

strain=abs(A.Var1(1:end))*100;
strain=strain-strain(1);
stress=A.Var2(1:end);
stress=stress-stress(1);

plot(strain,stress)



A=readtable("../../output/Default Dataset.csv");
strain=A.Var1;
stress=A.Var2;
plot(strain,stress*1e6,'k')

% x=strain;
% y=stress;
string='';

    for j=1:numel(x)
        string=[string sprintf('(%1.4f, %1.3f)', x(j), y(j))];

    end
     fprintf('\n')
     % clipboard('copy', data)
     disp(string)