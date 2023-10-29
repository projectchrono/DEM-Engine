clear;close all;
figure(); hold on
A=readtable("../../output/data_R0.001200_Int0.900000.csv");

strain=abs(A.Var1(5:end))*100;
strain=strain-strain(1);
stress=A.Var2(5:end);
stress=stress-stress(1);

plot(strain,stress)

A=readtable("../../output/data_R0.001200_Int1.100000.csv");

strain=abs(A.Var1(1:end))*100;
strain=strain-strain(1);
stress=A.Var2(1:end);
stress=stress-stress(1);

plot(strain,stress)

A=readtable("../../output/data_R0.001000_Int1.200000.csv");

strain=abs(A.Var1(5:end))*100;
strain=strain-strain(1);
stress=A.Var2(5:end);
stress=stress-stress(1);

plot(strain,stress)