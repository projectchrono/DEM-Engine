clear;close all;clc

%%% Experimental data from Sarno et al. 2018
%exp=readtable("../../data/granularFlow/heightSarno2018.dat");


folder=['../' ...
    '../build/DemoOutput_Granular_NonSherical_1/'];
files=dir(folder);

d=0.0060;

g=9.81;
rho=1592;
w=0.055;
A0=0.04*w;
P0=2*0.04+2*w;
Dh=4*A0/P0;
k=1.33;

N=2.344/(4/3*pi*(d/2)^3*rho);

fowler=0.221*A0*rho*(2*g*Dh)^0.50*(Dh/(k*d))^0.185;






time=(0:0.01:7)';
level_z=zeros(numel(time),1);
discharge=zeros(numel(time),1);


for i=1:1:350
    file=['DEMdemo_output_' num2str(i,'%04i.csv')];
    disp(file)
    data=readtable([folder file]);
    x=data.X;
    y=data.Y;
    z=data.Z+data.r;
    totalMass=numel(x);
    index=find(z<0.00 );
    vec=z(index);
    x=x(index);
    y=y(index);
    z=z(index);
    
    tempMin=0.0;
    if numel(index)>1
        % a=sort(vec,'descend');
        % value=mean(vec)+1.96*std(vec);
        % disp(value)
        level_z(i)=numel(index)/totalMass;
        discharge(i)=numel(index)/(totalMass*i*0.01);
    end
    

end


figure(2); hold on

plot(time,discharge,'DisplayName','DEME')
 % plot(time,level_z,'DisplayName','DEME')
 % plot(exp.Var1,exp.Var2,'DisplayName','Exp.')

axis([0 8 0 0.60])
