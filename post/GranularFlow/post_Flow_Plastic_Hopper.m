clear; close all
expBall=[0.00, 0.00
0.23, 3.76
0.53, 8.33
0.79, 13.43
1.06, 19.34
1.34, 25.25
1.56, 30.98
1.96, 36.88
2.32, 42.97
2.60, 48.70
2.86, 54.07
3.09, 58.28
3.29, 62.31
3.56, 67.41
3.81, 72.25
4.12, 78.60
4.46, 83.35
4.68, 87.65
5.02, 91.14
5.28, 92.39
5.64, 93.46
6.49, 94.09
7.15, 94.45];

expCyl=[0.03, 0.36
0.68, 9.13
1.00, 18.26
1.64, 30.71
2.40, 42.97
2.70, 47.90
3.10, 55.60
4.00, 73.05
4.52, 84.69
5.18, 92.57
6.12, 94.09
7.00, 94.18];

expCylWood=[0.02, 0.74
0.82, 10.86
1.48, 21.82
2.07, 30.92
2.67, 40.39
3.25, 49.30
3.88, 59.42
4.32, 66.48
4.80, 74.47
5.32, 81.99
6.10, 88.58
7.01, 88.49];

folder=['../'  '../build/Test_Plastic_Cylinder_Sphere/Hopper/'];
% folder=['../'  '../build/Test_Plastic_Sphere_Cylinder/Hopper/'];
  % folder=['../'  '../build/DemoOutput_Granular_WoodenSphere/Hopper/'];
   % folder=['../'  '../build/DemoOutput_Granular_WoodenCylinder/Hopper/3S_/'];
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


for i=1:1:699
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
        discharge(i)=numel(index)/(totalMass*time(i));
    end
    

end



figure(2); hold on; box on;grid on

% plot(time,discharge,'DisplayName','DEME')
 plot(time,level_z,'--' ,'DisplayName','DEME','Color',[1 0 0])
 plot(expBall(:,1),expBall(:,2)/100,'-sk','DisplayName','Exp.')
% plot(expCyl(:,1),expCyl(:,2)/100,'sr','DisplayName','Exp.')
 plot(expCylWood(:,1),expCylWood(:,2)/100,'-gd','DisplayName','Exp.')

axis([0 8 0 1.00])
xlabel('Time (s)','Interpreter','latex',FontSize=8)
ylabel('m/M [%]','Interpreter','latex',FontSize=8)
leg=legend('Interpreter','latex','Location','best');
ytickformat('%.1f')

set(gcf,'units','centimeters' ,'position',[1,1,8,5])
    
    leg.FontSize=8;
    leg.NumColumns=1;

    set(gca,'fontsize',8);

% f = gcf;
% exportgraphics(f,['P4'  '.png'],'Resolution',600)

[time,level_z]

clc
x=expCylWood(:,1);
y=expCylWood(:,2)/100;
code={'A','B','C','D','E'};
for i = 2   
    
    for j=1:numel(x)
        fprintf('(%1.2f, %1.3f) [%s]', x(j), y(j),code{i});

    end
    % fprintf('\n')    % clipboard('copy', data)
end
