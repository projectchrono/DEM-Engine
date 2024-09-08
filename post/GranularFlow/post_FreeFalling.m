clear; close all


 folder=['../'  '../build/DemoOutput_FreeFalling/'];
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
CoGz=zeros(numel(time),1);
discharge=zeros(numel(time),1);


for i=0:1:200
    file=['DEMdemo_output_' num2str(i,'%04i.csv')];
    disp(file)
    data=readtable([folder file]);
    x=data.X;
    y=data.Y;
    z=data.Z;
    CoGz(i+1)=sum(z) / size(z, 1);
    totalMass=numel(x);
        

end

position = CoGz(1) - 0.5 * 9.81 * time.^2;

figure(2); hold on; box on;grid on

% plot(time,discharge,'DisplayName','DEME')
 plot(time,CoGz,'--' ,'DisplayName','DEME','Color',[1 0 0])
plot(time,position)

axis([0 3 -3.5 0])
xlabel('Time (s)','Interpreter','latex',FontSize=8)
ylabel('m/M [%]','Interpreter','latex',FontSize=8)
leg=legend('Interpreter','latex','Location','best');
ytickformat('%.1f')

set(gcf,'units','centimeters' ,'position',[1,1,8,5])
    
    leg.FontSize=8;
    leg.NumColumns=1;

    set(gca,'fontsize',8);

f = gcf;
exportgraphics(f,['P4'  '.png'],'Resolution',600)

