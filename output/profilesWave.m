clear; close all; clc

% load reference solution F0

folder='../build/DemoOutput_Force2D/';

A=readtable([folder 'Contact_pairs_0030.csv']);
B=readtable([folder 'Contact_pairs_0060.csv']);

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
