clear; close all; clc

% load reference solution F0

folder='../build/DemoOutput_Force2D/';

A=readtable([folder 'Contact_pairs_0049.csv']);
B=readtable([folder 'Contact_pairs_0150.csv']);

radius=0.01;
tolerance=0.02*radius;

posZContactF0=A.Z;
index=find(posZContactF0<min(posZContactF0+tolerance));

F0=A.f_z(index);

posZContactFext=B.Z;
index=find(posZContactFext<min(posZContactFext+tolerance));

Fext=B.f_z(index);

plot(Fext./F0)
