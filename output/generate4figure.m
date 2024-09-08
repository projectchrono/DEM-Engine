clc; clear
load PlasticSphere.mat


code={'A','B','C','D','E'};
x=[0.00, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90];

string='';
for i = 1:5  
    A=measureAngle(:,:,i);
    y=mean(A,1);
    
    for j=1:numel(x)
        string=[string, sprintf('(%1.2f, %1.3f) [%s]', x(j), y(j),code{i})];
    end
    
end
% fprintf('\n')    % clipboard('copy', data)
clipboard('copy', string)