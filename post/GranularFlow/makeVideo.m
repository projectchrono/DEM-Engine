clear; close all; fclose all;

videoOut = VideoWriter('plasticSphere','Motion JPEG AVI'); %create the video object
videoOut.Quality = 100;
videoOut.FrameRate=50;
open(videoOut);
name='/home/zeus/Videos/sphere/plasticSphere.';
dt=0.01;
border=4;

for ii=0:699
   disp(ii)
  %I = im2double(imread(['' name num2str(ii,'%i') '.png'])); 
  I = (imread(['' name num2str(ii,'%04i') '.png'])); 
%   I=imresize(I,[1920 1080]);
%   I=I(1:end,1:end,:);
    position =  [10 80];
      value = ['time= ' num2str(ii*dt,'%1.2f') ' s'];
%      value='setfrdrawmode=true';
%     I = insertText(I,position,value,'AnchorPoint','LeftBottom','Font',...
     %    'Times','BoxColor','w','FontSize',40 );
%     value='setfrdrawmode=false';
%     position =  [10 650];
%     I = insertText(I,position,value,'AnchorPoint','LeftBottom','Font','LucidaBrightRegular','BoxColor','w','FontSize',20 );
  writeVideo(videoOut,I);
  
end
close(videoOut); 
