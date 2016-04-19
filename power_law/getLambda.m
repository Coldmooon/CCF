load('face/channel_mean.mat');

fs = fs(randperm(195,100),:);

load P_face.mat;
ls = chnsScaling( P.scales', fs, 1 );
ls = round(ls*10^5)/10^5;
lambda = ls
%saveas(gcf,'face/powerlaw_face','fig');