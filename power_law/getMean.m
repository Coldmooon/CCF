addpath(genpath('../../CCF'));
model_def = '../data/CaffeNets/VGG_ILSVRC_16_layers_conv3.prototxt';
model_file = '../data/CaffeNets/VGG_ILSVRC_16_layers.caffemodel';
cnn = struct('model_def',model_def,...
             'model_file',model_file,...
             'device',0,...
             'meanPix',[103.939 116.779 123.68]);

opts = struct('input_size',900,'stride',4,'pad',16,...
        'minDs',16,'nPerOct',8,'nOctUp',1,'nApprox',0,...
        'lambda',0.2666,'imresize',1,'imflip',0,...
        'addCf',0,'savePyrd',0);

% load some random bbs for power law checking (Is: height x width x nChannels x bbNum)
% load('path_to_example_bbs');
%load('~/codes/faceDetection/sampledWins/view4_Is1Stage0.mat');
Is_t = load('~/INRIA-test.mat'); Is_t = Is_t.ans;
nImage = size(Is_t,2);
image_size = size(Is_t{1,1});
Is = zeros(image_size(1), image_size(2), image_size(3), nImage);
for i=1:nImage
    Is(:,:,:,i) = Is_t{1,i};
end

num = size(Is,4);
try
    load('P_face.mat','P');
catch
    P = cnnPyramid(Is(:,:,:,1),opts,cnn);
    save('P_face.mat','P');
end
nScales = P.nScales;

try
    load('face/channel_mean.mat','fs');
catch
    fs=zeros(num,nScales,1);
    for i=1:num
        fprintf('%d/%d\n',i,num);
        P = cnnPyramid(Is(:,:,:,i),opts,cnn);
        for j=1:P.nScales
          fs(i,j,1)=mean(P.data{j}(:));
        end
    end
    save('face/channel_mean.mat','fs');
end