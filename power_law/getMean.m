addpath(genpath('../../CCF'));
addpath('/home/coldmoon/ComputerVision/caffe/matlab');
model_def = '../data/CaffeNets/VGG_ILSVRC_16_layers_conv3.prototxt';
model_file = '../data/CaffeNets/VGG_ILSVRC_16_layers.caffemodel';
meanfile = ''
cnn = struct('model_def',model_def,...
             'model_file',model_file,...
             'device',0,...
             'meanPix',[103.939 116.779 123.68], ...
             'meanfile', meanfile);

opts = struct('input_size',900,'stride',4,'pad',16,...
        'minDs',16,'nPerOct',8,'nOctUp',1,'nApprox',0,...
        'lambda',0.2666,'imresize',1,'imflip',0,...
        'addCf',0,'savePyrd',0);

% load some random bbs for power law checking (Is: height x width x nChannels x bbNum)
% load('path_to_example_bbs');
%load('~/codes/faceDetection/sampledWins/view4_Is1Stage0.mat');

Is = dataload('ImageNetval_1_200.mat',opts.stride);

num = size(Is,4);
caffe.reset_all();
caffe.set_mode_gpu();
net = caffe.Net(cnn.model_def, cnn.model_file, 'test');
cnn.net = net;
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