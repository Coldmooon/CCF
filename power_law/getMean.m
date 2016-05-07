for k=1:10
fprintf('Computing %d0000 model\n', k);
addpath(genpath('../../CCF'));
addpath('/home/coldmoon/ComputerVision/caffe/matlab');
model_def = '../data/CaffeNets/nin_elu_deploy.prototxt';
model_file = ['../data/CaffeNets/elu_gauss_iter_',num2str(k),'0000.caffemodel'];
meanfile = 'cifar10_mean_vertical.mat';
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

% Is = load('cifar10_train_vertical.mat'); Is = Is.cifar10_train;
Is = dataload('cifar10_train_vertical.mat',opts.stride);

num = size(Is,4);
caffe.reset_all();
caffe.set_mode_gpu();
net = caffe.Net(cnn.model_def, cnn.model_file, 'test');
cnn.net = net;

try
    load('P_face.mat','P');
catch
    P = cnnPyramid(Is(:,:,:,1),opts,cnn);
    disp(['saving P_face_cifar10train_nin_cccp6_elu_iter', num2str(k) ,'0k.mat']);
    save(['P_face_cifar10train_nin_cccp6_elu_iter', num2str(k) ,'0k.mat'],'P');
end
nScales = P.nScales;

try
    load('face/channel_mean.mat','fs');
catch
    fs=zeros(num,nScales,1);
    % 4 for batch_size. -- by liyang.
    for i=1:4:num
        fprintf('%d/%d ',i,num);
        if(mod(i-1, 40) == 0)
            fprintf('\n');
        end
        P = cnnPyramid(Is(:,:,:,i:i+3),opts,cnn);
        for l=1:4 % 4 for batch_size
            for j=1:P.nScales
              fs(i+l-1,j)=mean(P.data{j,l}(:));
            end
        end
    end
    disp(['face/channel_mean_cifar10train_nin_cccp6_elu_iter', num2str(k),'0k.mat']);
    save(['face/channel_mean_cifar10train_nin_cccp6_elu_iter', num2str(k),'0k.mat'],'fs');
end


end