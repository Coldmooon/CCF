% Index exceeds matrix dimensions.
% 
% Error in cnnPyramid (line 243)
%                     featPyrd{isR(i+k-1),l} =
%                     feat(yy1:yy2,xx1:xx2,:);
% 
% Error in getMean (line 40)
%     P = cnnPyramid(Is(:,:,:,1),opts,cnn);
% 
% The above error is because of the wrong definition of net prototxt

for k=1:1
fprintf('Computing %d0000 model\n', k);
addpath(genpath('../../CCF'));
addpath('/home/coldmoon/ComputerVision/caffe/matlab');
model_def = '/home/coldmoon/ComputerVision/caffe/models/Powerlaw/mstnet_elu_gauss_halfpyramid_pixelwise_scale_decay0_ave_concat_max_deploy.prototxt';
model_file = ['/home/coldmoon/ComputerVision/caffe/computing_lambda.caffemodel'];
% model_def = '../data/CaffeNets/nin_elu_deploy.prototxt';
% model_file = ['../data/CaffeNets/elu_gauss_iter_',num2str(k),'0000.caffemodel'];
% model_file = ['/home/coldmoon/ComputerVision/caffe/',num2str(k),'.caffemodel'];
meanfile = 'cifar10_mean_vertical.mat';
cnn = struct('model_def',model_def,...
             'model_file',model_file,...
             'device',0,...
             'meanPix',[-1.84252e-08, -8.76298e-09, -2.27323e-08], ...
             'meanfile', meanfile);
%              'meanPix',[103.939 116.779 123.68], ...
opts = struct('input_size',360,'stride',4,'pad',16,...
        'minDs',8,'nPerOct',3,'nOctUp',1,'nApprox',0,...
        'lambda',0.2666,'imresize',1,'imflip',0,...
        'addCf',0,'savePyrd',0);

% load some random bbs for power law checking (Is: height x width x nChannels x bbNum)
% load('path_to_example_bbs');
%load('~/codes/faceDetection/sampledWins/view4_Is1Stage0.mat');

% Is = load('cifar10_train_vertical.mat'); Is = Is.cifar10_train;
% Is = dataload('cifar10_train_vertical.mat',opts.stride);
Is = load('/home/coldmoon/ComputerVision/caffe/input_data.mat'); Is = Is.data_for_lambda;
Is = permute(Is, [3,4,2,1]);
% Is = load('/media/coldmoon/ExtremePro960G/Datasets/cifar10-gcn-zca.mat'); Is = Is.data(1:500,:,:,:);
% Is = permute(Is, [3,4,2,1]);

num = size(Is,4);
caffe.reset_all();
caffe.set_mode_cpu();
net = caffe.Net(cnn.model_def, cnn.model_file, 'test');
cnn.net = net;

try
    load('P_face.mat','P');
catch
    P = cnnPyramid(Is(:,:,:,1),opts,cnn);
    disp(['saving P_face_cifar10train_nin_cccp6_elu_iter', num2str(k) ,'0k.mat']);
%     save(['P_face_cifar10train_nin_cccp6_elu_iter', num2str(k) ,'.mat'],'P');
    save(['ppppp.mat'],'P');
end
nScales = P.nScales;

batch_size = 10;

try
    load('face/channel_mean.mat','fs');
catch
    fs=zeros(num,nScales,1);
    % 4 for batch_size. -- by liyang.
    for i=1:batch_size:num
        fprintf('%d/%d ',i,num);
        if(mod(i-1, 40) == 0)
            fprintf('\n');
        end
        P = cnnPyramid(Is(:,:,:,i:i+batch_size-1),opts,cnn);
        for l=1:batch_size % 4 for batch_size
            for j=1:P.nScales
              fs(i+l-1,j)=mean(P.data{j,l}(:));
            end
        end
    end
    disp(['face/channel_mean_cifar10train_nin_cccp6_elu_iter', num2str(k),'0k.mat']);
%     save(['face/channel_mean_cifar10train_nin_cccp6_elu_iter', num2str(k),'.mat'],'fs');
    save(['ccccc.mat'],'fs');
end


end