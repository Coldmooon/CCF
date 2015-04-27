function runDetect(id)
% id = 0: 1 thread running
% id = 1~8: 8 threads running
%
% For pedestrian detection, set nApprox in opts to 0,
% as power law doesn't hold in Caltech dataset.
%
% set addCf in opts to 1 if you want to use CCF+CF model.
% 
% set savePyrd in opts to 1 if you want to cache pyramids
% data, which needs MUCH disk space.
% 

addpath(genpath('./toolbox-master'));

% load detectors
clfDir = ['path_to_CCF_codes' '/model/'];
nDs = 1;
ds = cell(1,nDs);
for i=1:nDs
	d = load([clfDir 'Detector_caltech_depth5.mat']);
    ds{i} = d.detector;
end

% initialize caffe
addpath(['path_to_caffe_codes' '/matlab/caffe']);
%caffe('reset');
model_def = './data/CaffeNets/VGG_ILSVRC_16_layers_conv3.prototxt';
model_file = './data/CaffeNets/VGG_ILSVRC_16_layers.caffemodel';
cnn = struct('model_def',model_def,...
             'model_file',model_file,...
             'device',0,...
             'meanPix',[103.939 116.779 123.68]);
opts = struct('input_size',900,'stride',4,'pad',16,...
        'minDs',72,'nPerOct',6,'nOctUp',1,'nApprox',0,...
        'lambda',0.2666,'imresize',1,'imflip',0,...
        'addCf',0,'savePyrd',0);

imgDir = ['path_to_Caltech_dataset' '/test/'];
imgNms = bbGt('getFiles',{[imgDir 'images']});
if id>0
    imgNms = imgNms((id-1)*503+1:id*503);
end

allBBs = cnnDetect(imgNms, ds, opts, cnn);

save([clfDir '/allBBs_' num2str(id) '.mat'],...
     'allBBs');

end