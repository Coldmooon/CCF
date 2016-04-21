function [ Is ] = dataload( filepath, stride )
%DATALOADSADF Summary of this function goes here
%   Detailed explanation goes here

Is = load(filepath); 
Is = Is.data; 

if(iscell(Is))
    nImage = size(Is,2);
    % Delete gray images.
    % -------------------
    delete = zeros(1,nImage);
    for n=1:nImage
        if size(Is{1,n},3) < 3;
            delete(n)=1;
        end
    end
    Is(find(delete)) = [];
    nImage = size(Is,2);
    % ---------------------
    
    % Crop images to the same size.
    ds=[inf inf]; 
    for i=1:nImage, 
        ds=min(ds,[size(Is{i},1) size(Is{i},2)]); 
    end
    % ds=round(ds/pChns.shrink)*pChns.shrink;
    ds=floor(ds/stride)*stride; % for INRIA train pos. --by liyang 
    for i=1:nImage, 
        Is{i}=Is{i}(1:ds(1),1:ds(2),:); 
    end
    % ----------------------------

    % resize images
    % for i=1:nImage, 
    %     Is_t{i} = resample(Is_t{i}, 256, 256); 
    % end
    % --------------

    image_size = size(Is{1,1});
    Is_t = zeros(image_size(1), image_size(2), image_size(3), nImage);
    for i=1:nImage
        fprintf('%d\n', i);
        Is_t(:,:,:,i) = Is{1,i};
    end
    Is = Is_t;
else
    nImage = size(Is, 1);
    h_and_w = sqrt(size(Is, 2)/3);
    Is = permute(reshape(Is, [nImage,h_and_w,h_and_w,3]), [2 3 4 1]); 
    selected = randperm(10000, 200);
    Is = Is(:,:,:,selected);
end


end

