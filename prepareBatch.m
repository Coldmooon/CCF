%% ------------------------------------------------------ %%
function data = prepareBatch(batches, meanPixel, meanImage)
	% mean-subtraction, permute...
%% ------------------------------------------------------ %%
nBatch = length(batches);
data = cell(nBatch,1);
for i=1:nBatch
	im = batches{i}(:,:,[3 2 1]); % RGB to BGR. -- by liyang.
	im = permute(im,[2 1 3]);
    if(isempty(meanImage))
        for c=1:3
            im(:,:,c) = im(:,:,c)-meanPixel(c);
        end
    else
        im = double(im)-meanImage;
    end
	data{i} = im;
end

end