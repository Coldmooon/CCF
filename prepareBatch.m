%% ------------------------------------------------------ %%
function data = prepareBatch(batches, meanPixel)
	% mean-subtraction, permute...
%% ------------------------------------------------------ %%
nBatch = length(batches);
data = zeros([size(batches{1}),size(batches,2)]);
for i=1:nBatch
	im = batches{i}(:,:,[3 2 1]); % convert from RGB to BGR -- by liyang.
	im = permute(im,[2 1 3]); % permute width and height -- by liyang.
    for c=1:3
        im(:,:,c) = im(:,:,c)-meanPixel(c);
    end
	data(:,:,:,i) = im;
end

end