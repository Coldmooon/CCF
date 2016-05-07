%% ------------------------------------------------------ %%
function patch = bbCrop2(inputIm, bbs, imPyrd)
%% ------------------------------------------------------ %%
patch = cell(1, size(imPyrd,2));
patch(:) = {inputIm};
for j = 1:size(imPyrd, 2)
    
    for i=1:size(bbs,1)
        patch{j}(bbs(i,3):bbs(i,4),bbs(i,1):bbs(i,2),:) = imPyrd{i,j};
    end

end


end