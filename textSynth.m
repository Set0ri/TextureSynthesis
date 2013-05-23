function [Image] = textSynth(sampleTexture, win_size, newRows, newCols)
    
    %initialize constants
    sigma = win_size/6.4;
    err_threshold = 0.1;
    max_err_threshold = 0.3;
    gaussMask = fspecial('gaussian',win_size,sigma);
    
    %readin texture
    sample = im2double(imread(sampleTexture));
    [rows, cols, channels] = size(sample);

    
    if channels == 3
        red_sample = im2col(sample(:, :, 1), [win_size win_size], 'sliding'); 
        green_sample = im2col(sample(:, :, 2), [win_size win_size], 'sliding'); 
        blue_sample = im2col(sample(:, :, 3), [win_size win_size], 'sliding');

        grey_sample = [];
    else
        grey_sample = im2col(sample(:,:), [win_size win_size]);
        red_sample = []; 
        green_sample = []; 
        blue_sample = [];
    end
    
    texture = zeros(rows+newRows,cols+newCols,channels);
    texture(1:rows,1:cols,:) = sample;
    texture_rows = size(texture,1);
    texture_cols = size(texture,2);
    
    explored = false(texture_rows,texture_cols);
    explored(1:rows,1:cols) = true([rows cols]);
    
    nPixels = texture_rows*texture_cols;
    nPixels_filled = rows*cols;
    
    while nPixels_filled < nPixels
        progress = false;
        
        [pixelRows, pixelCols] = getUnfilledNeighbors(explored,win_size);
        
        for i = [pixelRows pixelCols]';
            
            [template, validMask] = getNeighWindow(win_size,texture,explored,i(1),i(2));
            [rowMatch, colMatch, bestMatch_err] = findMatch(gaussMask,win_size,sample, template, validMask, err_threshold, red_sample, green_sample, blue_sample, grey_sample);
            if (bestMatch_err < max_err_threshold)
                texture(i(1),i(2),:) = sample(round(rowMatch), round(colMatch)  ,:);
                explored(i(1), i(2)) = true;
                nPixels_filled=nPixels_filled+1;
                progress = true;
            end
        end
        
        imshow(texture);
        drawnow
        
        if (progress==false)
            max_err_threshold = max_err_threshold*1.1;
        end
    end
    
    imshow(texture);
    imwrite(texture,strcat(int2str(win_size),'_',sampleTexture),'jpg');
    Image = texture;
end

function [pixelRows, pixelCols] = getUnfilledNeighbors(explored,win_size)
    
    expanded_explored = bwmorph(explored,'dilate');
    unfilledPixels = expanded_explored - explored;
    [pixelRows, pixelCols] = find(unfilledPixels);
    
    randIndex = randperm(length(pixelRows));
    pixelRows = pixelRows(randIndex);
    pixelCols = pixelCols(randIndex);
    
    neighSums = colfilt(explored,[win_size win_size],'sliding',@sum);
    
   
    linearIndex = sub2ind(size(neighSums),pixelRows,pixelCols);
    [~, index] = sort(neighSums(linearIndex),'descend');
    sorted = linearIndex(index);
    [pixelRows, pixelCols] = ind2sub(size(explored),sorted);
end

function [rowMatch, colMatch, err] = findMatch(gaussMask,win_size,sample, template, validMask, err_threshold, red_sample, green_sample, blue_sample, grey_sample)
    
    weightTotal = sum(sum(gaussMask(validMask)));
    
    mask = validMask.*gaussMask/weightTotal;
    mask = mask(:)';
    
    if (size(template,3)==3)
        [nPixels_Window, numNeighborhoods] = size(red_sample);
        
        red_vals = template(:,:,1);
        red_vals = red_vals(:);

        green_vals = template(:,:,2);
        green_vals = green_vals(:);

        blue_vals = template(:,:,3);
        blue_vals = blue_vals(:);

        red_vals = repmat(red_vals, [1 numNeighborhoods]); 
        green_vals = repmat(green_vals, [1 numNeighborhoods]); 
        blue_vals = repmat(blue_vals, [1 numNeighborhoods]); 
   
        red_dist =  mask * (red_vals - red_sample).^2; 
        green_dist = mask * (green_vals - green_sample).^2; 
        blue_dist = mask * (blue_vals - blue_sample).^2; 

        SSD = (red_dist + green_dist + blue_dist); 
    else
        
        [nPixels_Window, numNeighborhoods] = size(grey_sample);

        grey_vals = template(:,:);
        grey_vals = grey_vals(:);


        grey_vals = repmat(grey_vals, [1 numNeighborhoods]);

        grey_dist = mask * (grey_vals - grey_sample).^2;

        SSD = grey_dist;
    end

    pixelMatches = find(SSD <= min(SSD) * (1+err_threshold));
    pixelMatch = pixelMatches(ceil(rand*length(pixelMatches)));
    err = SSD(pixelMatch);

    [rowMatch, colMatch] = ind2sub(size(sample) - win_size + 1, pixelMatch);

    half_win = (win_size-1)/2;
    rowMatch = rowMatch + half_win;
    colMatch = colMatch + half_win;
end

function [template, validMask] = getNeighWindow(win_size,texture, explored, pixelRow, pixelCol)


    halfWin = floor((win_size - 1) / 2);

    if mod(win_size,2)
        % window size is odd
        winRow_range = pixelRow - halfWin : pixelRow + halfWin;
        winCol_range = pixelCol - halfWin : pixelCol + halfWin;
    else
        % window size is even
        rowMove = round(rand);
        winRow_range = pixelRow - (halfWin + rowMove) : pixelRow + (halfWin + ~rowMove);

        colMove = round(rand);
        winCol_range = pixelCol - (halfWin + colMove) : pixelCol + (halfWin + ~colMove); 
    end

    row_outBounds = winRow_range < 1 | winRow_range > size(texture,1);
    col_outBounds = winCol_range < 1 | winCol_range > size(texture,2);

    if sum(row_outBounds) + sum(col_outBounds) > 0
        row_in_bounds = winRow_range(~row_outBounds);
        col_in_bounds = winCol_range(~col_outBounds);

        if size(texture,3) == 3
            template = zeros(win_size, win_size, 3);
            template(~row_outBounds, ~col_outBounds, :) = texture(row_in_bounds, col_in_bounds, :);
        else 
            template = zeros(win_size, win_size);
            template(~row_outBounds, ~col_outBounds) = texture(row_in_bounds, col_in_bounds);
        end

        validMask = false([win_size win_size]);
        validMask(~row_outBounds, ~col_outBounds) = explored(row_in_bounds, col_in_bounds);
    else
        template = texture(winRow_range, winCol_range, :);
        validMask = explored(winRow_range, winCol_range);
    end

end

        
                
                
        
        
        
        
        
        
    