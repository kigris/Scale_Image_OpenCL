__kernel void scaleNeighbour(__read_only image2d_t inputImg, __write_only image2d_t outputImg, sampler_t sampler, float scaleFactor){
    // Get thread ids
    int2 coords = {get_global_id(0), get_global_id(1)};

    // Get the coordinates of the original pixel dividing the current pixel coordinates by the scaleFactor
    int2 originalCoords = {(int)(coords.x/scaleFactor),(int)(coords.y/scaleFactor)};
    
    // Get the pixel from the original image
    float4 out = read_imagef(inputImg, sampler, originalCoords);
    // Write the pixel to the new image
    write_imagef(outputImg, coords, out);
}

__kernel void scaleBilinear(__read_only image2d_t inputImg, __write_only image2d_t outputImg, sampler_t sampler, float scaleFactor){
    // Get thread ids
    int2 coords = {get_global_id(0), get_global_id(1)};

    // Get the coordinates of the original pixel dividing the current pixel coordinates by the scaleFactor
    int2 originalCoords = {(int)(coords.x/scaleFactor),(int)(coords.y/scaleFactor)};
    bool isOrigPixel = ( remainder(coords.x,scaleFactor) == 0 ) && ( remainder(coords.y,scaleFactor) == 0 );
    
    // If this is a scaled pixel from the original image
    if(isOrigPixel){
        // Get the pixel from the original image
        float4 out = read_imagef(inputImg, sampler, originalCoords);
        // Write the pixel to the new image
        write_imagef(outputImg, coords, out);
    } else {
    // Otherwise we need to interpolate it
        float2 sum; // Store the two interpolations of the columns pixels here
        // We need to interpolate between the 4 neighbour pixels in the x position
        for(int i=0;i<2;i++){ // Iterate over each row
            for(int j=0;j<2;j++){ // Iterate over each column
                // Coordinates of our search point
                int2 coordsToRead={originalCoords.x+j, originalCoords.y+i};
                // We read its value from original image
                float4 currentPixel = read_imagef(inputImg, sampler, coordsToRead);
                // We get the original pixel X position in the scaled space
                int coordsToSubstract = (originalCoords.x+j)*scaleFactor;
                // We measure the distance between this current pixel and the original pixel in the scaled space in the X position only
                int distance = {abs(coords.x-coordsToSubstract)};
                // We get the multiply value by substracting from 1 the distance calculated before and the scale factor, so if one pixel is 1 pixel apart in the X axis from the original pixel it will be multiplied by 1-(1/3)=2/3 assuming the multiplying factor is 3
                float multiplier = 1.0f-(float)((float)distance/(float)scaleFactor);
                // We sum this to the variable that we are using to store this values
                sum[i]+=multiplier*currentPixel.x;
            }
        }
        float yResult={0.0f}; // Store the y result here
        // We need to interpolate between the 2 values we got from the X axis
        for(int i=0;i<2;i++){
            // We get the original pixel Y position in the scaled space
            int coordsToSubstract = {(originalCoords.y+i)*scaleFactor};
            // We measure the distance between this current pixel and the original pixel in the scaled space in the Y position only
            int distance = {abs(coords.y-coordsToSubstract)};
            // We get the multiply factor as before
            float multiplier = 1.0f-(float)((float)distance/(float)scaleFactor);
            // We sum the result
            yResult+=multiplier*sum[i];
        }
        // We save the final result in grayscale image
        float4 result;
        result.x=yResult;
        write_imagef(outputImg, coords, result);
    }
}
