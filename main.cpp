//
//  main.cpp
//  Lab3
//
//  Created by Adrian Daniel Bodirlau on 22/11/2022.
//

#include <stdio.h>
#include "CL/openCLutils.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <ctime>
using namespace std;
using namespace cv;

int main(int argc, const char* argv[]) {
    cl_platform_id* platforms;
    CLGetPlatforms(platforms);
    
    cl_device_id* devices;
    CLGetDevices(platforms[0], devices, CL_DEVICE_TYPE_GPU);
    cl_device_id device = devices[0];
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue commands = clCreateCommandQueue(context, device, 0, NULL);
    
    string programSourceStr = readFile((char*)"Lab3.3/source.CL");
    const char* programSource = programSourceStr.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &programSource, NULL, NULL);
    CLBuildProgram(&program, &device);
    
    Mat imgSrc = imread("Lab3.3/input.jpg", 0);
    imgSrc.convertTo(imgSrc, CV_32F);
    
    cl_image_format imgFormat;
    imgFormat.image_channel_order = CL_R;
    imgFormat.image_channel_data_type = CL_FLOAT;
    cl_image_desc imgDesc;
    imgDesc.image_width = imgSrc.cols;
    imgDesc.image_height = imgSrc.rows;
    imgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    
    float scaleFactor = 7;
    int newScale[2]{(int)(imgSrc.cols*scaleFactor),(int)(imgSrc.rows*scaleFactor)};
    cl_image_desc imgScaledDesc;
    imgScaledDesc.image_width = newScale[0];
    imgScaledDesc.image_height = newScale[1];
    imgScaledDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    
    size_t origin[]{0,0,0};
    size_t region[]{(size_t)imgSrc.cols,(size_t)imgSrc.rows,1};
    size_t regionScaled[]{(size_t)newScale[0],(size_t)newScale[1],1};
    cl_mem imgInBuff = clCreateImage(context, CL_MEM_READ_ONLY, &imgFormat, &imgDesc, NULL, NULL);
    clEnqueueWriteImage(commands, imgInBuff, CL_TRUE, origin, region, 0, 0, imgSrc.ptr<float>(), 0, NULL, NULL);
    cl_mem imgOutBuff = clCreateImage(context, CL_MEM_WRITE_ONLY, &imgFormat, &imgScaledDesc, NULL, NULL);
    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, NULL);
    
    cl_kernel kernel = clCreateKernel(program, "scaleBilinear", NULL);
//    cl_kernel kernel = clCreateKernel(program, "scaleNeighbour", NULL);
    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &imgInBuff);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &imgOutBuff);
    clSetKernelArg(kernel, 2, sizeof(cl_sampler), &sampler);
    clSetKernelArg(kernel, 3, sizeof(float), &scaleFactor);
    clFinish(commands);
    // Profiling
    clock_t begin, end;
    
    size_t globalWorkers[]{(size_t)newScale[0],(size_t)newScale[1]};
    begin = clock();
    clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkers, NULL, 0, NULL, NULL);
    clFinish(commands);
    end = clock();
    double timeSpent = double(end-begin)/CLOCKS_PER_SEC;
    printf("Time spent: %f\n", timeSpent);
    
    float* imgOutArray = (float*)malloc(newScale[0]*newScale[1]*sizeof(float));
    clEnqueueReadImage(commands, imgOutBuff, CL_TRUE, origin, regionScaled, 0, 0, imgOutArray, 0, NULL, NULL);

    Mat imgOut{newScale[1],newScale[0],imgSrc.type(),imgOutArray};
    imwrite("Lab3.3/output.bmp", imgOut);
    
    
    
    return 0;
}
