//
// Created by DELL on 2021/1/2.
//

#ifndef CNN_CNN_H
#define CNN_CNN_H

#include "Matrix.h"
#include <atomic>
#include "face_binary_cls.cpp"
#include <opencv2/opencv.hpp>

using namespace cv;

class picture {
private:
    int picture_size;//128
    int channels;//initial 3
    float *pixel;//像素
    atomic_int *counter=0;
public:
    picture();
    picture(string path);
    ~picture(){}
    void ConvBNRelu(const conv_param& kernel);
    Matrix p_trans(int kernel_size,int in_channels, int pad, int stride);//把原始图片进行改变，变成按照kernel排列的新矩阵
    Matrix p_trans2(int kernel_size,int in_channels, int pad, int stride);//把原始图片进行改变，变成按照kernel排列的新矩阵
    void Maxpool2d(int kernel_size,int pad);
    float * getPixel() const{return pixel;}
    void FC_Softmax(const fc_param &fc);
    picture(int picture_size,int channels,float *pixel);

};

Matrix w_trans(int kernel_size,int in_channels,int out_channels,const float * weights);
Matrix w_trans2(int kernel_size,int in_channels,int out_channels,const float * weights, const float * bias);
Matrix b_trans(const float* bias, int out_size,int out_channels);


#endif //CNN_CNN_H
