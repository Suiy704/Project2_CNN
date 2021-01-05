//
// Created by DELL on 2021/1/2.
//

#include "CNN.h"
#include <cmath>

picture::picture() {
    picture_size = 0;
    channels = 0;
    pixel = nullptr;
    counter = new atomic_int;
    *(counter) = 1;
}

picture::picture(string path) {
    cout << "This picture is: " << path << endl;
    Mat image = imread(path);
    Mat BGR[3];
    resize(image, image, Size(128, 128));
    split(image, BGR);
    picture_size = image.rows;
    channels = image.channels();
    pixel = new float[picture_size * picture_size * channels]();
    int point = 0;
    for (int i = 0; i < BGR[2].total(); i++) {
        pixel[point++] = (float) BGR[2].data[i] / 255.0f;
    }//R
    for (int i = 0; i < BGR[1].total(); i++) {
        pixel[point++] = (float) BGR[1].data[i] / 255.0f;
    }//G
    for (int i = 0; i < BGR[0].total(); i++) {
        pixel[point++] = (float) BGR[0].data[i] / 255.0f;
    }

    counter = new atomic_int;
    *counter = 1;
}

picture::picture(int picture_size, int channels, float *pixel) {
    this->channels = channels;
    this->picture_size = picture_size;
    this->pixel = new float[picture_size * picture_size * channels]();
    for (int i = 0; i < picture_size * picture_size * channels; i++) {
        *(this->pixel + i) = pixel[i];
    }
    this->counter = new atomic_int;
    *counter = 1;
}


Matrix picture::p_trans(int kernel_size, int in_channels, int pad, int stride) {

    int out_size = floor((picture_size - kernel_size + 2 * pad) / stride + 1);
    Matrix p_matrix(out_size * out_size, kernel_size * kernel_size * in_channels);
    int point = 0;
    for (int m = 0; m + kernel_size <= picture_size + pad * 2; m += stride) {
        for (int n = 0; n + kernel_size <= picture_size + pad * 2; n += stride) {
            for (int i = 0; i < channels; ++i) {
                for (int j = 0; j < kernel_size; j++) {
                    for (int k = 0; k < kernel_size; k++) {
                        if (m + j >= pad && m + j < picture_size + pad && n + k >= pad && n + k < picture_size + pad) {
                            //不是第一行 && 不是最后一行 && 不是第一列 && 不是最后一列
                            p_matrix.getData()[point] = *(this->pixel + i * picture_size * picture_size +
                                                          (m + j - pad) * picture_size + n + k - pad);
                            point++;
                        } else {
                            point++;
                        }
                    }
                }
            }
        }
    }
    return p_matrix;
}

Matrix picture::p_trans2(int kernel_size, int in_channels, int pad, int stride) {

    int out_size = floor((picture_size - kernel_size + 2 * pad) / stride + 1);
    Matrix p_matrix(out_size * out_size, kernel_size * kernel_size * in_channels+1);
    int point = 0;
    for (int m = 0; m + kernel_size <= picture_size + pad * 2; m += stride) {
        for (int n = 0; n + kernel_size <= picture_size + pad * 2; n += stride) {
            for (int i = 0; i < channels; ++i) {
                for (int j = 0; j < kernel_size; j++) {
                    for (int k = 0; k < kernel_size; k++) {
                        if (m + j >= pad && m + j < picture_size + pad && n + k >= pad && n + k < picture_size + pad) {
                            //不是第一行 && 不是最后一行 && 不是第一列 && 不是最后一列
                            p_matrix.getData()[point] = *(this->pixel + i * picture_size * picture_size +
                                                          (m + j - pad) * picture_size + n + k - pad);
                            point++;
                        } else {
                            point++;
                        }
                    }
                }
            }
            p_matrix.getData()[point]=1;
            point++;
        }
    }
    return p_matrix;
}

Matrix w_trans(int kernel_size, int in_channels, int out_channels, const float *weights) {
    Matrix w_matrix(kernel_size * kernel_size * in_channels, out_channels);
    int point = 0;
    for (int i = 0; i < kernel_size * kernel_size * in_channels; ++i) {
        for (int j = 0; j < out_channels; ++j) {//这是第j组weights
            w_matrix.getData()[point] = weights[i + j * kernel_size * kernel_size * in_channels];
            point++;
        }
    }
    return w_matrix;
}

Matrix w_trans2(int kernel_size, int in_channels, int out_channels, const float *weights, const float * bias) {
    Matrix w_matrix(kernel_size * kernel_size * in_channels+1, out_channels);
    int point = 0;
    for (int i = 0; i < kernel_size * kernel_size * in_channels; ++i) {
        for (int j = 0; j < out_channels; ++j) {//这是第j组weights
            w_matrix.getData()[point] = weights[i + j * kernel_size * kernel_size * in_channels];
            point++;
        }
        w_matrix.getData()[point]=bias[i];
        point++;
    }
    w_matrix.getData()[point]=bias[kernel_size * kernel_size * in_channels];
    return w_matrix;
}


Matrix b_trans(const float *bias, int out_size, int out_channels) {
    Matrix b_matrix(out_size * out_size, out_channels);
    int point = 0;
    for (int i = 0; i < out_channels; ++i) {
        for (int j = 0; j < out_size * out_size; ++j) {
            b_matrix.getData()[point++] = bias[i];
        }
    }
    return b_matrix;
}

void picture::ConvBNRelu(const conv_param &kernel) {
    Matrix p_matrix = p_trans(kernel.kernel_size, kernel.in_channels, kernel.pad, kernel.stride);
//    Matrix p_matrix = p_trans2(kernel.kernel_size, kernel.in_channels, kernel.pad, kernel.stride);
//    cout << endl << "p_matrix size: " << p_matrix.getRow() << "   " << p_matrix.getColumn() << endl;
    Matrix w_matrix = w_trans(kernel.kernel_size, kernel.in_channels, kernel.out_channels, kernel.p_weight);
//    Matrix w_matrix = w_trans2(kernel.kernel_size, kernel.in_channels, kernel.out_channels, kernel.p_weight,kernel.p_bias);
//    cout << endl << "w_matrix size: " << w_matrix.getRow() << "   " << w_matrix.getColumn() << endl;
    int out_size = floor((this->picture_size - kernel.kernel_size + 2 * kernel.pad) / kernel.stride + 1);
    Matrix b_matrix = b_trans(kernel.p_bias, out_size, kernel.out_channels);
//    cout << endl << "b_matrix size: " << b_matrix.getRow() << "   " << b_matrix.getColumn() << endl;
    Matrix result = (p_matrix * w_matrix) + b_matrix;
//    Matrix result = (p_matrix * w_matrix) ;

//    cout << endl << "r_matrix size: " << result.getRow() << "   " << result.getColumn() << endl;


    for (int i = 0; i < result.getColumn() * result.getRow(); ++i) {
        if (result.getData()[i] < 0) {
            result.getData()[i] = 0.0f;
        }
    }

    this->picture_size = out_size;
    this->channels = kernel.out_channels;

    if (*(counter) == 1) {
        delete counter;
        delete[] this->pixel;
    } else (*counter)--;

    this->pixel = result.getData();
    this->counter = result.getCount();
    (*this->counter)++;

}

void picture::Maxpool2d(int kernel_size, int stride) {
    int out_size = this->picture_size % stride == 0 ? picture_size / stride : this->picture_size / stride + 1;
    float *result = new float[channels * out_size * out_size]();
    if (this->picture_size % kernel_size == 0) {
//        float *result = new float[channels * (picture_size / stride) * (picture_size / stride)];
        int point = 0;
        for (int i = 0; i < channels; ++i) {
            for (int m = 0; m < picture_size; m += stride) {//行
                for (int n = 0; n < picture_size; n += stride) {//列
                    //左上角坐标确定为
                    int index_now = i * picture_size * picture_size + m * picture_size + n;
                    float max = pixel[index_now];
                    for (int j = 0; j < kernel_size; ++j) {//宽，第几行
                        for (int k = 0; k < kernel_size; ++k) {//长，第几列
                            int index_new = index_now + k + j * picture_size;
                            float now = pixel[index_new];
                            if (now > max) {
                                max = now;
                            }
                        }
                    }
                    result[point] = max;
                    point++;
                }
            }
        }
//        if (*(counter) == 1) {
//            delete counter;
//            delete[] this->pixel;
//        } else if (*(counter) == 0) {
//            delete counter;
//        } else (*counter)--;
    } else {
        int point = 0;
        for (int i = 0; i < channels; ++i) {
            for (int m = 0; m < picture_size - 1; m += stride) {//行
                for (int n = 0; n < picture_size - 1; n += stride) {//列
                    //左上角坐标确定为
                    int index_now = i * picture_size * picture_size + m * picture_size + n;
                    float max = pixel[index_now];
                    for (int j = 0; j < kernel_size; ++j) {//宽，第几行
                        for (int k = 0; k < kernel_size; ++k) {//长，第几列
                            int index_new = index_now + k + j * picture_size;
                            float now = pixel[index_new];
                            if (now > max) {
                                max = now;
                            }
                        }
                    }
                    result[point] = max;
                    point++;
                }
                float temp = max(pixel[i * picture_size * picture_size + m * picture_size + picture_size - 1],
                                 pixel[i * picture_size * picture_size + (m + 1) * picture_size + picture_size - 1]);
                result[point] = temp;
                point++;
            }
            for (int j = i * picture_size * picture_size + picture_size * (picture_size - 1);
                 j < i * picture_size * picture_size + picture_size * picture_size - 2; j += stride) {
                float temp = max(pixel[j], pixel[j + 1]);
                result[point] = temp;
                point++;
            }
            int index = i * picture_size * picture_size + (picture_size * picture_size - 1);
            float temp = pixel[index];
            result[point] = temp;
            point++;
        }
    }
    if (*(counter) == 1) {
        delete counter;
        delete[] this->pixel;
    } else (*counter)--;

    this->pixel = result;
    this->picture_size = out_size;
    this->counter = new atomic_int;
    *counter = 1;
}


void picture::FC_Softmax(const fc_param &fc) {
    Matrix weights = Matrix(fc.out_features, fc.in_features, fc.p_weight);
    Matrix p_now(fc.in_features, 1, pixel);
    Matrix bias(fc.out_features, 1, fc.p_bias);

    if ((*counter) == 1) {
        delete[] pixel;
        delete counter;
    } else {
        (*counter)--;
    }

    Matrix p_new = (weights * p_now);
    p_new = p_new + bias;
    this->pixel = p_new.getData();
    this->picture_size = fc.out_features;
    this->counter = p_new.getCount();
    (*counter)++;

    float sum = 0.0f;
    for (int i = 0; i < fc.out_features; ++i) {
        sum += exp(p_new.getData()[i]);
    }
    for (int i = 0; i < fc.out_features; ++i) {
        p_new.getData()[i] = exp(p_new.getData()[i]) / sum;
        cout << p_new.getData()[i] << endl;
    }
    if (p_new.getData()[1]<0.955){
        cout << "This picture is not a portrait." << endl;
    }else{
        cout << "There is about " << p_new.getData()[1]*100 << "% probability that this picture is a portrait." << endl;
    }
}