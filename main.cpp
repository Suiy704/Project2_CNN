#include <iostream>
#include <opencv2/opencv.hpp>
#include "CNN.cpp"

using namespace std;


int main() {
    picture bg_CNN = picture("../bg.jpg");
    auto start1 = std::chrono::steady_clock::now();
    bg_CNN.ConvBNRelu(conv_params[0]);
    bg_CNN.Maxpool2d(2, 2);
    bg_CNN.ConvBNRelu(conv_params[1]);
    bg_CNN.Maxpool2d(2, 2);
    bg_CNN.ConvBNRelu(conv_params[2]);
    bg_CNN.FC_Softmax(fc_params[0]);
    auto end1 = std::chrono::steady_clock::now();
    cout
            << "This CNN calculations took "
            << chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() << "µs ≈ "
            << chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << "ms ≈ "
            << chrono::duration_cast<std::chrono::seconds>(end1 - start1).count() << "s.\n\n";

    picture f_CNN = picture("../face.jpg");
    auto start2 = std::chrono::steady_clock::now();
    f_CNN.ConvBNRelu(conv_params[0]);
    f_CNN.Maxpool2d(2, 2);
    f_CNN.ConvBNRelu(conv_params[1]);
    f_CNN.Maxpool2d(2, 2);
    f_CNN.ConvBNRelu(conv_params[2]);
    f_CNN.FC_Softmax(fc_params[0]);
    auto end2 = std::chrono::steady_clock::now();
    cout
            << "This CNN calculations took "
            << chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() << "µs ≈ "
            << chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << "ms ≈ "
            << chrono::duration_cast<std::chrono::seconds>(end2 - start2).count() << "s.\n\n";

    return 0;
}

//    maxPollingTest 没有问题
//    float pixelMaxPolling[
//            5 * 5 * 2] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
//                          27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50};
//    picture maxPollingTest(5, 2, pixelMaxPolling);
//    float pixelMaxPolling[4 * 4 * 2] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,27, 28, 29, 30, 31, 32};
//    picture maxPollingTest(4, 4, pixelMaxPolling);
//    maxPollingTest.Maxpool2d(2, 2);
//    for (int i = 0; i < 18; i++) {
//        cout << maxPollingTest.getPixel()[i] << " ";
//        if ((i + 1) % 3 == 0) cout << endl;
//    }

//    weights_convTest 没有问题
//    Matrix m=w_trans(3,3,16,conv0_weight);
//    cout << m;

//    PictureToMatrix Test 没有问题
//    float firstSituation[4*4*3] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48};
//    picture first(4,3,firstSituation);
//    cout<<first.p_trans(3,3,1,2);
//    float secondSituation[5*5*3] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75};
//    picture second(5,3,secondSituation);
//    cout<<second.p_trans(3,3,1,2);

//    convToMatrix Test 没有问题
//    float pweitht[3*3*3*2] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54};
//    float p_bias[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
//    conv_param cp{1,2,3,3,2,pweitht,p_bias};
//    cout<< w_trans(3,3,2,pweitht);
//    cout<< b_trans(p_bias,5,16);