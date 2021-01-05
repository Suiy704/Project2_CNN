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
