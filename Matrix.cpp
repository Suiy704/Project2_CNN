#include "Matrix.h"
#include <iostream>
#include <string.h>

using namespace std;

float dotProduct(const float *p1, const float *p2, size_t n, size_t temp) {
    float sum = 0.0f;
    for (size_t i = 0, j = 0; i < n; i++, j += temp)
        sum += (p1[i] * p2[j]);
    return sum;
}

Matrix::Matrix() {
    row = 0;
    column = 0;
    data = nullptr;
    count = new atomic_int;
    *count= 1;
}

Matrix::Matrix(int row,int column) {
    this->row = row;
    this->column = column;
    this->data=new float[row*column]();
    this->count=new atomic_int ;
    *this->count=1;
}

Matrix::Matrix(int row,int column,const float * data) {
    this->row = row;
    this->column = column;
    this->data=new float[row*column]();
    for (int i = 0; i < row*column; ++i) {
        *(this->data+i)=*(data+i);
    }
    this->count=new atomic_int ;
    *this->count=1;
}

Matrix::Matrix(const Matrix &m) {
    row = m.row;
    column = m.column;
    data = m.data;
    count = m.count;
    (*count) += 1;
//    cout << (*count) << "   Copy constructor." << endl;
}

Matrix::~Matrix() {
//    cout << *count << endl;
    if (*(count) == 1) {
        delete[] data;
        delete count;
//        cout << "data has been deleted." << endl;
    } else {
        (*count)--;
//        cout << "count has been minus 1." << endl;
    }
}

Matrix Matrix::operator*(const Matrix &m) const {
    if (column == m.row) {
        Matrix resultMatrix(row,m.column);
        resultMatrix.data = new float[this->row * m.column]();
        int k = 0;
        for (int j = 0; j < m.getColumn(); j++) {
            for (int i = 0; i < this->row; i++) {
                float result = dotProduct(this->data + i * this->column, m.data + j,
                                          this->column, m.column);
                *(resultMatrix.data + k) = result;
                k++;
            }
        }
//        return resultMatrix;

//        float *result = new float[this->row * m.column];
//        for (int i = 0; i < m.column; i++) {
//            for (int j = 0; j < this->row; ++j) {
//                for (int k = 0; k < this->column; ++k) {
//                    resultMatrix.data[i*this->row+j]= resultMatrix.data[i*this->row+j] + (data[j* this->column +k] * m.data[k*m.column+i]);
//
//                }
//            }
//        }
//        Matrix results(row,m.column, result);
        return resultMatrix;
    } else {
        cout << "These two matrices cannot be multiplied." << endl << endl;
        Matrix results(-1, -1, 0);
        return results;
    }
}

Matrix Matrix::operator+(const Matrix &m) const {
    if (row == m.row && column == m.column) {
        Matrix results(row, m.column);
        //   float *result = new float[row * m.column]();
        for (int i = 0; i < row * column; ++i) {
            *(results.data + i) = *(data + i) + *(m.data + i);
        }
        return results;
    } else {
        cout << "These two matrices cannot be added." << endl << endl;
        Matrix results(-1, -1, 0);
        return results;
    }

}

Matrix Matrix::operator-(const Matrix &m) const {
    if (row == m.row && column == m.column) {
        float *result = new float[row * m.column]();
        for (int i = 0; i < row * column; ++i) {
            *(result + i) = *(data + i) - *(m.data + i);
//                cout << *(result+i) << endl;

        }
        Matrix results(row, column, result);
        return results;
    } else {
        Matrix results(-1, -1, 0);
        cout << "These two matrices cannot be subtracted." << endl << endl;
        return results;
    }
}

Matrix operator*(float f, Matrix &m) {
    float *result = new float[m.row * m.column]();
    for (int i = 0; i < m.row; ++i) {
        for (int j = 0; j < m.column; ++j) {
            *(result + i * m.column + j) = f * *(m.data + i * m.column + j);
//            cout << *(result+i*m.column+j) << endl;
        }
    }
    Matrix results(m.row, m.column, result);
    return results;
}

Matrix operator*(Matrix &m, float f) {
    float *result = new float[m.row * m.column]();
    for (int i = 0; i < m.row; ++i) {
        for (int j = 0; j < m.column; ++j) {
            *(result + i * m.column + j) = f * *(m.data + i * m.column + j);
//            cout << *(result+i*m.column+j) << endl;
        }
    }
    Matrix results(m.row, m.column, result);
    return results;
}

Matrix &Matrix::operator=(const Matrix &m) {
    if (this == &m) {
        return *this;
    }
    if (*(count) == 1) {
        delete count;
        delete[] data;
    } else {
        (*count)--;
    }

    row = m.row;
    column = m.column;
    data=m.data;
    count = m.count;
    (*count)++;
    return *this;
}

ostream &operator<<(ostream &os, const Matrix &m) {
    cout.setf(ios_base::floatfield, ios_base::fixed);
    cout.precision(2);
    if (m.row != -1) {
        for (int i = 0; i < m.row; ++i) {
            for (int j = 0; j < m.column; ++j) {
                os << *(m.data + i * m.column + j) << " ";
            }
            os << endl;
        }
    }
    os << endl;
    return os;
}



