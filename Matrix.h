#include <string>
#include <iostream>

#ifndef PROJECT1_MULTIPLY_H
#define PROJECT1_MULTIPLY_H
#include <atomic>
using namespace std;

class Matrix {
private:
    int row;
    int column;
    float *data;
    atomic_int *count = 0;

public:
    Matrix();
    Matrix(int row,int column);
    Matrix(int row,int column,const float * data);

//    Matrix(int r, int c, float *da = NULL);

    Matrix(const Matrix &m);

    ~Matrix();
    float * getData() const{return data;}
    atomic_int * getCount() const{return count;}
    void setCount(int  count){ *(this->count)= count;}
    Matrix operator*(const Matrix &m) const;
    int getRow() const{return row;}
    int getColumn() const{return column;}

    friend Matrix operator*(float f, Matrix &m);

    friend Matrix operator*(Matrix &m, float f);

    Matrix operator+(const Matrix &m) const;

    Matrix operator-(const Matrix &m) const;

    Matrix &operator=(const Matrix &m);

    friend ostream &operator<<(ostream &os, const Matrix &m);
};

#endif //PROJECT1_MULTIPLY_H
