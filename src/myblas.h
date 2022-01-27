#pragma once

namespace myblas {
    template<class T>
    T dot(const int &n, const T *a, const T *b) {

        T sum = 0.0;
#pragma omp parallel for reduction( + : sum )
        for (int i = 0; i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    template<class T>
    void multi(const int &n, float &s, const T *b, T *c) {
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            c[i] = s * b[i];
        }
    }

    template<class T>
    void multi(const int &n, const double &s, const T *b, T *c) {
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            c[i] = s * b[i];
        }
    }


    template<class T>
    void plus(const int &n, const T *a, const T *b, T *c) {
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }

    template<class T>
    void minus(const int &n, const T *a, const T *b, T *c) {
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            c[i] = a[i] - b[i];
        }
    }

    template<class T>
    void to(const int &n, const T *a, T *b) {
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            b[i] = a[i];
        }
    }

}
