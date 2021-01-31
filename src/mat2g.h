#pragma once


#define d(i)    d[i]
#define a(i,j)  a[j*2+i]
#define b(i,j)  b[j*2+i]
#define c(i,j)  c[j*2+i]

#define x(i)  x[i]
#define r(i)  r[i]


template <class T>
void matxdiag2g(T* a, T* d, T* c) {
    c(0, 0) = a(0, 0) * d(0);
    c(1, 0) = a(1, 0) * d(1);
    c(0, 1) = a(0, 1) * d(0);
    c(1, 1) = a(1, 1) * d(1);
}

template <class T>
void diagxmat2g(T* d, T* b, T* c) {
    c(0, 0) = d(0) * b(0, 0);
    c(1, 0) = d(0) * b(1, 0);
    c(0, 1) = d(1) * b(0, 1);
    c(1, 1) = d(1) * b(1, 1);
}

template <class T>
void matxmat2g(T* a, T* b, T* c) {
    c(0, 0) = a(0, 0) * b(0, 0) + a(1, 0) * b(0, 1);
    c(1, 0) = a(0, 0) * b(1, 0) + a(1, 0) * b(1, 1);
    c(0, 1) = a(0, 1) * b(0, 0) + a(1, 1) * b(0, 1);
    c(1, 1) = a(0, 1) * b(1, 0) + a(1, 1) * b(1, 1);
}

template <class T>
void matxvec2g(T* a, T* r, T* x) {

    x(0) = a(0, 0) * r(0) + a(1, 0) * r(1);
    x(1) = a(0, 1) * r(0) + a(1, 1) * r(1);
}

template <class T>
void addmat2g(T* a, T* b, T* c) {

    c(0, 0) = a(0, 0) + b(0, 0);
    c(1, 0) = a(1, 0) + b(1, 0);
    c(0, 1) = a(0, 1) + b(0, 1);
    c(1, 1) = a(1, 1) + b(1, 1);

}

template <class T>
void submat2g(T* a, T* b, T* c) {

    c(0, 0) = a(0, 0) - b(0, 0);
    c(1, 0) = a(1, 0) - b(1, 0);
    c(0, 1) = a(0, 1) - b(0, 1);
    c(1, 1) = a(1, 1) - b(1, 1);

}

template <class T>
void copyTomat2g(T* a, T* b) {

    b(0, 0) = a(0, 0);
    b(1, 0) = a(1, 0);
    b(0, 1) = a(0, 1);
    b(1, 1) = a(1, 1);

}


template <class T>
void invmat2g(T* a, T* b) {

    T rdet = 1 / (a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1));
    b(0, 0) = rdet * a(1, 1);
    b(1, 0) = -rdet * a(1, 0);
    b(0, 1) = -rdet * a(0, 1);
    b(1, 1) = rdet * a(0, 0);
}

template <class T>
void solmat2g(T* a, T* b, T* x) {
    T c[4];

    invmat2g<T>(a, c);
    matxvec2g<T>(c, b, x);
}

#undef a
#undef b
#undef c
#undef x
#undef r
