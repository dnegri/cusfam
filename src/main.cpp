#include <iostream>
#include <ctime>
#include <numeric>
#include "CrossSection.h"
#include "Solution.h"

#define m_phif(ig,l,k) d_phif[(k*nxy+l)*ng+ig]
#define m_xsnf(ig,l,k) d_xsnf[(k*nxy+l)*ng+ig]

#define op *

int main() {

    int ng=8;
    int nx=16*30;
    int ny=16*30;
    int nz=300;
    int nxy=nx*ny;

    double * d_phif = new double[ng*nxy*nz];
    double * d_xsnf = new double[ng*nxy*nz];
    double * d_psi = new double[nxy*nz];

    CrossSection xs;
    xs.init(ng, nxy, nz);

    Solution s;
    s.init(ng, nxy, nz);

    double sum=0.0;

    double * p_phif = d_phif, *p_xsnf=d_xsnf;

    p_phif = d_phif;
    auto begin = clock();
    auto end = clock();
    auto elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    begin = clock();
    for (int k = 0; k < nz; ++k) {
        for (int l = 0; l < nxy; ++l) {
            for (int ig = 0; ig < ng; ++ig) {
                double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
                s.phi(ig,l,k) = r;
                xs.xsnf(ig,l,k) = r*0.5;
            }
        }
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    printf("Elapsed time for initializing values by object is %12.5f\n", elapsed_secs);

    begin = clock();
    for (int k = 0; k < nz; ++k) {
        for (int l = 0; l < nxy; ++l) {
            for (int ig = 0; ig < ng; ++ig) {
                double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
                m_phif(ig,l,k) = r;
                m_xsnf(ig,l,k) = r*0.5;
            }
        }
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    printf("Elapsed time for initializing values by macro is %12.5f\n", elapsed_secs);





    sum=0.0;
    for (int i = 0; i < 10; ++i) {
        begin = clock();
        for (int k = 0; k < nz; ++k) {
            for (int l = 0; l < nxy; ++l) {
                for (int ig = 0; ig < ng; ++ig) {
                    sum += s.phi(ig,l,k) op xs.xsnf(ig,l,k);
                }
            }
        }
        end = clock();

        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//    accumulate(d_psi , d_psi+nxy*nz , sum);

        printf("Elapsed time with (ig,l,k) object is %12.5f seconds and sum is %e\n", elapsed_secs, sum);
    }



    sum=0.0;
    for (int i = 0; i < 10; ++i) {
        begin = clock();
        for (int k = 0; k < nz; ++k) {
            for (int l = 0; l < nxy; ++l) {
                for (int ig = 0; ig < ng; ++ig) {
                    sum += m_phif(ig,l,k) op m_xsnf(ig,l,k);
                }
            }
        }
        end = clock();

        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//    accumulate(d_psi , d_psi+nxy*nz , sum);

        printf("Elapsed time with (ig,l,k) macro is %12.5f seconds and sum is %e\n", elapsed_secs, sum);
    }



    return 0;
}
