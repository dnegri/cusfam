#include <iostream>
#include <ctime>
#include "blitz/array.h"
#include <numeric>

using namespace blitz;

#define m_phif(ig,l,k) d_phif[(k*nxy+l)*ng+ig]
#define m_xsnf(ig,l,k) d_xsnf[(k*nxy+l)*ng+ig]
#define m_psi(l,k) d_psi[k*nxy+l]

#define op /

int main() {

    int ng=8;
    int nx=16*30;
    int ny=16*30;
    int nz=300;
    int nxy=nx*nz;

    Array<float, 3> phif_col(ng, nxy, nz, ColumnMajorArray<3>()), xsnf_col(ng, nxy, nz, ColumnMajorArray<3>());
    Array<float, 3> phif_for(Range(1,ng), Range(0,nxy), Range(0,nz), FortranArray<3>()), xsnf_for(Range(1,ng), Range(0,nxy), Range(0,nz), FortranArray<3>());
    Array<float, 2> psi(nxy,nz,ColumnMajorArray<2>());

    Array<float, 3> phif_row(ng, nxy, nz), xsnf_row(ng, nxy, nz);
    Array<float, 2> psi3(nxy,nz);

    Array<float, 3> phif_grp(nz, nxy, ng), xsnf_grp(nz, nxy, ng);
    Array<float, 2> psi2(nz,nxy);

    phif_for(1,0,0) = 0.0;

    float * d_phif = new float[ng*nxy*nz];
    float * d_xsnf = new float[ng*nxy*nz];
    float * d_psi = new float[nxy*nz];

    float * p_phif = d_phif, *p_xsnf=d_xsnf;

    auto begin = clock();

    for (int k = 0; k < nz; ++k) {
        for (int l = 0; l < nxy; ++l) {
            for (int ig = 0; ig < ng; ++ig) {
                float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                *p_phif++ = r;
                *p_xsnf++ = r*0.5;
            }
        }
    }
    auto end = clock();
    auto elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    printf("Elapsed time for initializing values by legacy is %12.5f\n", elapsed_secs);


    begin = clock();
    for (int k = 0; k < nz; ++k) {
        for (int l = 0; l < nxy; ++l) {
            for (int ig = 0; ig < ng; ++ig) {
                float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                phif_col(ig, l, k) = r;
                xsnf_col(ig, l, k) = r * 0.5;
            }
        }
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    printf("Elapsed time for initializing values by blitz column major is %12.5f\n", elapsed_secs);


    begin = clock();
    for (int k = 0; k < nz; ++k) {
        for (int l = 0; l < nxy; ++l) {
            for (int ig = 0; ig < ng; ++ig) {
                float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                phif_grp(k, l, ig) = r;
                xsnf_grp(k, l, ig) = r * 0.5;
            }
        }
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    printf("Elapsed time for initializing values by blitz grp major is %12.5f\n", elapsed_secs);

    begin = clock();
    for (int k = 0; k < nz; ++k) {
        for (int l = 0; l < nxy; ++l) {
            for (int ig = 0; ig < ng; ++ig) {
                float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                phif_row(ig, l, k) = r;
                xsnf_row(ig, l, k) = r * 0.5;
            }
        }
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    printf("Elapsed time for initializing values by blitz row major is %12.5f\n", elapsed_secs);

    begin = clock();
    for (int k = 0; k <= nz; ++k) {
        for (int l = 0; l <= nxy; ++l) {
            for (int ig = 1; ig <= ng; ++ig) {
                float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                phif_for(ig, l, k) = r;
                xsnf_for(ig, l, k) = r * 0.5;
            }
        }
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    printf("Elapsed time for initializing values by blitz fortran major is %12.5f\n", elapsed_secs);


    auto sum=0.0;
    for (int i=0; i<10; ++i) {
        p_phif = d_phif, p_xsnf=d_xsnf;
        begin = clock();
        for (int k = 0; k < nz; ++k) {
            for (int l = 0; l < nxy; ++l) {
//            m_psi(l,k) = 0.0;
                for (int ig = 0; ig < ng; ++ig) {
                    sum += *(p_phif) op *(p_xsnf);
//                m_psi(l,k) += *p_phif++ op *p_xsnf++;
                }
            }
        }
        end = clock();

        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//    accumulate(d_psi , d_psi+nxy*nz , sum);
        printf("Elapsed time with pointer is %12.5f seconds and sum is %e\n", elapsed_secs, sum);
    }


    sum=0.0;
    for (int i=0; i<10; ++i) {
        begin = clock();
        for (int k = 0; k < nz; ++k) {
            for (int l = 0; l < nxy; ++l) {
//            m_psi(l,k) = 0.0;
                for (int ig = 0; ig < ng; ++ig) {
                    sum += d_phif[i] op d_xsnf[i];
//                m_psi(l,k) += d_phif[i] op d_xsnf[i];
//                i++;
                }
            }
        }
        end = clock();

        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//    accumulate(d_psi , d_psi+nxy*nz , sum);

        printf("Elapsed time with [i] is %12.5f seconds and sum is %e\n", elapsed_secs, sum);
    }


    sum=0.0;
    for (int i=0; i<10; ++i) {
        begin = clock();
        for (int k = 0; k < nz; ++k) {
            for (int l = 0; l < nxy; ++l) {
                auto lk0 = k * nxy + l;
//            m_psi(l,k) = 0.0;
                for (int ig = 0; ig < ng; ++ig) {
                    sum += d_phif[lk0 * ng + ig] op d_xsnf[lk0 * ng + ig];
//                m_psi(l,k) += d_phif[lk0*ng+ig] op d_xsnf[lk0*ng+ig];
                }
            }
        }
        end = clock();

        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//    accumulate(d_psi , d_psi+nxy*nz , sum);

        printf("Elapsed time with (ig,lk) index is %12.5f seconds and sum is %e\n", elapsed_secs, sum);
    }

    sum=0.0;
    for (int i=0; i<10; ++i) {
        begin = clock();
        for (int k = 0; k < nz; ++k) {
            for (int l = 0; l < nxy; ++l) {
//            m_psi(l,k) = 0.0;
                for (int ig = 0; ig < ng; ++ig) {
                    auto lk0 = k * nxy + l;
                    sum += d_phif[(k * nxy + l) * ng + ig] op d_xsnf[(k * nxy + l) * ng + ig];
//                m_psi(l,k) += d_phif[(k*nxy+l)*ng+ig] op d_xsnf[(k*nxy+l)*ng+ig];
                }
            }
        }
        end = clock();
//    accumulate(d_psi , d_psi+nxy*nz , sum);

        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

        printf("Elapsed time with (ig,l,k) index is %12.5f seconds and sum is %e\n", elapsed_secs, sum);
    }


    sum=0.0;
    for (int i=0; i<10; ++i) {
        begin = clock();
        for (int k = 0; k < nz; ++k) {
            for (int l = 0; l < nxy; ++l) {
                m_psi(l, k) = 0.0;
                for (int ig = 0; ig < ng; ++ig) {
                    auto lk0 = k * nxy + l;
                    sum += m_phif(ig, l, k) op m_xsnf(ig, l, k);
//                m_psi(l,k) += m_phif(ig,l,k) op m_xsnf(ig,l,k);
                }
            }
        }
        end = clock();

        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//    accumulate(d_psi , d_psi+nxy*nz , sum);

        printf("Elapsed time with (ig,l,k) macro is %12.5f seconds and sum is %e\n", elapsed_secs, sum);
    }

//    sum=0.0;
//    for (int i=0; i<10; ++i) {
//        begin = clock();
//        for (int k = 0; k < nz; ++k) {
//            for (int l = 0; l < nxy; ++l) {
//                for (int ig = 0; ig < ng; ++ig) {
////                psi3(l,k) = 0.0;
//                    sum += (phif_row(ig, l, k) op xsnf_row(ig, l, k));
////                psi2(k,l) += (phif_col(k,l,ig) op xsnf_col(k,l,ig));
//                }
//            }
//        }
//        end = clock();
//
//        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
////    accumulate(psi2.data() , psi2.data()+nxy*nz , sum);
//
//        printf("Elapsed time with blitz row major is %12.5f seconds and sum is %e\n", elapsed_secs, sum);
//    }

    sum=0.0;
    for (int i=0; i<10; ++i) {
        begin = clock();
        for (int k = 0; k < nz; ++k) {
            for (int l = 0; l < nxy; ++l) {
//            psi(l,k) = 0.0;
                for (int ig = 0; ig < ng; ++ig) {
                    sum += (phif_col(ig, l, k) op xsnf_col(ig, l, k));
//                psi(l,k) += (phif_col(ig,l,k) op xsnf_col(ig,l,k));
                }
            }
        }
        end = clock();

        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//    accumulate(psi.data() , psi.data()+nxy*nz , sum);

        printf("Elapsed time with blitz column major is %12.5f seconds and sum is %e\n", elapsed_secs, sum);
    }

//    sum=0.0;
//    for (int i=0; i<10; ++i) {
//        begin = clock();
//        for (int ig = 0; ig < ng; ++ig) {
//            for (int l = 0; l < nxy; ++l) {
////            psi2(k,l) = 0.0;
//                for (int k = 0; k < nz; ++k) {
//                    sum += (phif_grp(k, l, ig) op xsnf_grp(k, l, ig));
////                psi2(k,l) += (phif_col(k,l,ig) op xsnf_col(k,l,ig));
//                }
//            }
//        }
//        end = clock();
//
//        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
////    accumulate(psi2.data() , psi2.data()+nxy*nz , sum);
//
//        printf("Elapsed time with blitz group major is %12.5f seconds and sum is %e\n", elapsed_secs, sum);
//    }

    sum=0.0;
    for (int i=0; i<10; ++i) {
        begin = clock();
        for (int k = 1; k <= nz; ++k) {
            for (int l = 1; l <= nxy; ++l) {
//            psi2(k,l) = 0.0;
                for (int ig = 1; ig <= ng; ++ig) {
                    sum += (phif_for(ig, l, k) op xsnf_for(ig, l, k));
//                psi2(k,l) += (phif_col(k,l,ig) op xsnf_col(k,l,ig));
                }
            }
        }
        end = clock();

        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//    accumulate(psi2.data() , psi2.data()+nxy*nz , sum);

        printf("Elapsed time with blitz fortran for is %12.5f seconds and sum is %e\n", elapsed_secs, sum);
    }
    return 0;
}
