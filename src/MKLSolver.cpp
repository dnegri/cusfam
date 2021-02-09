#include "MKLSolver.h"

MKLSolver::MKLSolver(Geometry& g) : CSRSolver(g)
{
	iparam = new int [64]{};
    idum = new int[1] {};

    for (int i = 0; i < 64; i++) {
        pt[i] = 0;
    }

    iparam[0] = 1; // no solver default
    iparam[1] = 3; // Fill-in reducing ordering for the input matrix.
    iparam[2] = 0; // reserved
    iparam[3] = 41; // LU-preconditioned CGS iteration with a criterion of 1.0E-5 for nonsymmetric matrices
    iparam[4] = 0; // no user fill-in reducing permutation
    iparam[5] = 0; // write solution on x.
    iparam[6] = 0; // output. Number of iterative refinement steps performed.
    iparam[7] = 0; // Maximum number of iterative refinement steps 
    iparam[8] = 0; // reserved
    iparam[9] = 13; // perturb the pivot elements with 1E-13
    iparam[10] = 1; // use nonsymmetric permutation and scaling MPS
    iparam[23] = 1; // two-level factorization algorithm to improve scalability on many OpenMP threads (more than eight)
    iparam[27] = 0; // Input arrays (a, x and b) and all internal arrays must be presented in double precision.
    iparam[33] = 1; // Optimal number of OpenMP threads for conditional numerical reproducibility (CNR) mode.
    iparam[34] = 1; // zero - based indexing of columns and rows.
    iparam[36] = 0; // Use CSR format for matrix storage.
}

MKLSolver::~MKLSolver()
{

}

void MKLSolver::solve(CMFD_VAR * b, double* x)
{
    int error = 0;


    int phase = 33;
    pardiso(pt, &_maxfct, &_mnum, &_mtype, &phase, &_n, _a, _rowptr, _idx_col, idum, &_nrhs, iparam, &_msglvl, b, x, &error);

    if (error != 0) {
        printf("ERROR WHILE SOLVING MKL SOLVER : %d\n", error);
        exit(error);
    }

}

void MKLSolver::prepare()
{
//
//	/* Matrix data. */
//	int n = 8;
//	int ia[9] = { 1, 4, 6, 9, 12, 12, 12, 12, 14 };
//	int ja[13] = { 1, 2, 4,
//		1, 2,
//		3, 4, 5,
//		1, 3, 4,
//		2, 5 };
//	int a[13] = { 1.0, -1.0, -3.0,
//		-2.0, 5.0,
//		4.0, 6.0, 4.0,
//		-4.0, 2.0, 7.0,
//		8.0, -5.0 };
//	MKL_INT mtype = 11; /* Real unsymmetric matrix */
//	/* RHS and solution vectors. */
//	double b[8], x[8];
//	MKL_INT nrhs = 1; /* Number of right hand sides. */
//	/* Internal solver memory pointer pt, */
//	/* 32-bit: int pt[64]; 64-bit: long int pt[64] */
//	/* or void *pt[64] should be OK on both architectures */
//	void* pt[64];
//	/* Pardiso control parameters. */
//	MKL_INT iparm[64];
//	MKL_INT maxfct, mnum, phase, error, msglvl;
//	/* Auxiliary variables. */
//	MKL_INT i;
//	double ddum; /* Double dummy */
//	MKL_INT idum; /* Integer dummy. */
///* -------------------------------------------------------------------- */
///* .. Setup Pardiso control parameters. */
///* -------------------------------------------------------------------- */
//	for (i = 0; i < 64; i++) {
//		iparm[i] = 0;
//	}
//	iparm[0] = 1; /* No solver default */
//	iparm[1] = 2; /* Fill-in reordering from METIS */
//	/* Numbers of processors, value of OMP_NUM_THREADS */
//	iparm[2] = 1;
//	iparm[3] = 0; /* No iterative-direct algorithm */
//	iparm[4] = 0; /* No user fill-in reducing permutation */
//	iparm[5] = 0; /* Write solution into x */
//	iparm[6] = 0; /* Not in use */
//	iparm[7] = 2; /* Max numbers of iterative refinement steps */
//	iparm[8] = 0; /* Not in use */
//	iparm[9] = 8; /* Perturb the pivot elements with 1E-13 */
//	iparm[10] = 1; /* Use nonsymmetric permutation and scaling MPS */
//	iparm[11] = 0; /* Not in use */
//	iparm[12] = 0; /* Not in use */
//	iparm[13] = 0; /* Output: Number of perturbed pivots */
//	iparm[14] = 0; /* Not in use */
//	iparm[15] = 0; /* Not in use */
//	iparm[16] = 0; /* Not in use */
//	iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
//	iparm[18] = -1; /* Output: Mflops for LU factorization */
//	iparm[19] = 0; /* Output: Numbers of CG Iterations */
//	iparm[34] = 1; // zero - based indexing of columns and rows.
//	maxfct = 1; /* Maximum number of numerical factorizations. */
//	mnum = 1; /* Which factorization to use. */
//	msglvl = 1; /* Print statistical information in file */
//	error = 0; /* Initialize error flag */
///* -------------------------------------------------------------------- */
///* .. Initialize the internal solver memory pointer. This is only */
///* necessary for the FIRST call of the PARDISO solver. */
///* -------------------------------------------------------------------- */
//	for (i = 0; i < 64; i++) {
//		pt[i] = 0;
//	}
//	/* -------------------------------------------------------------------- */
//	/* .. Reordering and Symbolic Factorization. This step also allocates */
//	/* all memory that is necessary for the factorization. */
//	/* -------------------------------------------------------------------- */
//	phase = 11;
//	pardiso(pt, &_maxfct, &_mnum, &_mtype, &phase, &_n, _a, _rowptr, _idx_col, &idum, &_nrhs, iparam, &_msglvl, &ddum, &ddum, &error);
//	if (error != 0) {
//		printf("\nERROR during symbolic factorization: %d", error);
//		exit(1);
//	}
//	printf("\nReordering completed ... ");
//	printf("\nNumber of nonzeros in factors = %d", iparm[17]);
//	printf("\nNumber of factorization MFLOPS = %d", iparm[18]);
//	/* -------------------------------------------------------------------- */
//	/* .. Numerical factorization. */
//	/* -------------------------------------------------------------------- */
//	phase = 22;
//	pardiso(pt, &_maxfct, &_mnum, &_mtype, &phase, &_n, _a, _rowptr, _idx_col, &idum, &_nrhs, iparam, &_msglvl, &ddum, &ddum, &error);
//	if (error != 0) {
//		printf("\nERROR during numerical factorization: %d", error);
//		exit(2);
//	}
//	printf("\nFactorization completed ... ");
//	/* -------------------------------------------------------------------- */
//	/* .. Back substitution and iterative refinement. */
//	/* -------------------------------------------------------------------- */
//	phase = 33;
//	iparm[7] = 2; /* Max numbers of iterative refinement steps. */
//	/* Set right hand side to one. */
//	for (i = 0; i < n; i++) {
//		b[i] = 1;
//	}
//	pardiso(pt, &_maxfct, &_mnum, &_mtype, &phase, &_n, _a, _rowptr, _idx_col, &idum, &_nrhs, iparam, &_msglvl, b, x, &error);
//	if (error != 0) {
//		printf("\nERROR during solution: %d", error);
//		exit(3);
//	}
//	printf("\nSolve completed ... ");
//	printf("\nThe solution of the system is: ");
//	for (i = 0; i < n; i++) {
//		printf("\n x [%d] = % f", i, x[i]);
//	}
//	printf("\n");

    int error=0;
    //Reordering and Symbolic Factorization, This step also allocates all memory that is necessary for the factorization

    int phase = 11; //only reorderingand symbolic factorization
    pardiso(pt, &_maxfct, &_mnum, &_mtype, &phase, &_n, _a, _rowptr, _idx_col, idum, &_nrhs, iparam, &_msglvl, nullptr, nullptr, &error);


    if (error != 0) {
        printf("ERROR WHILE PREPARING MKL SOLVER : %d\n", error);
        exit(error);
    }

    phase = 22; //only reorderingand symbolic factorization
    pardiso(pt, &_maxfct, &_mnum, &_mtype, &phase, &_n, _a, _rowptr, _idx_col, idum, &_nrhs, iparam, &_msglvl, nullptr, nullptr, &error);


    if (error != 0) {
        printf("ERROR WHILE PREPARING MKL SOLVER : %d\n", error);
        exit(error);
    }
}
