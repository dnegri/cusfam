from itertools import islice
from random import random
from time import perf_counter
#from cusfam import fast_tanh
from cusfam import *
import time

if __name__ == "__main__":
    init("../run/geom.simon", "../run/KMYGN34C01_PLUS7_XSE.XS", "../run");
    setBurnup(1.0);
    s = SteadyOption();
    s.crit = CBC;
    s.feedtf = True;
    s.feedtm = True;
    s.xenon = XE_EQ;
    s.tin = 295.8;
    s.eigvt = 1.0;
    s.maxiter = 100;
    s.ppm = 800.0;
    s.plevel = 1.0;
    burn_del = 1000.0;
    burn_end = 13650;
    burn_step = int(burn_end/burn_del);

    start = time.time();
    for i in range(burn_step) :
        calcStatic(s);
        print("STEP : ", i, "         PPM : ", s.ppm);
        deplete(XE_EQ, SM_TR, burn_del);

    end = time.time();
    print(end - start);
    print("END");
    
