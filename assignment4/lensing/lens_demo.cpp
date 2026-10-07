/* 
   159.735 Parallel Programming - Assignment 4
   Sequential CPU implementation of gravitational lensing by inverse ray tracing.
   
   Compile:  g++ -O3 -o lens_demo lens_demo.cpp fitsfile.o lenses.o -lcfitsio -lm
   Run:      ./lens_demo
*/
#include <ctime>
#include <iostream>
#include <cmath>
#include "lenses.h"
#include "arrayff.hxx"

// Lens plane physical boundaries
const float WL  = 2.0f;
const float XL1 = -WL;
const float XL2 =  WL;
const float YL1 = -WL;
const float YL2 =  WL;

double diffclock(clock_t c1, clock_t c2) {
    return (c1 - c2) * 1000.0 / CLOCKS_PER_SEC;
}

int main(int argc, char* argv[])
{
    // Choose lens configuration: set_example_1, _2, _3, or _n(N,...)
    float *xlens, *ylens, *eps;
    const int nlenses = set_example_3(&xlens, &ylens, &eps);
    std::cout << "# Simulating " << nlenses << " lens system" << std::endl;

    // Source star parameters
    const float rsrc = 0.1f;   // radius
    const float ldc  = 0.5f;   // limb darkening coefficient
    const float xsrc = 0.0f;
    const float ysrc = 0.0f;
    const float rsrc2 = rsrc * rsrc;

    // Image resolution (decrease lens_scale for a finer, larger image)
    const float lens_scale = 0.005f;
    const int npixx = static_cast<int>(floor((XL2 - XL1) / lens_scale)) + 1;
    const int npixy = static_cast<int>(floor((YL2 - YL1) / lens_scale)) + 1;
    std::cout << "# Building " << npixx << "x" << npixy << " lens image" << std::endl;

    Array<float, 2> lensim(npixy, npixx);

    clock_t tstart = clock();

    int numuse = 0;
    for (int iy = 0; iy < npixy; ++iy)
    for (int ix = 0; ix < npixx; ++ix) {
        // Map pixel to physical position on the lens plane
        float xl = XL1 + ix * lens_scale;
        float yl = YL1 + iy * lens_scale;

        // Shoot ray back to source plane via lens equation
        float xs, ys;
        shoot(xs, ys, xl, yl, xlens, ylens, eps, nlenses);

        // Test if ray lands within the source star disc
        float xd   = xs - xsrc;
        float yd   = ys - ysrc;
        float sep2 = xd * xd + yd * yd;
        if (sep2 < rsrc2) {
            float mu = sqrtf(1.0f - sep2 / rsrc2);
            lensim(iy, ix) = 1.0f - ldc * (1.0f - mu);   // limb darkening
            ++numuse;
        }
    }

    clock_t tend = clock();
    std::cout << "# Time elapsed: " << diffclock(tend, tstart)
              << " ms   pixels lit: " << numuse << std::endl;

    dump_array<float, 2>(lensim, "lens.fit");
    std::cout << "# Written to lens.fit" << std::endl;

    delete[] xlens; delete[] ylens; delete[] eps;
    return 0;
}
