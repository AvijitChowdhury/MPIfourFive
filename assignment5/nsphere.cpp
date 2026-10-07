/*
   159.735 Parallel Programming - Assignment 5
   Sequential CPU implementation: count lattice points inside an N-dimensional hypersphere.

   Compile:  g++ -O3 -o nsphere nsphere.cpp
   Run:      ./nsphere [ndim] [radius]
   Example:  ./nsphere 2 2.05    -> 13
             ./nsphere 3 1.5     -> 19
             ./nsphere 1 25.5    -> 51
*/
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>

// Compute n^k (integer exponentiation)
long powlong(long n, long k) {
    long p = 1;
    for (long i = 0; i < k; ++i) p *= n;
    return p;
}

// Convert decimal 'num' to base-B digits stored in index[]
void convert(long num, long base, std::vector<long>& index) {
    const long ndim = index.size();
    for (long i = 0; i < ndim; ++i) index[i] = 0;
    long idx = 0;
    while (num != 0) {
        index[idx++] = num % base;
        num /= base;
    }
}

// ---------------------------------------------------------------
// Algorithm 1: convert loop counter to base-B index each iteration
// ---------------------------------------------------------------
long count_in_v1(long ndim, double radius) {
    const long   halfb   = static_cast<long>(floor(radius));
    const long   base    = 2 * halfb + 1;
    const double rsquare = radius * radius;
    const long   ntotal  = powlong(base, ndim);

    long count = 0;
    std::vector<long> index(ndim, 0);

    for (long n = 0; n < ntotal; ++n) {
        convert(n, base, index);
        double rtestsq = 0.0;
        for (long k = 0; k < ndim; ++k) {
            double xk = index[k] - halfb;
            rtestsq += xk * xk;
        }
        if (rtestsq < rsquare) ++count;
    }
    return count;
}

// ---------------------------------------------------------------
// Algorithm 2: tick a digital counter (avoids repeated division)
// ---------------------------------------------------------------
void addone(std::vector<long>& index, long base, long i) {
    long ndim = index.size();
    long newv = index[i] + 1;
    if (newv >= base) {
        index[i] = 0;
        if (i < ndim - 1) addone(index, base, i + 1);
    } else {
        index[i] = newv;
    }
}

long count_in_v2(long ndim, double radius) {
    const long   halfb   = static_cast<long>(floor(radius));
    const long   base    = 2 * halfb + 1;
    const double rsquare = radius * radius;
    const long   ntotal  = powlong(base, ndim);

    long count = 0;
    std::vector<long> index(ndim, 0);

    for (long n = 0; n < ntotal; ++n) {
        double rtestsq = 0.0;
        for (long k = 0; k < ndim; ++k) {
            double xk = index[k] - halfb;
            rtestsq += xk * xk;
        }
        if (rtestsq < rsquare) ++count;
        addone(index, base, 0);
    }
    return count;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " [ndim] [radius]" << std::endl;
        return 1;
    }
    const int    ndim = std::atoi(argv[1]);
    const double r    = std::atof(argv[2]);

    std::cout << "Dimensions : " << ndim << std::endl;
    std::cout << "Radius     : " << r    << std::endl;

    const long num1 = count_in_v1(ndim, r);
    const long num2 = count_in_v2(ndim, r);

    std::cout << "Count (v1) : " << num1 << std::endl;
    std::cout << "Count (v2) : " << num2 << std::endl;
    std::cout << "Match      : " << (num1 == num2 ? "YES" : "NO") << std::endl;

    return 0;
}
