/* shared.h: a set of simple functions used by both exponential.h and normal.h */
#pragma once

#include <stdlib.h>
#include <math.h>
#include "MT19937.h"

/* All of the arrays are scaled by 2**63 (not 2**64), so this operation is used 
* to draw uniform values in the ziggurat overhangs*/

#define MAX_INT64   0x7fffffffffffffff
#define RANDOM_INT63() ( Rand++->sl & MAX_INT64 )

/* Test to see if rejection sampling is required in the overhang. See Fig. 2
 * in main text. */

#define _FAST_PRNG_SAMPLE_X(X_j, U) (   *(X_j)*pow(2, 63) + ((X_j)[-1] - *(X_j) )*(U) )
#define _FAST_PRNG_SAMPLE_Y(i,   U) ( Y[(i)-1]*pow(2, 63) + (Y[(i)  ] - Y[(i)-1])*(U) )

