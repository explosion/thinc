#pragma once
#include <unistd.h>
#include <time.h>
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

/** New functionality:
 *
 *  __cycle__ (def: 500): defines the size of random number array in bytes*16. 
 *  It should be larger than 382, otherwise will SEG-FAULT. 
 *
 *	rand64_t: 64-bit type for access of random bytes
 *	dw128_t	: 128-bit type for SIMD instructions
 *
 *  mt_init(): initializes PRN generator. Seeds from CPU time, PID, and parent 
 *  PID. Can be called multiple times without any issues or overhead. 
 *
 *  wide_uniform()              -> 128 random bits.
 *  uniform_double_PRN()        -> uniform double PRN on domain [0, 1).
 *  rand_long(unsigned long n)  -> uniform unsigned long PRN on [0, n).
 *  rand_long64()               -> uniform unsigned long PRN on [2, 2^64).
 *
 *  static rand64_t Rand: a public random variable that points to the next 
 *  unused element in the random number array. This pointer allows for more
 *  rapid access of these numbers, but can SEG FAULT if used improperly. 
 *  After 64 random bits are used, this pointer should be incremented. Call
 *  MT_FLUSH() every two times this pointer is incremented to avoid a SEG FAULT.
 *
 *	MT_FLUSH(): repopulates array of random numbers when Rand is near end of 
 *	array and moves Rand back to beginning of array.  
 * 
 *  Defining the variable REPORT_PRNS will execute a command that reports the
 *  number of PRNs used during execution at exit. 
 * 
 *	Lastly, this code now uses 19937 as the default Mersenne Prime.
 *
 **/

#define __cycle__		500	

/** 
 * Original Documentation: 
 *
 *
 * @file SFMT.h 
 *
 * @brief SIMD oriented Fast Mersenne Twister(SFMT) pseudorandom
 * number generator
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (Hiroshima University)
 *
 * Copyright (C) 2006, 2007 Mutsuo Saito, Makoto Matsumoto and Hiroshima
 * University. All rights reserved.
 *
 * The new BSD License is applied to this software.
 * see LICENSE.txt
 *
 * @note We assume that your system has inttypes.h.  If your system
 * doesn't have inttypes.h, you have to typedef uint32_t and uint64_t,
 * and you have to define PRIu64 and PRIx64 in this file as follows:
 * @verbatim
 typedef unsigned int uint32_t
 typedef unsigned long long uint64_t  
 #define PRIu64 "llu"
 #define PRIx64 "llx"
@endverbatim
 * uint32_t must be exactly 32-bit unsigned integer type (no more, no
 * less), and uint64_t must be exactly 64-bit unsigned integer type.
 * PRIu64 and PRIx64 are used for printf function to print 64-bit
 * unsigned int and 64-bit unsigned int in hexadecimal format.
 */
 
#ifndef SFMT_H
#define SFMT_H

#define MEXP	19937

#include <stdio.h>

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
  #include <inttypes.h>
#elif defined(_MSC_VER) || defined(__BORLANDC__)
  typedef unsigned int uint32_t;
  typedef unsigned __int64 uint64_t;
  #define inline __inline
#else
  #include <inttypes.h>
  #if defined(__GNUC__)
    #define inline __inline__
  #endif
#endif

#ifndef PRIu64
  #if defined(_MSC_VER) || defined(__BORLANDC__)
    #define PRIu64 "I64u"
    #define PRIx64 "I64x"
  #else
    #define PRIu64 "llu"
    #define PRIx64 "llx"
  #endif
#endif

#if defined(__GNUC__)
#define ALWAYSINLINE __attribute__((always_inline))
#else
#define ALWAYSINLINE
#endif

#if defined(_MSC_VER)
  #if _MSC_VER >= 1200
    #define PRE_ALWAYS __forceinline
  #else
    #define PRE_ALWAYS inline
  #endif
#else
  #define PRE_ALWAYS inline
#endif



#endif


/** 
 * @file  SFMT.c
 * @brief SIMD oriented Fast Mersenne Twister(SFMT)
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (Hiroshima University)
 *
 * Copyright (C) 2006,2007 Mutsuo Saito, Makoto Matsumoto and Hiroshima
 * University. All rights reserved.
 *
 * The new BSD License is applied to this software, see LICENSE.txt
 */
#include <string.h>
#include <assert.h>

#ifndef SFMT_PARAMS_H
#define SFMT_PARAMS_H

/*-----------------
  BASIC DEFINITIONS
  -----------------*/
/** Mersenne Exponent. The period of the sequence 
 *  is a multiple of 2^MEXP-1.
 * #define MEXP 19937 */
/** SFMT generator has an internal state array of 128-bit integers,
 * and iN is its size. */
#define iN (MEXP / 128 + 1)
/** N32 is the size of internal state array when regarded as an array
 * of 32-bit integers.*/
#define N32 (iN * 4)
/** N64 is the size of internal state array when regarded as an array
 * of 64-bit integers.*/
#define N64 (iN * 2)

/*----------------------
  the parameters of SFMT
  following definitions are in paramsXXXX.h file.
  ----------------------*/
/** the pick up position of the array.Fwd: Renting the Harvard Cabin
#define POS1 122 
*/

/** the parameter of shift left as four 32-bit registers.
#define SL1 18
 */

/** the parameter of shift left as one 128-bit register. 
 * The 128-bit integer is shifted by (SL2 * 8) bits. 
#define SL2 1 
*/

/** the parameter of shift right as four 32-bit registers.
#define SR1 11
*/

/** the parameter of shift right as one 128-bit register. 
 * The 128-bit integer is shifted by (SL2 * 8) bits. 
#define SR2 1 
*/

/** A bitmask, used in the recursion.  These parameters are introduced
 * to break symmetry of SIMD.
#define MSK1 0xdfffffefU
#define MSK2 0xddfecb7fU
#define MSK3 0xbffaffffU
#define MSK4 0xbffffff6U 
*/

/** These definitions are part of a 128-bit period certification vector.
#define PARITY1	0x00000001U
#define PARITY2	0x00000000U
#define PARITY3	0x00000000U
#define PARITY4	0xc98e126aU
*/

#ifndef SFMT_PARAMS19937_H
#define SFMT_PARAMS19937_H

#define POS1	122
#define SL1	18
#define SL2	1
#define SR1	11
#define SR2	1
#define MSK1	0xdfffffefU
#define MSK2	0xddfecb7fU
#define MSK3	0xbffaffffU
#define MSK4	0xbffffff6U
#define PARITY1	0x00000001U
#define PARITY2	0x00000000U
#define PARITY3	0x00000000U
#define PARITY4	0x13c9e684U


/* PARAMETERS FOR ALTIVEC */
#if defined(__APPLE__)	/* For OSX */
    #define ALTI_SL1	(vector unsigned int)(SL1, SL1, SL1, SL1)
    #define ALTI_SR1	(vector unsigned int)(SR1, SR1, SR1, SR1)
    #define ALTI_MSK	(vector unsigned int)(MSK1, MSK2, MSK3, MSK4)
    #define ALTI_MSK64 \
	(vector unsigned int)(MSK2, MSK1, MSK4, MSK3)
    #define ALTI_SL2_PERM \
	(vector unsigned char)(1,2,3,23,5,6,7,0,9,10,11,4,13,14,15,8)
    #define ALTI_SL2_PERM64 \
	(vector unsigned char)(1,2,3,4,5,6,7,31,9,10,11,12,13,14,15,0)
    #define ALTI_SR2_PERM \
	(vector unsigned char)(7,0,1,2,11,4,5,6,15,8,9,10,17,12,13,14)
    #define ALTI_SR2_PERM64 \
	(vector unsigned char)(15,0,1,2,3,4,5,6,17,8,9,10,11,12,13,14)
#else	/* For OTHER OSs(Linux?) */
    #define ALTI_SL1	{SL1, SL1, SL1, SL1}
    #define ALTI_SR1	{SR1, SR1, SR1, SR1}
    #define ALTI_MSK	{MSK1, MSK2, MSK3, MSK4}
    #define ALTI_MSK64	{MSK2, MSK1, MSK4, MSK3}
    #define ALTI_SL2_PERM	{1,2,3,23,5,6,7,0,9,10,11,4,13,14,15,8}
    #define ALTI_SL2_PERM64	{1,2,3,4,5,6,7,31,9,10,11,12,13,14,15,0}
    #define ALTI_SR2_PERM	{7,0,1,2,11,4,5,6,15,8,9,10,17,12,13,14}
    #define ALTI_SR2_PERM64	{15,0,1,2,3,4,5,6,17,8,9,10,11,12,13,14}
#endif	/* For OSX */
#define IDSTR	"SFMT-19937:122-18-1-11-1:dfffffef-ddfecb7f-bffaffff-bffffff6"

#endif /* SFMT_PARAMS19937_H */
#endif /* SFMT_PARAMS_H */

/*------------------------------------------------------
  128-bit SIMD data type for SSE2 or standard C
  ------------------------------------------------------*/
#if defined(__SSE2__)
  #include <emmintrin.h>

/** 128-bit data structure */
union W128_T {
    __m128i si;
    uint32_t u[4];
    double d[2];
};
/** 128-bit data type */
typedef union W128_T w128_t;

#else

/** 128-bit data structure */
struct W128_T {
    uint32_t u[4];
};
/** 128-bit data type */
typedef struct W128_T w128_t;

#endif

/* Container for random digits */

typedef union RAND64_t {
	uint8_t  s[8];
	uint32_t i[2];
	double d;
	float f[2];
	uint64_t l;
	signed long sl;
} rand64_t;

typedef union dW128_T {
    __m128i si;
    __m128d sd;
    uint64_t l[2];
    uint32_t i[4];
    double d[2];
} dw128_t;

/*--------------------------------------
  FILE GLOBAL VARIABLES
  internal state, index counter and flag 
  --------------------------------------*/
/** the 128-bit internal state array */
static w128_t sfmt[iN];
/** the 32bit integer pointer to the 128-bit internal state array */
static uint32_t *psfmt32 = &sfmt[0].u[0];
/** index counter to the 32-bit internal state array */
static int idx;
/** a flag: it is 0 if and only if the internal state is not yet
 * initialized. */
/** a parity check vector which certificate the period of 2^{MEXP} */
static uint32_t parity[4] = {PARITY1, PARITY2, PARITY3, PARITY4};

/*----------------
  STATIC FUNCTIONS
  ----------------*/
static void gen_rand_array(w128_t *array, int size);
inline static uint32_t func1(uint32_t x);
inline static uint32_t func2(uint32_t x);
static void period_certification(void);

static void mt_init(void);
static inline dw128_t wide_uniform(void);
static inline double uniform_double_PRN(void);
static inline unsigned long rand_long(unsigned long n);
static inline unsigned long rand_long64(void); 
void _report_PRN_total(void);
#if defined(__SSE2__)
/** 
 * @file  SFMT-sse2.h
 * @brief SIMD oriented Fast Mersenne Twister(SFMT) for Intel SSE2
 *
 * @author Mutsuo Saito (Hiroshima University)Fwd: Renting the Harvard Cabin
 * @author Makoto Matsumoto (Hiroshima University)
 *
 * @note We assume LITTLE ENDIAN in this file
 *
 * Copyright (C) 2006, 2007 Mutsuo Saito, Makoto Matsumoto and Hiroshima
 * University. All rights reserved.
 *
 * The new BSD License is applied to this software, see LICENSE.txt
 */

#ifndef SFMT_SSE2_H
#define SFMT_SSE2_H



PRE_ALWAYS static __m128i mm_recursion(__m128i *a, __m128i *b, __m128i c,
				   __m128i d, __m128i mask) ALWAYSINLINE;

/**
 * This function represents the recursion formula.
 * @param a a 128-bit part of the interal state array
 * @param b a 128-bit part of the interal state array
 * @param c a 128-bit part of the interal state array
 * @param d a 128-bit part of the interal state array
 * @param mask 128-bit mask
 * @return output
 */
PRE_ALWAYS static __m128i mm_recursion(__m128i *a, __m128i *b, 
				   __m128i c, __m128i d, __m128i mask) {
    __m128i v, x, y, z;
    
    x = _mm_load_si128(a);
    y = _mm_srli_epi32(*b, SR1);
    z = _mm_srli_si128(c, SR2);
    v = _mm_slli_epi32(d, SL1);
    z = _mm_xor_si128(z, x);
    z = _mm_xor_si128(z, v);
    x = _mm_slli_si128(x, SL2);
    y = _mm_and_si128(y, mask);
    z = _mm_xor_si128(z, x);
    z = _mm_xor_si128(z, y);
    return z;
}

/**
 * This function fills the user-specified array with pseudorandom
 * integers.
 *
 * @param array an 128-bit array to be filled by pseudorandom numbers.  
 * @param size number of 128-bit pesudorandom numbers to be generated.
 */
static void gen_rand_array(w128_t *array, int size) {
    int i, j;
    __m128i r, r1, r2, mask;
    mask = _mm_set_epi32(MSK4, MSK3, MSK2, MSK1);

    r1 = _mm_load_si128(&sfmt[iN - 2].si);
    r2 = _mm_load_si128(&sfmt[iN - 1].si);
    for (i = 0; i < iN - POS1; i++) {
	r = mm_recursion(&sfmt[i].si, &sfmt[i + POS1].si, r1, r2, mask);
	_mm_store_si128(&array[i].si, r);
	r1 = r2;
	r2 = r;
    }
    for (; i < iN; i++) {
	r = mm_recursion(&sfmt[i].si, &array[i + POS1 - iN].si, r1, r2, mask);
	_mm_store_si128(&array[i].si, r);
	r1 = r2;
	r2 = r;
    }
    /* main loop */
    for (; i < size - iN; i++) {
	r = mm_recursion(&array[i - iN].si, &array[i + POS1 - iN].si, r1, r2,
			 mask);
	_mm_store_si128(&array[i].si, r);
	r1 = r2;
	r2 = r;
    }
    for (j = 0; j < 2 * iN - size; j++) {
	r = _mm_load_si128(&array[j + size - iN].si);
	_mm_store_si128(&sfmt[j].si, r);
    }
    for (; i < size; i++) {
	r = mm_recursion(&array[i - iN].si, &array[i + POS1 - iN].si, r1, r2,
			 mask);
	_mm_store_si128(&array[i].si, r);
	_mm_store_si128(&sfmt[j++].si, r);
	r1 = r2;
	r2 = r;
    }
}

#endif

#endif


/**
 * This function represents a function used in the initialization
 * by init_by_array
 * @param x 32-bit integer
 * @return 32-bit integer
 */
static uint32_t func1(uint32_t x) {
    return (x ^ (x >> 27)) * (uint32_t)1664525UL;
}

/**
 * This function represents a function used in the initialization
 * by init_by_array
 * @param x 32-bit integer
 * @return 32-bit integer
 */
static uint32_t func2(uint32_t x) {
    return (x ^ (x >> 27)) * (uint32_t)1566083941UL;
}

/**
 * This function certificate the period of 2^{MEXP}
 */
static void period_certification(void) {
    int inner = 0;
    int i, j;
    uint32_t work;union dW128_T {
    __m128i si;
    __m128d sd;
    uint64_t u[2];
    uint32_t u32[4];
    double d[2];
};


    for (i = 0; i < 4; i++)
	inner ^= psfmt32[i] & parity[i];
    for (i = 16; i > 0; i >>= 1)
	inner ^= inner >> i;
    inner &= 1;
    /* check OK */
    if (inner == 1) {
	return;
    }
    /* check NG, and modification */
    for (i = 0; i < 4; i++) {
	work = 1;
};

	for (j = 0; j < 32; j++) {
	    if ((work & parity[i]) != 0) {
		psfmt32[i] ^= work;
		return;
	    }
	    work = work << 1;
	}
}

/*----------------
  PUBLIC FUNCTIONS
  ----------------*/

/**
 * This function initializes the internal state array,
 * with an array of 32-bit integers used as the seeds
 * @param init_key the array of 32-bit integers, used as a seed.
 * @param key_length the length of init_key.
 */
#define __EXP_SET__		0x3ff0000000000000

static w128_t iRandS[__cycle__], *iRend = &iRandS[__cycle__-1]; 
rand64_t *Rand;

static __m128d sse2_double_m_one;
static __m128i sse2_int_set;

#ifdef REPORT_PRNS
static long __n_cycles__ = 0;

static void _report_PRN_total(void) {
    printf("Used ~%ld 64-bit uniform PRNs.\n", 2*__cycle__*__n_cycles__ + Rand - (rand64_t *)iRandS); /* __cycle__ is in dimensions of 128-bit SIMD */    
}
#endif 

static void mt_init(void) {
        /* Avoid initializing twice */
    static int old = 0;
    if (old==1) return;
    old = 1;
        /* Use Process ID, Parent Process ID, and current time to seed the PRNG */
#ifdef _WIN32
    uint32_t init_key[] = {(int)getpid(), (int)time(NULL)}, key_length = 2;
#else
    uint32_t init_key[] = {(int)getpid(), (int)time(NULL), (int)getppid()}, key_length = 3;
#endif
        /* See http://www.math.sci.hiroshima-u.ac.jp/~%20m-mat/MT/SFMT/index.html 
         * for the remainder. */
#ifdef REPORT_PRNS
    atexit(_report_PRN_total);
#endif
    sse2_double_m_one = _mm_set_pd(-1.0, -1.0);
    sse2_int_set = _mm_set_epi64((__m64)__EXP_SET__, (__m64)__EXP_SET__);
    
    int i, j, count;
    uint32_t r;
    int lag;
    int mid;
    int size = iN * 4;
    if (size >= 623) {
	  lag = 11;
    } else if (size >= 68) {
	  lag = 7;
    } else if (size >= 39) {
	  lag = 5;
    } else {
	  lag = 3;
    }
    mid = (size - lag) / 2;

    memset(sfmt, 0x8b, sizeof(sfmt));
    if (key_length + 1 > N32) {
	count = key_length + 1;
    } else {
	count = N32;
    }
    r = func1(psfmt32[0] ^ psfmt32[mid] 
	      ^ psfmt32[N32 - 1]);
    psfmt32[mid] += r;

    r += key_length;
    psfmt32[mid + lag] += r;
    psfmt32[0] = r;

    count--;
    for (i = 1, j = 0; (j < count) && (j < key_length); j++) {
	r = func1(psfmt32[i] ^ psfmt32[(i + mid) % N32] 
		  ^ psfmt32[(i + N32 - 1) % N32]);
	psfmt32[(i + mid) % N32] += r;
	r += init_key[j] + i;
	psfmt32[(i + mid + lag) % N32] += r;
	psfmt32[i] = r;
	i = (i + 1) % N32;
    }
    for (; j < count; j++) {
	r = func1(psfmt32[i] ^ psfmt32[(i + mid) % N32] 
		  ^ psfmt32[(i + N32 - 1) % N32]);
	psfmt32[(i + mid) % N32] += r;
	r += i;
	psfmt32[(i + mid + lag) % N32] += r;
	psfmt32[i] = r;
	i = (i + 1) % N32;
    }
    for (j = 0; j < N32; j++) {
	r = func2(psfmt32[i] + psfmt32[(i + mid) % N32] 
		  + psfmt32[(i + N32 - 1) % N32]);
	psfmt32[(i + mid) % N32] ^= r;
	r -= i;
	psfmt32[(i + mid + lag) % N32] ^= r;
	sse2_double_m_one = _mm_set_pd(-1.0, -1.0);
	psfmt32[i] = r;
	i = (i + 1) % N32;
    }

    idx = N32;
    period_certification();
	gen_rand_array(iRandS,__cycle__); 
	Rand = (rand64_t *)iRandS;
}

#ifdef REPORT_PRNS
#define INCREMENT_N_CYCLES() (__n_cycles__++);
#else
#define INCREMENT_N_CYCLES() ;
#endif

#define MT_FLUSH() { if (Rand > (rand64_t *)iRend) { \
gen_rand_array(iRandS,__cycle__); \
Rand = (rand64_t *) iRandS; \
INCREMENT_N_CYCLES() \
}; } 

static inline dw128_t wide_uniform(void) {
  MT_FLUSH();
  dw128_t W;
  W.si = _mm_set_epi64x(Rand[0].l, Rand[1].l);	
  Rand+=2;
  W.si = _mm_or_si128(_mm_srli_epi64(W.si, 2), sse2_int_set); 
  W.sd = _mm_add_pd(W.sd, sse2_double_m_one);
  return W;
}

static inline double uniform_double_PRN(void) {
  MT_FLUSH();
  Rand->l = (Rand->l >> 2) | __EXP_SET__;
  return Rand++->d - 1;
}

static inline unsigned long rand_long(unsigned long n){
  MT_FLUSH();
  return Rand++->l % n;
}

static inline unsigned long rand_long64(void){
  MT_FLUSH();
  return Rand++->l;
}

