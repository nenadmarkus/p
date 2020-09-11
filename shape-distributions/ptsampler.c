#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/*
	multiply with carry PRNG
*/

uint32_t mwcrand_r(uint64_t* state)
{
	uint32_t* m;

	//
	m = (uint32_t*)state;

	// bad state?
	if(m[0] == 0)
		m[0] = 0xAAAA;

	if(m[1] == 0)
		m[1] = 0xBBBB;

	// mutate state
	m[0] = 36969 * (m[0] & 65535) + (m[0] >> 16);
	m[1] = 18000 * (m[1] & 65535) + (m[1] >> 16);

	// output
	return (m[0] << 16) + m[1];
}

uint64_t prngglobal = 0x12345678000fffffLL;

void smwcrand(uint32_t seed)
{
	prngglobal = 0x12345678000fffffLL*seed;
}

uint32_t mwcrand()
{
	return mwcrand_r(&prngglobal);
}

/*
	
*/

float* A;
float* P;

int* I;

int areacmp(const void* i1, const void* i2)
{
	if ( A[*(int*)i1] <  A[*(int*)i2] )
		return -1;
	if ( A[*(int*)i1] == A[*(int*)i2] )
		return 0;
	if ( A[*(int*)i1] >  A[*(int*)i2] )
		return +1;
}

void sample_surface_points(float* samples, int nsamples, float* V, int nV, int* F, int nF)
{
	int i;
	float tA;
	//
	A = (float*)malloc(nF*sizeof(float));
	P = (float*)malloc((nF+1)*sizeof(float));

	I = (int*)malloc(nF*sizeof(int));

	//
	tA = 0.0f;

	for(i=0; i<nF; ++i)
	{
		/*
			 implements Heron's formula
		*/

		float a2, b2, c2;

		//
		a2 = (V[3*F[3*i+0]+0]-V[3*F[3*i+1]+0])*(V[3*F[3*i+0]+0]-V[3*F[3*i+1]+0])+(V[3*F[3*i+0]+1]-V[3*F[3*i+1]+1])*(V[3*F[3*i+0]+1]-V[3*F[3*i+1]+1])+(V[3*F[3*i+0]+2]-V[3*F[3*i+1]+2])*(V[3*F[3*i+0]+2]-V[3*F[3*i+1]+2]);
		b2 = (V[3*F[3*i+0]+0]-V[3*F[3*i+2]+0])*(V[3*F[3*i+0]+0]-V[3*F[3*i+2]+0])+(V[3*F[3*i+0]+1]-V[3*F[3*i+2]+1])*(V[3*F[3*i+0]+1]-V[3*F[3*i+2]+1])+(V[3*F[3*i+0]+2]-V[3*F[3*i+2]+2])*(V[3*F[3*i+0]+2]-V[3*F[3*i+2]+2]);
		c2 = (V[3*F[3*i+1]+0]-V[3*F[3*i+2]+0])*(V[3*F[3*i+1]+0]-V[3*F[3*i+2]+0])+(V[3*F[3*i+1]+1]-V[3*F[3*i+2]+1])*(V[3*F[3*i+1]+1]-V[3*F[3*i+2]+1])+(V[3*F[3*i+1]+2]-V[3*F[3*i+2]+2])*(V[3*F[3*i+1]+2]-V[3*F[3*i+2]+2]);

		//
		if(4*a2*b2 - (a2+b2-c2)*(a2+b2-c2)>=0)
			A[i] = 0.25f*sqrt(4*a2*b2 - (a2+b2-c2)*(a2+b2-c2));
		else
			A[i] = 0; // ?????

		I[i] = i;

		tA = tA + A[i];
	}

	//
	qsort(I, nF, sizeof(int), areacmp);

	//
	P[0] = 0.0f;
	for(i=1; i<nF; ++i)
		P[i] = P[i-1] + A[I[i-1]]/tA;
	P[nF] = 1.0f;

	//
	for(i=0; i<nsamples; ++i)
	{
		int t;
		float p, r1, r2, v[3];

		// pick a random triangle
		p = (mwcrand()%10000)/9999.0f;

		for(t=0; t<nF; ++t) // binary search would be more efficient
			if(P[t]<=p && p<P[t+1])
				break;

		if(t >= nF)
			t = nF - 1;

		t = I[t];

		// sample a random point within the picked triangle
		r1 = (mwcrand()%10000)/9999.0f;
		r2 = (mwcrand()%10000)/9999.0f;

		r1 = sqrt(r1);

		v[0] = (1-r1)*V[3*F[3*t+0]+0] + r1*(1-r2)*V[3*F[3*t+1]+0] + r1*r2*V[3*F[3*t+2]+0];
		v[1] = (1-r1)*V[3*F[3*t+0]+1] + r1*(1-r2)*V[3*F[3*t+1]+1] + r1*r2*V[3*F[3*t+2]+1];
		v[2] = (1-r1)*V[3*F[3*t+0]+2] + r1*(1-r2)*V[3*F[3*t+1]+2] + r1*r2*V[3*F[3*t+2]+2];

		/*
			computes the normal
		*/

		/*
		float n[3], l;
		int i1 = F[3*t+0], i2 = F[3*t+1], i3 = F[3*t+2];

		// iuter product
		n[0] = ((V[3*i2+1]-V[3*i1+1])*(V[3*i3+2]-V[3*i1+2])) - ((V[3*i2+2]-V[3*i1+2])*(V[3*i3+1]-V[3*i1+1]));
		n[1] = ((V[3*i2+2]-V[3*i1+2])*(V[3*i3+0]-V[3*i1+0])) - ((V[3*i2+0]-V[3*i1+0])*(V[3*i3+2]-V[3*i1+2]));
		n[2] = ((V[3*i2+0]-V[3*i1+0])*(V[3*i3+1]-V[3*i1+1])) - ((V[3*i2+1]-V[3*i1+1])*(V[3*i3+0]-V[3*i1+0]));

		// normalization
		l = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);

		if(l > 0.0f)
		{
			n[0] = n[0]/l;
			n[1] = n[1]/l;
			n[2] = n[2]/l;
		}
		else
		{
			n[0] = 0.0f;
			n[1] = 0.0f;
			n[2] = 0.0f;
		}
		*/

		/*

		*/

		samples[3*i+0] = v[0];
		samples[3*i+1] = v[1];
		samples[3*i+2] = v[2];
	}

	//
	free(A);
	free(P);
	free(I);
}