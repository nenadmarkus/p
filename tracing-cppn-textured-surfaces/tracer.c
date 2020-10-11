#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))
#define ABS(x) ((x)<0?(-(x)):(x))

/*

*/

#define ABS(x) ((x)<0?(-(x)):(x))
#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))

float sdb_sphere(float x, float y, float z, float r)
{
	return sqrt(x*x + y*y + z*z) - r;
}

float sdb_cuboid(float x, float y, float z, float a, float b, float c)
{
	float d[3] = {
		ABS(x) - a/2.0,
		ABS(y) - b/2.0,
		ABS(z) - c/2.0
	};
	return sqrt(MAX(d[0], 0.0)*MAX(d[0], 0.0) + MAX(d[1], 0.0)*MAX(d[1], 0.0) + MAX(d[2], 0.0)*MAX(d[2], 0.0)) +
		MIN(MAX(d[0], MAX(d[1], d[2])), 0.0);
}

float sdb_cone(float x, float y, float z, float r, float h)
{
	float nr = -h/sqrt(r*r + h*h);
	float nz = -r/sqrt(r*r + h*h);
	float off = -nr*r;

	r = sqrt(x*x + y*y);

	return MAX(MAX(-r, -z), -nr*r - nz*z - off);
}


float sdb_cylinder(float x, float y, float z, float r, float h)
{
	return MAX(sqrt(x*x + y*y)-r, MAX(z-h/2.0, -z-h/2.0));
}

float sdb_torus(float x, float y, float z, float R, float r)
{
	float t = R - sqrt(x*x + y*y);
	return sqrt(t*t + z*z) - r;
}

float sdb_octahedron(float x, float y, float z, float s)
{
	x = ABS(x);
	y = ABS(y);
	z = ABS(z);
	return (x + y + z - s)*0.57735027;
}

float sdb_capsule(float x, float y, float z, float xa, float ya, float za, float xb, float yb, float zb, float r)
{
	float pa[] = {x - xa, y - ya, z - za};
	float ba[] = {xb - xa, yb - ya, zb - za};
	float h = (pa[0]*ba[0] + pa[1]*ba[1] + pa[2]*ba[2])/(ba[0]*ba[0] + ba[1]*ba[1] + ba[2]*ba[2]);
	h = MAX(0.0, MIN(1.0, h));
	float d[] = {pa[0] - h*ba[0], pa[1] - h*ba[1], pa[2] - h*ba[2]};
	return sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]) - r;
}

float sde_primitives(float _x, float _y, float _z)
{
	// apply some random rotation matrix
	float x = -0.96960723*_x + -0.04224664*_y + 0.2409918*_z;
	float y = 0.0951771*_x + -0.97252597*_y + 0.21244895*_z;
	float z = 0.22539553*_x + 0.22892894*_y + 0.94698912*_z;

	int i = 1;
	const float radius = 0.3;
	float a = 0.0;

	float d1 = sdb_sphere(x, y, z, 0.1);

	a = (i-1)*2.0*3.14156/(7-1); ++i;
	float d2 = sdb_cuboid(x-radius*sin(a), y-radius*cos(a), z, 0.15, 0.15, 0.15);
	a = (i-1)*2.0*3.14156/(7-1); ++i;
	float d3 = sdb_cone(x-radius*sin(a), y-radius*cos(a), z, 0.1, 0.2);
	a = (i-1)*2.0*3.14156/(7-1); ++i;
	float d4 = sdb_cylinder(x-radius*sin(a), y-radius*cos(a), z, 0.1, 0.2);
	a = (i-1)*2.0*3.14156/(7-1); ++i;
	float d5 = sdb_torus(x-radius*sin(a), y-radius*cos(a), z, 0.08, 0.04);
	a = (i-1)*2.0*3.14156/(7-1); ++i;
	float d6 = sdb_octahedron(x-radius*sin(a), y-radius*cos(a), z, 0.15);
	a = (i-1)*2.0*3.14156/(7-1); ++i;
	float d7 = sdb_capsule(x-radius*sin(a), y-radius*cos(a), z, 0, 0, -0.075, 0, 0, +0.075, 0.075);

	// union of all primitives
	return MIN(MIN(MIN(MIN(MIN(MIN(d1, d2), d3), d4), d5), d6), d7);
}

float sde_sierpinski_tetrahedron(float x, float y, float z)
{
	int n = 0;
	const int scale = 2.0;

	while( n < 5 )
	{
		float tmp;
		if(x + y < 0) {tmp=x; x=-y; y=-tmp;}
		if(x + z < 0) {tmp=x; x=-z; z=-tmp;}
		if(y + z < 0) {tmp=y; y=-z; z=-tmp;}
		x = scale*x - 0.4*(scale-1.0);
		y = scale*y - 0.4*(scale-1.0);
		z = scale*z - 0.4*(scale-1.0);
		++n;
	}

	return sqrtf(x*x + y*y + z*z)*powf(scale, -n) - 0.0075f;
}

float sde_knot(float x, float y, float z)
{
	//
	float scale = 6;

	x *= scale;
	y *= scale;
	z *= scale;

	//
	float phi = atan2(y, x);
	float r1 = sinf(1.5 * phi) + 1.5;
	float z1 = cosf(1.5 * phi);
	float r2 = sinf(1.5 * phi + 3.14156) + 1.5;
	float z2 = cosf(1.5 * phi + 3.14156);
	//
	float x1=r1*cosf(phi), y1=r1*sinf(phi);
	float x2=r2*cosf(phi), y2=r2*sinf(phi);
	//
	float r = sqrtf(x*x + y*y + z*z);
	float f = (1 + r)/1.25;
	if (r>4.0) f = 1.0;
	return f*(MIN(sqrtf((x-x1)*(x-x1)+(y-y1)*(y-y1)+(z-z1)*(z-z1)), sqrtf((x-x2)*(x-x2)+(y-y2)*(y-y2)+(z-z2)*(z-z2))) - 0.3)/4.0/scale;
}

float sde_scene(float x, float y, float z)
{
	if ( x*x + y*y + z*z > 4 )
		return sqrtf(x*x + y*y + z*z) - sqrtf(3);
	else
		//return 2*sde_primitives(x/2, y/2, z/2);
		//return sde_sierpinski_tetrahedron(x, y, z);
		return 2*sde_knot(x/2, y/2, z/2);
}

/*
	
*/

void get_normal_at(
	float (*sdef)(float ptx, float pty, float ptz),
	float x, float y, float z,
	float* nx, float* ny, float* nz,
	float eps
)
{
	//
	float n[3] = {
		sdef(x+eps, y, z) - sdef(x-eps, y, z),
		sdef(x, y+eps, z) - sdef(x, y-eps, z),
		sdef(x, y, z+eps) - sdef(x, y, z-eps)
	};
	//
	*nx = n[0]/sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
	*ny = n[1]/sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
	*nz = n[2]/sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
}

float trace_ray(
	float origin[], float direction[], 
	float (*sdef)(float x, float y, float z),
	float* t,
	float eps
)
{
	int i;
	float abshmin;

	*t = 0.0f;
	abshmin = 1e6f;

	for(i=0; i<1024; ++i)
	{
		float point[3] = {
			origin[0] + *t*direction[0],
			origin[1] + *t*direction[1],
			origin[2] + *t*direction[2]
		};

		float h = sdef(point[0], point[1], point[2]);

		if (ABS(h) < abshmin)
			abshmin = ABS(h);

		if (ABS(h)<eps)
			return abshmin;

		if (ABS(*t)>1/eps)
			return abshmin;

		*t += h;
	}

	return abshmin;
}

void compute(int nrows, int ncols, float intfield[], float mindistfield[], float normals[], float eps)
{
	float maxdist = 100.0f; // distance above which we do not render
	float mindist = 0.001f; // tolerance, minimum distance

	// camera params
	float focallength = 3.0f;
	float screenhw[] = {2.0f, 2.0f};

	// do the rendering
	int r;
	for(r=0; r<nrows; ++r)
	{
		int c;
		for(c=0; c<ncols; ++c)
		{
			// compute ray
			float origin[3] = {
				+screenhw[0]/2.0f - r*screenhw[0]/(nrows-1),
				-screenhw[1]/2.0f + c*screenhw[1]/(ncols-1),
				focallength
			};

			float direction[] = {
				origin[0]/sqrtf(origin[0]*origin[0] + origin[1]*origin[1] + origin[2]*origin[2]),
				origin[1]/sqrtf(origin[0]*origin[0] + origin[1]*origin[1] + origin[2]*origin[2]),
				origin[2]/sqrtf(origin[0]*origin[0] + origin[1]*origin[1] + origin[2]*origin[2])
			};

			origin[2] = -2;

			// do tracing and shading
			float t;
			mindistfield[r*ncols + c] = trace_ray(origin, direction, sde_scene, &t, eps);
			intfield[3*r*ncols + 3*c + 0] = origin[0] + t*direction[0];
			intfield[3*r*ncols + 3*c + 1] = origin[1] + t*direction[1];
			intfield[3*r*ncols + 3*c + 2] = origin[2] + t*direction[2];
			if (mindistfield[r*ncols + c] <= eps)
				get_normal_at(
					sde_scene,
					intfield[3*r*ncols + 3*c + 0], intfield[3*r*ncols + 3*c + 1], intfield[3*r*ncols + 3*c + 2],
					&normals[3*r*ncols + 3*c + 0], &normals[3*r*ncols + 3*c + 1], &normals[3*r*ncols + 3*c + 2],
					eps
				);
		}
	}
}
