#include <motion_energy.h>

#include <stdio.h>		// printf
#include <stdlib.h>     // srand, rand
#include <time.h>       // time

int main() {
	printf("main()\n");

	srand(time(NULL));

	const int numPx = 8;		// numPx x numPx pixels stimulus
	const int numChannels = 1;	// 1==gray, 3==RGB

	const int numDir = 8;		// number of directions V1 neurons are selective to (hard-coded for now)
	const double speed = 1.5;	// preferred speed of V1 neurons

	MotionEnergy me(numPx, numPx, numChannels);

	float* v1Comp = (float*)malloc(numPx*numPx*numDir*sizeof(float));
	unsigned char* stim = (unsigned char*)malloc(numPx*numPx*sizeof(unsigned char));

	for (int i=0; i<10; i++) {
		memset(stim, 0, numPx*numPx*sizeof(unsigned char));
		// create random stim
		for (int x=0; x<numPx; x++) {
			for (int y=0; y<numPx; y++) {
				stim[x*numPx+y] = rand() % 256;
			}
		}

		// calculate V1 complex cell response
		me.calcV1complex(stim, v1Comp, speed, false);

		// calculate mean firing rate
		float sumActivity = 0.0f;
		for (int x=0; x<numPx*numPx*numDir; x++) {
			sumActivity += v1Comp[x];
		}
		printf("%d: mean activity = %f\n", i, sumActivity / (numPx*numPx*numDir));
	}

	free(stim);
	free(v1Comp);
	return 1;
}