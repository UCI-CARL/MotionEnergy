#ifndef _MOTION_ENERGY_H_
#define _MOTION_ENERGY_H_

#include <vector_types.h>			// dim3


class MotionEnergy {
public:
	MotionEnergy(int nrX, int nrY, int nrC);
	~MotionEnergy();
	
	void setNumChannels(int nrC);
	
	void calcV1complex(unsigned char* stim, float* V1comp, double speed=1.0, bool GPUpointers=true);

	double getScaleV1Linear() { return scaleV1Linear_; }
	double getScaleV1FullWaveRect() { return scaleV1FullWaveRect_; }
	double getScaleV1Blur() { return scaleV1Blur_; }
	double getScaleV1NormPopK() { return scaleV1NormPopK_; }
	double getScaleV1NormStrength() { return scaleV1NormStrength_; }
	double getScaleV1Complex() { return scaleV1Complex_; }
	double getScaleV1ComplexFiring() { return scaleV1ComplexFiring_; }
	double getScaleV1C50() { return scaleV1C50_; }

	void setScaleV1Linear(double s) { scaleV1Linear_ = s; }
	void setScaleV1FullWaveRect(double s) { scaleV1FullWaveRect_ = s; }
	void setScaleV1Blur(double s) { scaleV1Blur_ = s; }
	void setScaleV1NormPopK(double s) { scaleV1NormPopK_ = s; }
	void setScaleV1NormStrength(double s) { scaleV1NormStrength_ = s; }
	void setScaleV1Complex(double s) { scaleV1Complex_ = s; }
	void setScaleV1ComplexFiring(double s) { scaleV1ComplexFiring_ = s; }
	void setScaleV1C50(double s) { scaleV1C50_ = s; }


private:
	void initME();
	void initParams(); // shPars.m
	void accumDiffStims(float *d_resp_tmp, float* diffV1GausBuf, dim3 _sizes, int orderX, int orderY, int orderT);
	void conv2D(float* idata, float* odata, dim3 _sizes, const float* filt, int filtlen);
	void conv3D(float* idata, float* odata, dim3 _sizes, const float* filt, int filtlen);
	float* diff(float* idata, dim3 _sizes, int order, int dim);

	void loadInput(unsigned char* stim);
	void calcV1linear();
	void calcV1rect();
	void calcV1blur();
	void calcV1normalize();
	void calcV1direction(double speed);
	
	int nrX_;
	int nrY_;
	int nrC_;

	int nrScales_;

	int min_nrX_;
	int min_nrY_;

	// from shPars.m
	double scaleV1Linear_;
	double scaleV1FullWaveRect_;
	double scaleV1Blur_;
	double scaleV1NormPopK_;
	double scaleV1NormStrength_;
	double scaleV1Complex_;
	double scaleV1C50_;
	double scaleV1ComplexFiring_;

	int stimChannels;

	float* scalingFilt;
	float* v1Gaus;
	float* complexV1Filt;
	float* normV1filt;
	float* diff1filt;
	float* diff2filt;
	float* diff3filt;

	float* d_resp_; // the returned V1ME filter responses
	float* d_respV1c; // the returned V1 complex cell responses
	float* d_stimBuf;
	float* d_scalingStimBuf; // the temporary matrix that will be filtered once for each scale...
	float* d_v1GausBuf;
	float* d_v1GausBuf2;
	float* d_diffV1GausBuf;
	float* diffV1GausBufT;

	unsigned char* d_stim; // the video input
	float* d_pop;
};

#endif