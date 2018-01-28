#ifndef TIMER_H
#define TIMER_H
#include<Windows.h>

#ifndef UINT
#define UINT
typedef unsigned int uint;
#endif

class Timer {
protected:
	// How many frames have been simulated 
	uint nframes;

	// Record time and calculate FPS per [interval] frames
	uint interval;

	// Frame per second
	float FPS;


	// Start time
	DWORD startTime;

	// End time
	DWORD endTime;

	// Present time
	DWORD nowTime;

	// Time before [interval] frames than nowTime
	DWORD oldTime;

public:
	Timer(uint interval);
	Timer();

	void startTiming();
	float calcFPS();
	void endTiming();




};



#endif
