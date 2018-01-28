#include"timer.h"


Timer::Timer(uint interval) :interval(interval) {}
Timer::Timer() :Timer(5){}

void Timer::startTiming() {
	startTime = GetTickCount();
	oldTime = nowTime = startTime;
	nframes = 0;
}
float Timer::calcFPS() {
	if (!(nframes = (nframes + 1) % interval)) {
		nowTime = GetTickCount();
		FPS = interval*1000.f / (nowTime - oldTime);
		oldTime = nowTime;
	}
	return FPS;
}
void Timer::endTiming() {
	endTime = GetTickCount();
}
