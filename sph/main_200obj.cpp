#include<Windows.h>
#include"particlesystem.h"
#include"rendergl.h"
#include"ofxMarchingCubes.h"
#include"timer.h"
#include<iostream>
#include<thread>



int width = 800, height = 600;
int numObj = 0;


// view params
int ox, oy;
int buttonState = 0;
float camera_trans[] = { 0, 0, -1.5f };
float camera_rot[] = { 0, 0, 0 };
float camera_trans_lag[] = { 0, 0, -1.5f };
float camera_rot_lag[] = { 0, 0, 0 };
float modelView[16];
const float inertia = 0.1f;


bool displaySliders = false;



Timer timer;
ParticleSystem sys;	
RenderGL renderer;
ofxMarchingCubes mc;



void initCuda(){

	int numDevice;

	cudaGetDeviceCount(&numDevice);
	if (numDevice == 0){
		printf("No device found! Exiting...");
		exit(0);
	}

	int i;
	for (i = 0; i < numDevice; ++i) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess && prop.major >= 1) {
			break;
		}
	}

	if (i == numDevice) {
		printf("No device supports CUDA! Exiting...");
		exit(0);
	}

	cudaSetDevice(i);

	printf("CUDA initialized.\n");
}



void initGL(int *argc, char **argv) {
	
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("SPH Fluid Simulation");
	glEnable(GL_DEPTH_TEST);

	glewInit();
}

void initSimulation() {
	sys.initSimulation();
	timer.startTiming();

}


void display() {
	
	if (!sys.bPaused) 
		sys.update();


	// render
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// view transform
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	for (int c = 0; c < 3; ++c)
	{
		camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
		camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
	}

	glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
	glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
	glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

	renderer.resetWinHeight(height);
	renderer.updateVBO();
	renderer.renderParticles();


	glutSwapBuffers();

	
	static char title[40];
	sprintf(title, "SPH Fluid Simulation  FPS = %3.1f  numParticles = %d", timer.calcFPS(), sys.getNum());
	glutSetWindowTitle(title);
}

void idle() {
	glutPostRedisplay();
}

void key(GLubyte key, int x, int y){
	static char objName[30];
	switch (key) {
	case ' ':
		sys.bPaused = !sys.bPaused;
		break;
	case 'o':
		if (sys.bPaused) {
			printf("Generating model...\n");

			sys.setIsoValue(mc);
			mc.bUpdateMesh = true;
			mc.update(1.0f);	// Metaball field threshold
			sprintf(objName, "%03d", numObj++);

			mc.exportObj(objName);

		}
		else {
			printf("You can only export model when paused!\n");
		}
		break;
	}



}

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		buttonState = 1;
	}
	else if (state == GLUT_UP)
	{
		buttonState = 0;
	}

	ox = x; oy = y;

	glutPostRedisplay();
}


void reshape(int width, int height) {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (double)width / (double)height, 0.1, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, width, height);

}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	

	if (buttonState == 3){

			// left+middle = zoom
			camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
	}
	else if (buttonState & 2){
	
			// middle = translate
			camera_trans[0] += dx / 100.0f;
			camera_trans[1] -= dy / 100.0f;
	}
	else if (buttonState & 1){
			// left = rotate
			camera_rot[0] += dy / 5.0f;
			camera_rot[1] += dx / 5.0f;
	}

		

	ox = x;
	oy = y;
	glutPostRedisplay();
}

void bindFunctionGL() {
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(key);
	glutIdleFunc(idle);
}



int main(int argc, char **argv) {
	
	initCuda();

	printf("Welcome to SPH simulation!\n");
	printf("Initializing system and loading scene...\n");
	sys.param.setDefault();
	sys.param.init();



	printf("You can input space button to pause/continue\n");
	printf("When paused, you can input 'o' button to export the curret frame as .obj model\n");
	printf("Initializing rendering...\n");


	sys.init();
	sys.setupScene();
	mc.setup(sys.param.dimRes.x + 1, sys.param.dimRes.y + 1, sys.param.dimRes.z + 1);
	mc.setSmoothing(true);

	//initGL(&argc, argv);
	//bindFunctionGL();

	//renderer.initRender(sys.getPos(), sys.getNum(), sys.getRadius());
	//renderer.createVBO(sys.getNum());

	for (int i = 0;i < 300;++i) sys.update();
	
	static char objName[30];
	int count = 0;
	for (int i = 300;i < 900;++i) {
		

		sys.update();
		if (count == 0) {
			printf("Generating model...\n");

		
			sys.setIsoValue(mc);
			mc.bUpdateMesh = true;
			mc.update(1.0f);	// Metaball field threshold
			sprintf(objName, "%03d", numObj++);

			mc.exportObj(objName);
		}
		count = (count + 1) % 3;
	}



	//glutMainLoop();



		


}