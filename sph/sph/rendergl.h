#ifndef RENDER_GL
#define RENDER_GL

// OpenGL


#include"../lib/OpenGL/include/GL/glew.h"
#include"../lib/OpenGL/include/GL/freeglut.h"

#include<cuda_runtime.h>

#include"shader.h"

#ifndef UINT
#define UINT
typedef unsigned int uint;
#endif




class RenderGL{
protected:
	GLuint program;
	GLuint vbo;
	float3 *hPos;
	uint numParticles;
	float h;	// Radius
	Shader shader;


protected:
	GLuint compileProgram(const char *srcVert, const char *srcFra);

public:
	// Projection parameters
	int winHeight;
	float fov;
public:
	

	void renderParticles();

	void initRender(float3 *hPos, uint numParticles, float h, float fov=60.0f);

	void createVBO(uint size);

	void updateVBO();

	void setVertexBuffer(GLuint vbo, uint numParticles);

	void resetWinHeight(int winHeight);

	

};



#endif