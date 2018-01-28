#include"rendergl.h"



GLuint RenderGL::compileProgram(const char *srcVert, const char *srcFrag) {

	GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(vertShader, 1, &srcVert, 0);
	glShaderSource(fragShader, 1, &srcFrag, 0);

	glCompileShader(vertShader);
	glCompileShader(fragShader);

	GLuint program = glCreateProgram();

	glAttachShader(program, vertShader);
	glAttachShader(program, fragShader);

	glLinkProgram(program);


	GLint success = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success)
	{
		char temp[256];
		glGetProgramInfoLog(program, 256, 0, temp);
		printf("Failed to link program:\n%s\n", temp);
		glDeleteProgram(program);
		program = 0;
	}

	return program;

}

void RenderGL::renderParticles() {
	static const float pi = 3.1415926535f;

	glColor3f(1.0f, 1.0f, 1.0f);
	glutWireCube(1.0);

	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);


	glUseProgram(program);

	glUniform1f(glGetUniformLocation(program, "pointScale"), winHeight / tanf(fov*0.5f*pi / 180.0f));
	glUniform1f(glGetUniformLocation(program, "pointRadius"), h);

	glColor3f(0.5f, 0.8f, 1.0f);

	if (vbo) {

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);

		glDrawArrays(GL_POINTS, 0, numParticles);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	else {
		glBegin(GL_POINTS);
		for (uint i = 0; i < numParticles; ++i) {
			glVertex3f(hPos[i].x, hPos[i].y, hPos[i].z);
		}
		glEnd();
	}

	glUseProgram(0);
	glDisable(GL_POINT_SPRITE_ARB);

}

void RenderGL::initRender(float3 *hPos, uint numParticles, float h, float fov) {
	this->hPos = hPos;
	this->numParticles = numParticles;
	this->h = h*0.5f;
	this->fov = fov;
	this->vbo = 0;

	program = compileProgram(shader.getSrcVert(), shader.getSrcFrag());
}

void RenderGL::createVBO(uint size) {

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RenderGL::updateVBO() {
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, numParticles*sizeof(float3), (void *)hPos, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RenderGL::setVertexBuffer(GLuint vbo, uint numParticles)
{
	this->vbo = vbo;
	this->numParticles = numParticles;
}

void RenderGL::resetWinHeight(int winHeight) {
	this->winHeight = winHeight;
}