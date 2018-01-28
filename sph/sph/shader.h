#ifndef SHADER_H
#define SHADER_H
#include<string>

class Shader {
protected:
	std::string srcVert;
	std::string srcFrag;
	float pointRadius;

protected:
	void genVert();
	void genFrag();
public:
	Shader();

	const char *getSrcVert();
	const char *getSrcFrag();

};

#endif