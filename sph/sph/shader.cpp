#include"shader.h"

#define TOSTRING(str) #str

void Shader::genVert() {
	srcVert = TOSTRING(
		uniform float pointRadius;  // point size in world space
		uniform float pointScale; // Scale particle's real size to screen space
	
	void main(){
		// Particle's position in eye space
		vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
		// Scale the viewed radius according to distance
		float dist = length(posEye);
		gl_PointSize = pointRadius * pointScale / dist;

		gl_TexCoord[0] = gl_MultiTexCoord0;
		gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

		gl_FrontColor = gl_Color;
	}
	);

}


void Shader::genFrag() {
	srcFrag = TOSTRING(
	void main(){

		const vec3 lightDir = vec3(0.577, 0.577, 0.577);

		// calculate normal from texture coordinates
		vec3 N;
		N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
		float mag = dot(N.xy, N.xy);

		if (mag > 1.0) discard;   // kill pixels outside circle

		N.z = sqrt(1.0 - mag);

		// calculate lighting
		float diffuse = max(0.0, dot(lightDir, N));

		gl_FragColor = gl_Color * diffuse;
	}



	);
}


const char *Shader::getSrcVert() {
	return srcVert.c_str();
}

const char *Shader::getSrcFrag() {
	return srcFrag.c_str();
}

Shader::Shader() {
	genVert();
	genFrag();
}
