
// This marching cubes' algorithm is 
// It originally depends on other frameworks, I modified it a little so that reliance can be ignored
// On github:
// https://github.com/larsberg/ofxMarchingCubes

//
//  ofxMarchingCubes.h
//  ofxMarchingCubes

//http://paulbourke.net/geometry/polygonise/
//

/*
TODO::
 -get worldposition in grid
 -add iso value at world position
 */


#ifndef OFX_MARCHINGCUBES_H
#define OFX_MARCHINGCUBES_H


#include "mcTables.h"


#include<iostream>
#include<fstream>
#include<string>
#include<vector>

using namespace std;


struct Vec3f {
	float x;
	float y;
	float z;

public:
	Vec3f(float _x, float _y, float _z) :x(_x), y(_y), z(_z) {}
	Vec3f() :Vec3f(0.0f, 0.0f, 0.0f) {}



	void set(float _x, float _y, float _z) {
		x = _x, y = _y, z = _z;
	}

	Vec3f normalize() const {
		float l = sqrt(x*x + y*y + z*z);
		return Vec3f(x / l, y / l, z / l);
	}

	Vec3f operator+(const Vec3f &v) const{
		return Vec3f(x + v.x, y + v.y, z + v.z);
	}
	Vec3f operator-(const Vec3f &v) const {
		return Vec3f(x - v.x, y - v.y, z - v.z);
	}
	Vec3f crossed(const Vec3f &v) const {
		return Vec3f(
			y*v.z - z*v.y,
			z*v.x - x*v.z,
			x*v.y - y*v.x
		);
	}
	Vec3f operator*(const float &t)const {
		return Vec3f(x*t, y*t, z*t);
	}
	Vec3f operator/(const float &t)const {
		return Vec3f(x/t, y/t, z/t);
	}

};



class ofxMarchingCubes{
public:
	ofxMarchingCubes();
	~ofxMarchingCubes();
	
	void setMaxVertexCount( int _maxVertexCount = 10000000 );
	
	void setup( int resX=30, int resY=20, int resZ=30, int _maxVertexCount=10000000);
	void update(){		update( threshold );}
	void update(float _threshold);
	

	
	
	void flipNormals(){	flipNormalsValue *= -1;}
	void setResolution( int _x=10, int _y=10, int _z=10 );
	void polygonise( int i, int j, int k );
	void computeNormal( int i, int j, int k );
	void vertexInterp(float threshold, int i1, int j1, int k1, int i2, int j2, int k2, Vec3f& v, Vec3f& n);
	
	void setIsoValue( int x, int y, int z, float value);
	void addToIsoValue( int x, int y, int z, float value){
		getIsoValue(x,y,z) += value;
        bUpdateMesh = true;
	}
	
	bool getSmoothing(){	return bSmoothed;}
	void setSmoothing( bool _bSmooth ){		bSmoothed = _bSmooth;}
	
	
	void wipeIsoValues( float value=0.f);
	
	void clear();//deletes all the data. use whip
	
	
	void setGridPoints();
	

	
	inline float& getIsoValue( int x, int y, int z){
		return isoVals[ x*resY*resZ+ y*resZ + z ];
	}
	inline Vec3f& getGridPoint( int x, int y, int z){
		return gridPoints[ x*resY*resZ+ y*resZ + z ];
	}
	inline Vec3f& getNormalVal( int x, int y, int z){
		return normalVals[ x*resY*resZ+ y*resZ + z ];
	}
	inline unsigned int& getGridPointComputed( int x, int y, int z){
		return gridPointComputed[ x*resY*resZ+ y*resZ + z ];
	}
	
	void exportObj( string fileName );
	
public:

	int	resX, resY, resZ;
	int resXm1, resYm1, resZm1;
	float flipNormalsValue;
	Vec3f cellDim;
	vector<float> isoVals;
	vector<Vec3f> gridPoints;
	vector<Vec3f> normalVals;
	vector<unsigned int> gridPointComputed;

	
	vector< Vec3f > vertices;
	vector< Vec3f > normals;

	int vertexCount, maxVertexCount;
	
	Vec3f vertList[12], normList[12];
	
	float threshold;
	bool bSmoothed, beenWarned;

	

	bool bUpdateMesh;
};




#endif