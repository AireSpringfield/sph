 //
//  ofxMarchingCubes.cpp
//  ofxMarchingCubes
//

#include "ofxMarchingCubes.h"

#define min(x,y) x<y?x:y
#define max(x,y) x>y?x:y



ofxMarchingCubes::ofxMarchingCubes(){
	threshold = .5;
	bSmoothed = true;
	flipNormalsValue = -1;
	
};
ofxMarchingCubes::~ofxMarchingCubes(){};

void ofxMarchingCubes::setMaxVertexCount( int _maxVertexCount ){
	maxVertexCount = _maxVertexCount;
	beenWarned = false;
}


void ofxMarchingCubes::setup( int resX, int resY, int resZ, int _maxVertexCount){
	
	clear();

	
	
	float boxVerts[] = {-.5, -.5, -.5, .5, -.5, -.5, -.5, .5, -.5, .5, .5, -.5, -.5, -.5, .5, .5, -.5, .5, -.5, .5, .5, .5, .5, .5, -.5, -.5, .5, -.5, -.5, -.5, -.5, .5, .5, -.5, .5, -.5, .5, -.5, .5, .5, -.5, -.5, .5, .5, .5, .5, .5, -.5, -.5, .5, -.5, -.5, -.5, -.5, -.5, .5, .5, -.5, -.5, .5, .5, .5, -.5, .5, -.5, -.5, .5, .5, .5, .5, -.5, .5,};

	
	setResolution( resX, resY, resZ );
	setMaxVertexCount( _maxVertexCount );

	vertexCount = 0;
	
	vertices.resize( maxVertexCount );
	normals.resize( maxVertexCount );
	

}

void ofxMarchingCubes::update(float _threshold){
	
	if( bUpdateMesh || threshold != _threshold ){
			
		threshold = _threshold;
		
		std::fill( normalVals.begin(), normalVals.end(), Vec3f());
		std::fill( gridPointComputed.begin(), gridPointComputed.end(), 0 );
		
		vertexCount = 0;
		for(int x=0; x<resX; x++){
			for(int y=0; y<resY; y++){
				for(int z=0; z<resZ; z++){
					polygonise( x,y,z );
				}
			}
		}
		

		
		bUpdateMesh = false;
	}
}

void ofxMarchingCubes::polygonise( int i, int j, int k ){
	
	if( vertexCount+3 < maxVertexCount ){
		bUpdateMesh = true;
		/*
		 Determine the index into the edge table which
		 tells us which vertices are inside of the surface
		 */
		int cubeindex = 0;
		int i1 = min(i+1, resXm1), j1 = min(j+1, resYm1), k1 = min(k+1, resZm1);
		cubeindex |= getIsoValue(i,j,k) > threshold ?   1 : 0;
		cubeindex |= getIsoValue(i1,j,k) > threshold ?   2 : 0;
		cubeindex |= getIsoValue(i1,j1,k) > threshold ?   4 : 0;
		cubeindex |= getIsoValue(i,j1,k) > threshold ?   8 : 0;
		cubeindex |= getIsoValue(i,j,k1) > threshold ?  16 : 0;
		cubeindex |= getIsoValue(i1,j,k1) > threshold ?  32 : 0;
		cubeindex |= getIsoValue(i1,j1,k1) > threshold ?  64 : 0;
		cubeindex |= getIsoValue(i,j1,k1) > threshold ? 128 : 0;
		
		/* Cube is entirely in/out of the surface */
		if (edgeTable[cubeindex] == 0)		return;
		
		/* Find the vertices where the surface intersects the cube */
		
		if (edgeTable[cubeindex] & 1)		vertexInterp(threshold, i,j,k, i1,j,k, vertList[0], normList[0]);
		if (edgeTable[cubeindex] & 2)		vertexInterp(threshold, i1,j,k, i1,j1,k, vertList[1], normList[1]);
		if (edgeTable[cubeindex] & 4)		vertexInterp(threshold, i1,j1,k, i,j1,k, vertList[2], normList[2]);
		if (edgeTable[cubeindex] & 8)		vertexInterp(threshold, i,j1,k, i,j,k, vertList[3], normList[3]);
		if (edgeTable[cubeindex] & 16)		vertexInterp(threshold, i,j,k1, i1,j,k1, vertList[4], normList[4]);
		if (edgeTable[cubeindex] & 32)		vertexInterp(threshold, i1,j,k1, i1,j1,k1, vertList[5], normList[5]);
		if (edgeTable[cubeindex] & 64)		vertexInterp(threshold, i1,j1,k1, i,j1,k1, vertList[6], normList[6]);
		if (edgeTable[cubeindex] & 128)		vertexInterp(threshold, i,j1,k1, i,j,k1, vertList[7], normList[7]);
		if (edgeTable[cubeindex] & 256)		vertexInterp(threshold, i,j,k, i,j,k1, vertList[8], normList[8]);
		if (edgeTable[cubeindex] & 512)		vertexInterp(threshold, i1,j,k, i1,j,k1, vertList[9], normList[9]);
		if (edgeTable[cubeindex] & 1024)	vertexInterp(threshold, i1,j1,k, i1,j1,k1, vertList[10], normList[10]);
		if (edgeTable[cubeindex] & 2048)	vertexInterp(threshold, i,j1,k, i,j1,k1, vertList[11], normList[11]);
		
		for (i=0;triTable[cubeindex][i]!=-1;i+=3) {
			if(bSmoothed){
				//smoothed normals
				normals[vertexCount] = normList[triTable[cubeindex][i]];
				normals[vertexCount+1] = normList[triTable[cubeindex][i+1]];
				normals[vertexCount+2] = normList[triTable[cubeindex][i+2]];
				
			}
			else{
				//faceted
				Vec3f a = vertList[triTable[cubeindex][i+1]] - vertList[triTable[cubeindex][i]];
				Vec3f b = vertList[triTable[cubeindex][i+2]] - vertList[triTable[cubeindex][i+1]];
				
				normals[vertexCount] = normals[vertexCount+1] = normals[vertexCount+2] = a.crossed(b).normalize() * flipNormalsValue;
			}
			
			vertices[vertexCount] = vertList[triTable[cubeindex][i]];
			vertices[vertexCount+1] = vertList[triTable[cubeindex][i+1]];
			vertices[vertexCount+2] = vertList[triTable[cubeindex][i+2]];
			vertexCount += 3;
		}
	}
	else if(!beenWarned){
		cerr<< "ofxMarhingCubes: maximum vertex count exceded. try increasing the maxVertexCount with setMaxVertexCount()";
		beenWarned = true;
	}
}

void ofxMarchingCubes::vertexInterp(float threshold, int i1, int j1, int k1, int i2, int j2, int k2, Vec3f& v, Vec3f& n){
	
	Vec3f& p1 = getGridPoint(i1,j1,k1);
	Vec3f& p2 = getGridPoint(i2,j2,k2);
	
	float& iso1 = getIsoValue(i1,j1,k1);
	float& iso2 = getIsoValue(i2,j2,k2);
	
	if(bSmoothed){
		//we need to interpolate/calculate the normals
		computeNormal( i1, j1, k1 );
		computeNormal( i2, j2, k2 );
		
		Vec3f& n1 = getNormalVal(i1,j1,k1);
		Vec3f& n2 = getNormalVal(i2,j2,k2);
		
		if (abs(threshold-iso1) < 0.00001){
			v = p1;
			n = n1;
			return;
		}
		if (abs(threshold-iso2) < 0.00001){
			v = p2;
			n = n2;
			return;
		}
		if (abs(iso1-iso2) < 0.00001){
			v = p1;
			n = n1;
			return;
		}
		
		//lerp
		float t = (threshold - iso1) / (iso2 - iso1);
		v = p1 + (p2-p1) * t;
		n = n1 + (n2-n1) * t;
	}
	
	else{
		//we'll calc the normal later
		if (abs(threshold-iso1) < 0.00001){
			v = p1;
			return;
		}
		if (abs(threshold-iso2) < 0.00001){
			v = p2;
			return;
		}
		if (abs(iso1-iso2) < 0.00001){
			v = p1;
			return;
		}
		
		//lerp
		v = p1 + (p2-p1) * (threshold - iso1) / (iso2 - iso1);
	}
}

void ofxMarchingCubes::computeNormal( int i, int j, int k ) {
	

	if(getGridPointComputed(i,j,k) == 0){
		Vec3f& n = getNormalVal(i, j, k);// normalVals[i][j][k];
		n.set(getIsoValue(min(resXm1, i+1), j, k) - getIsoValue(max(0,i-1),j,k),
			  getIsoValue(i,min(resYm1, j+1),k) - getIsoValue(i,max(0,j-1),k),
			  getIsoValue(i,j,min(resZm1, k+1)) - getIsoValue(i,j,max(0,k-1)));
		
		n = n.normalize();
		n = n*flipNormalsValue;
		getGridPointComputed(i,j,k) = 1;
	}
};



void ofxMarchingCubes::setGridPoints(){
	cellDim = Vec3f( 1.f/resXm1, 1.f/resYm1, 1.f/resZm1 );
	
	for(int i=0; i<resX; i++){
		for(int j=0; j<resY; j++){
			for(int k=0; k<resZ; k++){
				getGridPoint( i, j, k ).set(float(i)*cellDim.x-.5,
											float(j)*cellDim.y-.5,
											float(k)*cellDim.z-.5);
			}
		}
	}


}





void ofxMarchingCubes::setIsoValue( int x, int y, int z, float value){
	getIsoValue(min(resXm1,x), min(resYm1,y), min(resZm1,z)) = value;
	getGridPointComputed(x,y,z) = 0;
	bUpdateMesh = true;
}

void ofxMarchingCubes::setResolution( int _x, int _y, int _z ){
	
	resX = _x;
	resY = _y;
	resZ = _z;
	resXm1 = resX-1;
	resYm1 = resY-1;
	resZm1 = resZ-1;
	
	isoVals.resize( resX*resY*resZ );
	gridPoints.resize( resX*resY*resZ );
	normalVals.resize( resX*resY*resZ );
	gridPointComputed.resize( resX*resY*resZ );
	
	setGridPoints( );
}

void ofxMarchingCubes::wipeIsoValues( float value){
	
	std::fill(gridPointComputed.begin(), gridPointComputed.end(), 0);
	std::fill(isoVals.begin(), isoVals.end(), value);

}


void ofxMarchingCubes::clear(){
	isoVals.clear();
	gridPoints.clear();
	normalVals.clear();
}

void ofxMarchingCubes::exportObj( string fileName ){
	//super simple obj export. doesn;t have any texture coords, materials, or anything super special.
	//just vertices and normals and not optimized either...
	
	//write file
	string title = "../output/" + fileName + ".obj";

	ofstream outfile (title);
	
	outfile << "# This file uses centimeters as units for non-parametric coordinates."<< endl;
	
	

	Vec3f v, n;
	
	for( int i=0; i<vertexCount; i++){
		v =  vertices[i];
		outfile<< "v ";
		outfile<< v.x << " ";
		outfile<< v.y << " ";
		outfile<< v.z << endl;
	}
	outfile << endl;
	
	for( int i=0; i<vertexCount; i++){
		n = normals[i];
		outfile<< "vn ";
		outfile<< n.x << " ";
		outfile<< n.y << " ";
		outfile<< n.z << endl;
	}
	outfile << endl;
	
	//this only works for triangulated meshes
	for( int i=0; i<vertexCount; i+=3){
		outfile<< "f ";
		for(int j=0; j<3; j++){
			outfile<< i+j+1 ;
			outfile<<"/"<<i+j+1 ;
			outfile<< " ";
		}
		outfile<< endl;
	}
	outfile << endl;
	
	outfile.close();
	
	cout <<"model exported as: "<<title << endl;
}