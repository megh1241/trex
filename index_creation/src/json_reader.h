#ifndef JSON_ixx
#define JSON_ixx

#include <vector>
#include "config.h"
#include "polytope.h"

class JSONReader{
	std::string filename;
	public:
	JSONReader();
	JSONReader(std::string fname): filename(fname) {} 
	std::vector<Polytope> readPolytopes() ;
	std::vector<Polytope> readPolytopesCSV() ;
	std::vector<std::vector<float>> readHyperplanes() ;
};
#endif
