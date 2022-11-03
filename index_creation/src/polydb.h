

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "model.h"
#include<vector>
#include <iterator>
#include <algorithm>


typedef struct DBElement
{
	int poly_index;
	int tree_num;
	float log_num_obs;
	float det_term;
	float euc_dist;
	float ll_score;
	int pred_class;
	float mu_sig;
	float maha_dist;
	float weight_dist;
}Element;

class PolyDB
{
	std::vector<Element> poly_db;
       	int num_trees;
	
public:
	PolyDB(){
	}
	void constructDB(Model M, std::vector<float> observations, bool ext);
	void sortBy(std::vector<Element>::iterator begin_ptr, std::vector<Element>::iterator end_ptr, char col);
	void sortByTree(std::vector<Element> &poly);
	int closestPoly(char field);
	int closestPolyTreewise();
	int intersectionExperiment();
	int intersectionExperimentExt();
	void print();
};

