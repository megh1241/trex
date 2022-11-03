#ifndef MODEL_H
#define MODEL_H
#include <unordered_set>
#include <vector>
#include "config.h"
#include "polytope.h"
#include "immintrin.h"
struct Poly{
	std::vector<float> mean;
	std::vector<float> var;
	//extra_term = log_num_obs - det_term
	float extra_term;
};

struct DistInd{
	float dist;
	int index;
};
class Model{
	std::vector<Polytope> polytopes;
	std::vector<unsigned long long int> weighted_means;
	//std::vector<std::vector<float>> weighted_means;
	std::vector<int> winning_indices;
	int num_trees;
	std::vector<float> pooled_var_inv;
	float pooled_det_term;
	std::vector<unsigned long long int> range_arr;
	std::vector<std::vector<float>> poly_simd;
	std::map<unsigned long long int, std::pair<int, int>> range_loc_map;
	std::vector<struct Poly> poly_eff;
	std::vector<std::vector<int>> neighbor_arr;
	std::vector<float> elapsed_arr;
	std::vector<float> elapsed_arr2;
	std::vector<float> elapsed_arr3;
	std::vector<float> elapsed_arr4;
	short int *classes;
	float* poly_eff_arr;
	std::vector<int> indices;
	
	public:
	std::pair<int, int> findEuclideanHyper(unsigned long long int observation);
	std::vector<int> findEuclideanHyper2(std::vector<float> observation);
	std::vector<std::pair<Polytope, unsigned long long int>> temp_pair_vec;
	std::vector<std::pair<Polytope, std::vector<float>>> temp_pair_vec2;
	std::vector<std::vector<float>> reverse_vec;
	float *hyperplanes_oned;
	std::vector<std::vector<float>>hyperplanes;
	int8_t *hyper_quant;
	int *class_arr_top;
	int *card_arr_top;
	std::vector<std::map<int8_t, float>> code_val_dict_arr;
	std::vector<std::map<float, int8_t>> reverse_code_val_dict_arr;
       //	= new float [num_hyperplanes*64]();
	Model(){
		num_trees = 0;
	}
	inline std::vector<Polytope> getPolytopes(){
		return polytopes;
	}
	inline int getNumTrees(){
		return num_trees;
	}
	inline void setPooledDetTerm(float term){
		pooled_det_term = term;
	}
	inline float getPooledDetTerm(){
		return pooled_det_term;
	}
	inline void setPooledVarInv(std::vector<float> term){
		pooled_var_inv.clear();
		for(auto ele: term)
			pooled_var_inv.push_back(ele);
	}
	inline std::vector<float> getPooledVarInv(){
		return pooled_var_inv;
	}
	inline void writeTimeToFile(){
        	std::fstream fout;
                std::string filename = "elapsed.csv";
                fout.open(filename, std::ios::out | std::ios::app);
                for(auto i: elapsed_arr){
                     fout<<i<<",";
                }
		fout.close();

                filename = "elapsed2.csv";
                fout.open(filename, std::ios::out | std::ios::app);
                for(auto i: elapsed_arr2){
                     fout<<i<<",";
                }
                fout.close();	
        	
               filename = "elapsed3.csv";
                fout.open(filename, std::ios::out | std::ios::app);
                for(auto i: elapsed_arr3){
                     fout<<i<<",";
                }
                fout.close();	
                filename = "elapsed4.csv";
                fout.open(filename, std::ios::out | std::ios::app);
                for(auto i: elapsed_arr4){
                     fout<<i<<",";
                }
                fout.close();	
	}
	void writeTest(std::vector<std::vector<float>> observation);	
	void read_neighbor_array();
	void readModel();
	void readModelPooledHyper(std::vector<std::vector<float>> observation);
	int findClosestPolytope(std::vector<float> observation);
	int findClosestPolytopeNew(std::vector<float> observation );
	int findTopOne(std::vector<float> observation);
	int findTopK(std::vector<float> observation);
	int findTopKFilter(std::vector<float> observation, std::vector<float>weight);
	int filterNew(std::unordered_set<unsigned long long int> observation, std::vector<float> orig_observation);
	int filterHyper(std::unordered_set<unsigned long long int> observation, std::vector<float> orig_observation);
};
#endif
