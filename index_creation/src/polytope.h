#ifndef POLY_H
#define POLY_H

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
class Polytope
{
	int id;
	std::vector<float> mean_vec;
	std::vector<float> var_vec;
	std::vector<float> ext_mean_vec;
	std::vector<float> ext_var_vec;
	float det_term;
	int num_obs;
	int pred_class;
	float log_num_obs;
	float old_num_obs;
	std::vector<float> pool_v;
	float pool_d;
	float new_term;	
	int card;
public:
	Polytope(int tno, std::vector<float> mv, std::vector<float> vv, float term1, int no, int pc): 
		id(tno),
		mean_vec(mv), 
		var_vec(vv), 
		det_term(term1), 
		num_obs(no), 
		log_num_obs(log((float)no)),
		pred_class(pc){}
	
	Polytope(int tno, std::vector<float> mv, std::vector<float> emv, std::vector<float> vv, std::vector<float> evv, float term1, int no, int pc): 
		id(tno),
		mean_vec(mv), 
		ext_mean_vec(emv), 
		var_vec(vv), 
		ext_var_vec(evv), 
		det_term(term1), 
		num_obs(no), 
		log_num_obs(log((float)no)),
		pred_class(pc){}
	
	
	void print();
	
	float ll_score(std::vector<float> observations);
	float ll_score_ext(std::vector<float> observations);
	
	float maha_dist(std::vector<float> observations);
	
	float euc_mean_dist(std::vector<float> observations);
	float euc_mean_dist_ext(std::vector<float> observations);
	float euc_mean_dist_ext2(std::vector<float> observations);
	float euc_mean_dist_ext3(std::vector<float> observations);
	
	float euc_varmean_dist(std::vector<float> observations);
	float filter_dist(std::vector<float> observations, std::vector<float> weight);

	inline float getNewTerm(){
		return new_term;
	}

	inline void setNewTerm(float a){
		new_term = a;
	}

	inline int getTreeNum(){
		return id;
	}

	inline int getID(){
		return id;
	}
	inline void setID(int currid){
		id = currid;
	}
	inline int getCard(){
		return card;
	}
	inline void setCard(int currid){
		card = currid;
	}

	inline int getPredClass(){
		return pred_class;
	}

	inline int getNumObs(){
		return num_obs;
	}
	
	inline float getLogNumObs(){
		return log_num_obs;
	}

	inline float getDetTerm(){
		return det_term;
	}

	inline void setDetTerm(float new_term){
		det_term = new_term;
	}

	inline void setObs(float new_term){
		old_num_obs = new_term;
	}

	inline float getObs(){
		return old_num_obs;
	}

	inline std::vector<float> getMean(){
		return mean_vec;
	}	
	
	inline std::vector<float> getVar(){
		return var_vec;
	}	
	
	inline std::vector<float> getExtMean(){
		return ext_mean_vec;
	}	
	
	inline std::vector<float> getExtVar(){
		return ext_var_vec;
	}	


	inline void setVar(std::vector<float> newvar){
              var_vec.clear();
              for(auto ele: newvar)
              	var_vec.push_back(ele);
               //copy(newvar.begin(), newvar.end(), var_vec.begin());
	 }	      
	inline void setMean(std::vector<float> newvar){
              mean_vec.clear();
              for(auto ele: newvar)
              	mean_vec.push_back(ele);
               //copy(newvar.begin(), newvar.end(), var_vec.begin());
	 }	      
	inline void setPool(std::vector<float> newvar){
              pool_v.clear();
              for(auto ele: newvar)
              	pool_v.push_back(ele);
               //copy(newvar.begin(), newvar.end(), var_vec.begin());
	 }	    
      	inline std::vector<float>getPool(){
		return pool_v;
	}	
	inline void setPoold(float d){
		pool_d = d;
	}
	inline float getPoold(){
		return pool_d;
	}
};

#endif
