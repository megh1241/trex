#include "polydb.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <cstdio>
#include <algorithm>

void PolyDB::constructDB(Model m, std::vector<float>observations, bool ext){
	std::vector<Polytope> poly_vec = m.getPolytopes();
	int iter = 0;
	if (!ext){
	    for (auto ele: poly_vec){
		Element curr_ele = {iter, 
			ele.getTreeNum(), 
			ele.getLogNumObs(), 
			ele.getDetTerm(), 
			ele.euc_mean_dist(observations), 
			ele.ll_score(observations),
			ele.getPredClass(), 
			ele.euc_varmean_dist(observations),
			ele.maha_dist(observations),
			ele.filter_dist(observations, m.getPooledVarInv())
		};
		iter++;
		poly_db.push_back(curr_ele);
	    }
	}else{
	    std::vector<float> observations2;
	    for(auto i: observations)
		    observations2.push_back(i);
	    observations2.push_back(0);
	    for (auto ele: poly_vec){
		Element curr_ele = {iter, 
			ele.getTreeNum(), 
			ele.getLogNumObs(), 
			ele.getDetTerm(), 
			ele.euc_mean_dist(observations), 
			ele.ll_score(observations),
			ele.getPredClass(), 
			ele.euc_mean_dist_ext2(observations2), 
			ele.ll_score_ext(observations2)
		};
		iter++;
		poly_db.push_back(curr_ele);
	    }
	}
	num_trees = m.getNumTrees();
}


void PolyDB::sortByTree(std::vector<Element> &poly_row){
	std::sort(poly_row.begin(), poly_row.end(), [](const Element &x, const Element &y)
			{ 
			return (x.tree_num < y.tree_num);	
			}
		 );
}

void PolyDB::sortBy(std::vector<Element>::iterator begin_ptr, std::vector<Element>::iterator end_ptr, char col){
	switch(col){
		case 'i':
			std::sort(begin_ptr, end_ptr, [](const Element &x, const Element &y){ return (x.poly_index < y.poly_index);});
			return;
		case 'm':
			std::sort(begin_ptr, end_ptr, [](const Element &x, const Element &y){ return (x.euc_dist < y.euc_dist);});
			return;
		case 'd':
			std::sort(begin_ptr, end_ptr, [](const Element &x, const Element &y){ return (x.det_term < y.det_term);});
			return;
		case 'l':
			std::sort(begin_ptr, end_ptr, [](const Element &x, const Element &y){ return (x.ll_score < y.ll_score);});
			return;
		case 'o':
			std::sort(begin_ptr, end_ptr, [](const Element &x, const Element &y){ return (x.log_num_obs < y.log_num_obs);});
			return;
		case 'h':
			std::sort(begin_ptr, end_ptr, [](const Element &x, const Element &y){ return (x.maha_dist < y.maha_dist);});
			return;
		case 'f':
			std::sort(begin_ptr, end_ptr, [](const Element &x, const Element &y){ return (x.weight_dist < y.weight_dist);});
			return;
		default:
			std::sort(begin_ptr, end_ptr, [](const Element &x, const Element &y){ return (x.mu_sig < y.mu_sig);});
			//std::sort(begin_ptr, end_ptr, [](const Element &x, const Element &y){ return (x.log_num_obs < y.log_num_obs);});
			return;
	}
}


void PolyDB::print(){
	/*
        int poly_index;
        int tree_num;
        float log_num_obs;
        float det_term;
        float euc_dist;
        float ll_score;
        int pred_class;
        float mu_sig;
	*/
	int k = std::atoi(Config::getValue("topk").c_str());
	for(int i=0; i<k; ++i){
	Element ele = poly_db[i];
	std::cout<<"index: "<<ele.poly_index<<"\n";
	std::cout<<"log_num_obs: "<<ele.log_num_obs<<"\n";
	std::cout<<"ll score: "<<ele.ll_score<<"\n";
	std::cout<<"euc dist: "<<ele.euc_dist<<"\n";
	std::cout<<"det term: "<<ele.det_term<<"\n";
	std::cout<<"*************************************\n";
	}
	fflush(stdout);
}

int PolyDB::closestPolyTreewise(){
	int siz = poly_db.size();
	sortByTree(poly_db);
	const int n_mean = std::atoi(Config::getValue("percmean").c_str());
	const int n_det = std::atoi(Config::getValue("percdet").c_str());
	const int n_obs = std::atoi(Config::getValue("percobs").c_str());
	int curr_tree_num = poly_db[0].tree_num;
	int tree_beg_iter = 0;
	int tree_end_iter = 0;
	int tree_size = 0;
	std::map<int, int> class_vote_map;
	for(int i=0; i<=siz; ++i){
		if((i==siz)  || poly_db[i].tree_num != curr_tree_num){
			sortBy(poly_db.begin()+tree_beg_iter, poly_db.begin()+tree_beg_iter+tree_size, 'm');	
			//sortBy(poly_db.begin()+tree_beg_iter, poly_db.begin()+tree_beg_iter+std::min(tree_size, n_mean), 'n');	
			//sortBy(poly_db.begin()+tree_beg_iter, poly_db.begin()+tree_beg_iter+std::min(tree_size, n_det), 'o');	
			//sortBy(poly_db.begin()+tree_beg_iter, poly_db.begin()+tree_beg_iter+std::min(tree_size, n_obs), 'l');		
			if(class_vote_map.find(poly_db[0].pred_class) == class_vote_map.end())
				class_vote_map[poly_db[0].pred_class] = 0;

			class_vote_map[poly_db[0].pred_class]++;
			tree_beg_iter = i;
			tree_size = 0;
			curr_tree_num = poly_db[i].tree_num;
		}
		else
			tree_size++;
	}
	int max = -1;
	int max_class = -1;
	for(auto ele : class_vote_map){
		int cl = ele.first;
		int num = ele.second;
		if (num > max){
			max = num;
			max_class = cl;
		}

	}
	return max_class;
}


int PolyDB::closestPoly(char field){
	int siz = poly_db.size();
	const int n_mean = std::atoi(Config::getValue("percmean").c_str());
	const int n_det = std::atoi(Config::getValue("percdet").c_str());
	const int n_obs = std::atoi(Config::getValue("percobs").c_str());
	int curr_tree_num = poly_db[0].tree_num;
	int tree_beg_iter = 0;
	int tree_end_iter = 0;
	int tree_size = 0;
	std::map<int, int> class_vote_map;
	int k = std::atoi(Config::getValue("topk").c_str());
	for(int i=0; i<=siz; ++i){
		if(i==k){
			sortBy(poly_db.begin(), poly_db.begin() + i, field);	
			if(class_vote_map.find(poly_db[0].pred_class) == class_vote_map.end())
				class_vote_map[poly_db[0].pred_class] = 0;

			class_vote_map[poly_db[0].pred_class]++;
			tree_beg_iter = i;
			tree_size = 0;
			curr_tree_num = poly_db[i].tree_num;
		}
		else
			tree_size++;
	}
	int max = -1;
	int max_class = -1;
	for(auto ele : class_vote_map){
		int cl = ele.first;
		int num = ele.second;
		if (num > max){
			max = num;
			max_class = cl;
		}

	}
	return max_class;
}

int PolyDB::intersectionExperiment(){
	char field1 = 's';
	char field2 = 'h';
	int k = std::atoi(Config::getValue("topk").c_str());

	sortBy(poly_db.begin(), poly_db.end(), field1);
	std::vector<int> set1;
	std::vector<int> set2;
	for(int i=0; i<k; ++i){
		set1.push_back(poly_db[i].poly_index);
	}
	sortBy(poly_db.begin(), poly_db.end(), field2);
	for(int i=0; i<k; ++i){
		set2.push_back(poly_db[i].poly_index);
	}
	std::sort(set1.begin(), set1.end());
	std::sort(set2.begin(), set2.end());
	std::vector<int> inter(set1.size() + set2.size());
	std::vector<int>::iterator it, ls;
	ls = std::set_intersection(set1.begin(), set1.end() , set2.begin(), set2.end(), inter.begin());
	return ls - inter.begin();
}

int PolyDB::intersectionExperimentExt(){
	char field1 = 'l';
	char field2 = 'm';
	int k = std::atoi(Config::getValue("topk").c_str());
	//std::cout<<"checkpoint 0\n";
	fflush(stdout);

	sortBy(poly_db.begin(), poly_db.end(), field1);
	std::vector<int> set1;
	std::vector<int> set2;
	//std::cout<<"checkpoint 1\n";
	fflush(stdout);
	for(int i=0; i<k; ++i){
		set1.push_back(poly_db[i].poly_index);
	}
	//std::cout<<"checkpoint 2\n";
	fflush(stdout);
/*	for(int i=0; i<10; ++i){
		std::cout<<"LL ext: " <<poly_db[i].maha_dist<<"\n";
		std::cout<<"LL old: " <<poly_db[i].ll_score<<"\n";
	}
	std::cout<<"**************************************\n";
*/
	sortBy(poly_db.begin(), poly_db.end(), field2);
	for(int i=0; i<k; ++i){
		set2.push_back(poly_db[i].poly_index);
	}
	//std::cout<<"checkpoint 3\n";
	fflush(stdout);
	std::sort(set1.begin(), set1.end());
	std::sort(set2.begin(), set2.end());
	std::vector<int> inter(set1.size() + set2.size());
	std::vector<int>::iterator it, ls;
	ls = std::set_intersection(set1.begin(), set1.end() , set2.begin(), set2.end(), inter.begin());
	//std::cout<<"checkpoint 4\n";
	fflush(stdout);
	return ls - inter.begin();
}
