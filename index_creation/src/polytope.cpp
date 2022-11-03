#include "polytope.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <cstdio>

float Polytope::ll_score_ext(std::vector<float> observations){
	float dist;
	std::vector<float> X_bar;
	std::vector<float> X_bar2;
	std::vector<float> intermediate;
	int siz = ext_mean_vec.size();
	int sign = -1;
	if(ext_mean_vec[siz-1] < 0)
		sign = 1;
	X_bar = subtractVecs(observations, ext_mean_vec);
	intermediate = mulVecs(ext_var_vec, X_bar);
	int size_xbar = X_bar.size();
	for(int i=0; i<size_xbar-1; ++i)
		X_bar2.push_back(X_bar[i]);
	X_bar2.push_back(1);
	dist = 0.5*dotVecs(intermediate, X_bar2);
	return dist;
}

float Polytope::ll_score(std::vector<float> observations){
	float dist;
	std::vector<float> X_bar;
	std::vector<float> intermediate;

	X_bar = subtractVecs(observations, mean_vec);
	intermediate = mulVecs(var_vec, X_bar);
	dist = -0.5 * dotVecs(intermediate, X_bar);
	dist -= det_term;
	if (num_obs > 1) dist += log_num_obs;
	return -1*dist;	
}

float Polytope::euc_mean_dist_ext(std::vector<float> observations){
        float maha_dist;
        std::vector<float> X_bar;
        std::vector<float> X_bar2;
        std::vector<float> intermediate;
        std::vector<float> intermediate2;
        intermediate = mulVecs(ext_mean_vec, ext_var_vec);

        intermediate2 = mulVecs(intermediate, ext_mean_vec);
        X_bar = subtractVecs(observations, intermediate2);
        //X_bar = subtractVecs(intermediate, ext_mean_vec);
        for(int i=0; i<X_bar.size()-1; ++i){
                X_bar2.push_back(X_bar[i]);
        }
        X_bar2.push_back(1);
        maha_dist = dotVecs(X_bar, X_bar2);
        return maha_dist;
}

float Polytope::euc_mean_dist_ext2(std::vector<float> observations){
        float maha_dist;
        std::vector<float> X_bar;
        std::vector<float> X_bar2;
        std::vector<float> intermediate;
        std::vector<float> intermediate2;
	X_bar = subtractVecs(observations, ext_mean_vec);
        
        for(int i=0; i<X_bar.size()-1; ++i){
                X_bar2.push_back(X_bar[i]);
        }
        X_bar2.push_back(1);
	
        maha_dist = dotVecs(X_bar, X_bar2);
	//return maha_dist;
	
	intermediate2 = mulVecs(ext_mean_vec, ext_var_vec);
	float int3 = dotVecs(intermediate2, ext_mean_vec);
        return maha_dist + int3;
	
}
float Polytope::euc_mean_dist_ext3(std::vector<float> observations){
        float maha_dist;
        std::vector<float> X_bar;
        std::vector<float> X_bar2;
        std::vector<float> intermediate;
        std::vector<float> intermediate2;
	maha_dist = dotVecs(observations, observations);	
	/*X_bar = subtractVecs(observations, ext_mean_vec);
        
        for(int i=0; i<X_bar.size()-1; ++i){
                X_bar2.push_back(X_bar[i]);
        }
        X_bar2.push_back(1);

        maha_dist = dotVecs(X_bar, X_bar2);
	return maha_dist;
	*/
	intermediate2 = mulVecs(ext_mean_vec, ext_var_vec);
	float int3 = dotVecs(intermediate2, ext_mean_vec);
        return (maha_dist - int3)*(maha_dist - int3);
}

/*float Polytope::ll_score_ext(std::vector<float> observations){
	float dist;
	std::vector<float> X_bar;
	std::vector<float> intermediate;

	X_bar = subtractVecs(observations, ext_mean_vec);
	intermediate = mulVecs(ext_var_vec, X_bar);
	dist = 0.5 * dotVecs(intermediate, X_bar);
	return dist;	
}*/

float Polytope::maha_dist(std::vector<float> observations){
	float maha_dist;
	std::vector<float> X_bar;
	std::vector<float> intermediate;

	X_bar = subtractVecs(observations, mean_vec);
	intermediate = mulVecs(var_vec, X_bar);
	maha_dist = dotVecs(intermediate, X_bar);
	return maha_dist;
}


float Polytope::euc_mean_dist(std::vector<float> observations){
	float maha_dist;
	std::vector<float> X_bar;
	X_bar = subtractVecs(observations, mean_vec);
	maha_dist = 0.5 * dotVecs(X_bar, X_bar) - log_num_obs;
	float maha_dist2 = dotVecs(X_bar, X_bar);
		//- log_num_obs;;
	return maha_dist2;
}

float Polytope::filter_dist(std::vector<float> observations, std::vector<float> weight){
	float maha_dist;
	std::vector<float> X_bar;
	std::vector<float> X_bar2;
	std::vector<float> obs2;
	std::vector<float> mean2;
	obs2 = mulVecs(observations, weight);
	mean2 = mulVecs(mean_vec, weight);
        X_bar = subtractVecs(obs2, mean2);
	X_bar2 = subtractVecs(observations, mean_vec);
	maha_dist = 0.5*dotVecs(X_bar, X_bar2) - log_num_obs;
	return maha_dist;
}
/*float Polytope::euc_mean_dist_ext(std::vector<float> observations){
	float maha_dist;
	std::vector<float> X_bar;
	X_bar = subtractVecs(observations, ext_mean_vec);
	maha_dist = dotVecs(X_bar, X_bar);
	return maha_dist;
}*/


float Polytope::euc_varmean_dist(std::vector<float> observations){
	float maha_dist;
	std::vector<float> X_bar;
	std::vector<float> intermediate;
	std::vector<float> intermediate2;

	intermediate = mulVecs(mean_vec, var_vec);
	intermediate2 = mulVecs(intermediate, mean_vec);
	X_bar = subtractVecs(observations, intermediate2);
	maha_dist = dotVecs(X_bar, X_bar);
	return maha_dist;
}


/*float Polytope::ll_score_ext(std::vector<float> observations){
	float dist;
	std::vector<float> X_bar;
	std::vector<float> intermediate;

	X_bar = subtractVecs(observations, ext_mean_vec);
	intermediate = mulVecs(ext_var_vec, X_bar);
	dist = 0.5 * dotVecs(intermediate, X_bar);
	return dist;	
}*/



void Polytope::print(){
	std::cout<<"Mean vector: \n";
	for(auto ele: mean_vec)
		std::cout<<ele<<", ";
	std::cout<<"\n Variance vector: \n";
	for(auto ele: var_vec)
		std::cout<<ele<<", ";
	std::cout<<"Term 1: "<<det_term<<"\n";
	std::cout<<"Number of Observations: "<<num_obs<<"\n";
	std::cout<<"Class: "<<pred_class<<"\n";
}
