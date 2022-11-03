#include "json_reader.h"
#include "model.h"
#include "polytope.h"
#include "utils.h"
#include "config.h"
#include "hilbert.h"
#include <vector>
#include <iostream>
#include <array>
#include <functional>
#include <fstream>
#include <iostream>
#include <chrono>
#include <sstream>
#include<string>
#include <assert.h>
#include <limits> 
#include <map>
#include<algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <valarray>
#include <queue>
#include <climits>
#include <unordered_set>
//#include <execution>
//#include "static_sort.h"
#include "mortonND_BMI2.h"
#include "mortonND_LUT.h"
#include <bits/stdc++.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <stdio.h>
#include <new>
#include <math.h>
#include <algorithm>


#define NUM_BITS 10
//using MortonND_4D = mortonnd::MortonNDBmi<4, uint32_t>;
//TODO
//
//
bool has_zero(__m256i x, __m256i INF){
    //const __m256 SIGN_MASK = _mm256_set1_ps(-0.0);
    x = _mm256_cmpeq_epi8(x, INF);
    return _mm256_movemask_epi8(x) != 0;
}

__m256i range_compare(__m256i a, __m256i b, __m256i c, __m256i d) {
    return (_mm256_cmpgt_epi8(a, b) | _mm256_cmpeq_epi8(a, b))  & (_mm256_cmpgt_epi8(c, d) | _mm256_cmpeq_epi8(c, d) ) ;
 }

int8_t getQValue_nnew3(std::map<int8_t, float> dict, std::map<float, int8_t> reverse_dict, float hyper_ele, std::vector<float> reverse_vec, int i_dim){
        if(hyper_ele <= -99999)
                return -128;
        if(hyper_ele >= 99999)
                return 127;
	int siz = reverse_vec.size();
        int found=0;
        auto upper = std::lower_bound(reverse_vec.begin(),reverse_vec.begin()+siz, hyper_ele);       
        
	if(i_dim%2){
		if(upper != reverse_vec.begin())
			return reverse_dict[*(upper-1)];
		else
			return reverse_dict[*upper];
		/*for(int i=0; i<siz-2; ++i){
                        if(reverse_vec[i] <= hyper_ele && reverse_vec[i+1] >= hyper_ele)
                                return reverse_dict[reverse_vec[i]];
                }*/
        }
        else{
		return reverse_dict[*upper];
                /*for(int i=0; i<siz-2; ++i){
                       if(reverse_vec[i] <= hyper_ele && reverse_vec[i+1] >= hyper_ele) {
			       return reverse_dict[reverse_vec[i+1]];
			}
                }*/
        }
	return reverse_dict[reverse_vec[siz-1]];
}

int8_t getQValue(std::map<int8_t, float> dict, std::map<float, int8_t> reverse_dict, float hyper_ele, std::vector<float> reverse_veccc, int i_dim){
        if(hyper_ele <= -99999)
                return -128;
        if(hyper_ele >= 99999)
                return 127;
        std::vector<std::pair<float, int8_t>> reverse_dict_vector;
        reverse_dict_vector.resize(reverse_dict.size());

        std::copy(reverse_dict.begin(), reverse_dict.end(), reverse_dict_vector.begin());
        int found=0;
        if(i_dim%2){
                for(int i=0; i<reverse_dict_vector.size()-2; ++i){
                        if(reverse_dict_vector[i].first <= hyper_ele && reverse_dict_vector[i+1].first >= hyper_ele)
                                return (int8_t)reverse_dict_vector[i].second;
                }
        }
        else{
                for(int i=0; i<reverse_dict_vector.size()-2; ++i){
                        if(reverse_dict_vector[i].first <= hyper_ele && reverse_dict_vector[i+1].first >= hyper_ele)
                                return (int8_t)reverse_dict_vector[i+1].second;
                }
        }
                return (int8_t)reverse_dict_vector[reverse_dict_vector.size()-1].second;

}
int8_t getQValue_new(std::map<int8_t, float> dict, std::map<float, int8_t> reverse_dict, float hyper_ele, std::vector<float> reverse_vec, int i_dim){
        if(hyper_ele <= -99999)
                return -128;
        if(hyper_ele >= 99999)
                return 127;

	int siz = reverse_vec.size();
        int found=0;
        if(i_dim%2){
                for(int i=0; i<siz-2; ++i){
                        if(reverse_vec[i] <= hyper_ele && reverse_vec[i+1] >= hyper_ele)
                                return (int8_t)reverse_dict[reverse_vec[i]];
                }
        }
        else{
                for(int i=0; i<siz-2; ++i){
                        if(reverse_vec[i] <= hyper_ele && reverse_vec[i+1] >= hyper_ele)
                                return (int8_t)reverse_dict[reverse_vec[i+1]];
                }
        }
                return (int8_t)reverse_dict[reverse_vec[siz-1]];

}	
std::vector<float> readBbox(std::string filename){
        std::ifstream file_handler(filename);

        // use a std::vector to store your items.  It handles memory allocation automatically.
        std::vector<float> arr;
        float number;
        while (file_handler>>number) {
                arr.push_back(number);
                file_handler.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        return arr;
}
/*void Model::setHyperrectangles(){
	std::vector<float> bbox_old = readBbox(bbox_fname);
	std::vector<int> fourd_features = std::vector<int>{25, 27, 26, 24};

	__m256i INF = _mm256_set1_epi8(0);
	int num_hyperplanes = 10000;
	int dim = hyperplanes_old_old[0].size();
	//int rem_dimensions = (int)dim/2 - 16;
	std::vector<std::vector<float>> hyperplanes_old;
	float *hyperplane_oned = new float [num_hyperplanes*64]();
	float **hyperplanes = new float*[num_hyperplanes];
	for(int i = 0; i < num_hyperplanes; ++i) {
		hyperplanes[i] = new float[dim];
	}
	
	int8_t test_var;

	int8_t *hyper_quant = new (std::align_val_t(32))int8_t [num_hyperplanes * 64]();
	std::vector<int> f = std::vector<int>{25, 27, 26, 24, 5, 22, 0, 21, 3, 9, 23, 13, 17, 6, 1, 10, 7, 11, 18, 15, 19, 14, 2, 4, 8, 12, 16, 20};
	//std::vector<int> f = std::vector<int>{6, 0, 9, 1, 4, 17, 16, 7, 2, 5, 8, 3, 15, 14, 12, 11, 13, 10};
	std::vector<int> f2;
	for(auto i: f){
		f2.push_back(2*i);
		f2.push_back(2*i+1);
	}

}

*/
std::vector<int> readInt(std::string filename){
        std::ifstream file_handler(filename);

        // use a std::vector to store your items.  It handles memory allocation automatically.
        std::vector<int> arr;
        int number;
        while (file_handler>>number) {
                arr.push_back(number);
                file_handler.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        return arr;
}

std::vector<int> readIntComma(std::string filename){
	std::fstream fin;
	fin.open(filename);
        //fin.open(Config::getValue("featurefilename"));
        std::vector<std::string> row;
	std::string line, word, temp;
        std::vector<float> temp_vector;
	std::vector<int>indices;
        int num_obs = 0;
        while(getline(fin, line, '\n')){
                std::istringstream templine(line);
                std::string data;
                while(getline(templine, data, ',')){
                        indices.push_back(std::atoi(data.c_str()));
                }
        }
	return indices;


}

void Model::readModelPooledHyper(std::vector<std::vector<float>> observation){
	JSONReader json_obj = JSONReader(Config::getValue("polymodelfilename"));
	std::string mean_filename = Config::getValue("meanfilename");
	std::string pool_filename = Config::getValue("poolfilename");
	std::string class_filename = Config::getValue("classfilename");
	std::string card_filename = Config::getValue("cardfilename");
	std::string top_features_filename = Config::getValue("featurefilename");
	
	
	std::vector<int> class_list = readInt(class_filename);
	std::vector<int> card_list = readInt(card_filename);
	std::vector<float> pooled = readBbox(pool_filename);
	indices = readIntComma(top_features_filename);
	hyperplanes = json_obj.readHyperplanes();
	
	std::vector<float> inv_sqrt_pooled;
	for(auto ele: pooled){
		inv_sqrt_pooled.push_back(1.0 / sqrt(ele));
		//inv_sqrt_pooled.push_back(ele);
	}


	//polytopes[0].setPool(inv_sqrt_pooled);
        setPooledVarInv(inv_sqrt_pooled);

	int dim = hyperplanes[0].size();
        std::vector<std::vector<float>> temp_weighted_means;
        
	float min_num = 10000000;
	float max_num = -10000000;	
	
	int iter=0;
	std::string line;
	std::fstream fin;
	fin.open(mean_filename, std::ios::in);
	
	
	std::cout<<"Indics: "<<"\n";
	fflush(stdout);
	for(auto idx : indices){
		std::cout<<idx<<", ";
	}
	std::cout<<"\n";
	fflush(stdout);



	while(getline(fin, line, '\n')){
                std::istringstream templine(line);
                std::string data;
                std::vector<float> temp_vector;
                while(getline(templine, data, ',')){
                        temp_vector.push_back(std::atof(data.c_str()));
                }
                std::vector<float> meanvar = mulVecs(temp_vector, inv_sqrt_pooled);
                std::vector<float> newmeanvar;
                for(auto indx: indices){
                        min_num = std::min(min_num, meanvar[indx]);
                        max_num = std::max(max_num, meanvar[indx]);
                        newmeanvar.push_back(meanvar[indx]);
                }
                temp_weighted_means.push_back(newmeanvar);
		polytopes.push_back(Polytope(iter, temp_vector, temp_vector, 0, 0, class_list[iter]));
     
		iter++;
	}
	
	for(auto ele: observation){
		std::vector<float> meanvar = mulVecs(ele, inv_sqrt_pooled);
                for(auto indx: indices){
                        min_num = std::min(min_num, meanvar[indx]);
                        max_num = std::max(max_num, meanvar[indx]);
                }
	}
	
		
		
	read_neighbor_array();
        int bin_size = pow(2, NUM_BITS) - 1;
        int it1=0;
        int num_in_range = 0;
        int new_range_begin = 0;
        int new_range_end = 0;
        iter=0;

        double class_dict[2] = {0};

        //MortonND_4D::FieldBits = 9;
        //std::vector<std::pair<Polytope, unsigned long long int>> temp_pair_vec;
        for(auto row: temp_weighted_means){
                unsigned long long int temp1[4];
                it1 = 0;
                for(auto ele: row){
                        unsigned long long int new_ele = (unsigned long long int)(bin_size*(ele - min_num) / (max_num - min_num) );
                        //TODO: convert to hilbert

                        temp1[it1] = new_ele;
                        it1++;
                }
                //int f1 = (int) temp1[0];
                //int f2 = (int) temp1[1];
                //int f3 = (int) temp1[2];
                //int f4 = (int) temp1[3];
                //bitmask_ hilbert_c2i(unsigned nDims, unsigned nBits, bitmask_t const coord[])
                //unsigned long long int hilbert_index = (unsigned long long int)  MortonND_4D::Encode(f1, f2, f3, f4);
                unsigned long long int hilbert_index = hilbert_c2i(4, NUM_BITS, temp1);
                temp_pair_vec.push_back(std::pair<Polytope, unsigned long long int>(polytopes[iter] , hilbert_index));
                //weighted_means.push_back(hilbert_index);
                iter++;
        }
	std::cout<<"temp par vec size: "<<temp_pair_vec.size()<<"\n";
        std::cout<<"CHECKPOINT 2!!!\n";
        fflush(stdout);
        std::sort(temp_pair_vec.begin(), temp_pair_vec.end(), [](auto &left, auto &right) {return left.second < right.second;});
	polytopes.clear();
	int num_hyperplanes = hyperplanes.size();
	class_arr_top = new int[num_hyperplanes];
        card_arr_top = new int[num_hyperplanes];
        int kit=0;
        for(auto &ele: temp_pair_vec){
                //ele.setNewTerm(2*ele.getDetTerm - ele.getLogNumObs())
		polytopes.push_back(ele.first);
                temp_pair_vec2.push_back(std::pair<Polytope, std::vector<float> >(ele.first , temp_weighted_means[ele.first.getID()]));
                class_arr_top[kit] = ele.first.getPredClass();
                card_arr_top[kit] = ele.first.getCard();
                weighted_means.push_back(ele.second);
                kit++;
        }
	__m256i INF = _mm256_set1_epi8(0);
        hyperplanes_oned = new float [num_hyperplanes*64]();
        int8_t test_var;

        hyper_quant = new (std::align_val_t(32))int8_t [num_hyperplanes * 64]();

        int tot_num_features = (int) dim/2;
        std::vector<std::vector<float>> column_wise(dim);
        int j=0;
	std::cout<<"dim: "<<dim<<"\n";
	fflush(stdout);
        for(auto &ele: temp_pair_vec){
                int k=ele.first.getID();
                for(int i=0; i<dim; ++i){
                        hyperplanes_oned[j*64 + i] = hyperplanes[k][i];
                        column_wise[i].push_back(hyperplanes[k][i]);
                	//std::cout<<"here: "<<i<<"\n";
			//std::cout<<column_wise[i].size()<<"\n";
		}
                j++;
        }


	std::vector<std::vector<float>> hyperplanes_temp;
	for(auto row: hyperplanes){
		std::vector<float> temp;
		for(auto j: row){
			temp.push_back(j);
		}
		hyperplanes_temp.push_back(temp);
	}
	
	hyperplanes.clear();	

	for(auto &ele: temp_pair_vec){
		int k=ele.first.getID();
		std::vector<float> temp2;
		for(int j=0; j<dim; ++j){
			temp2.push_back(hyperplanes_temp[k][j]);
		}
		hyperplanes.push_back(temp2);	
	}
		
		
		
		
        std::cout<<"CHECKPOINT 6!!!\n";
        fflush(stdout);
	//std::vector<std::vector<float>> reverse_vec;
        for(auto &col: column_wise){
		std::sort(col.begin(), col.end());
                int start_pos=0;
                int end_pos=col.size()-1;
		int iter=0;
                int flag=0;
                std::map<int8_t, float> code_val_dict;
                std::map<float, int8_t> reverse_code_val_dict;
                for(auto ele: col){
                        //std::cout<<ele<<",";
                        if (ele<=-99999){
                                start_pos=iter;
                        }
                        if(ele >= 99999){
                                end_pos=iter;
                                break;
                        }
                        iter++;
                }
                //std::cout<<"start_pos; "<<start_pos<<"\n";
                //std::cout<<"end_pos; "<<end_pos<<"\n";
                int delt = (end_pos-start_pos+1)/253;
                delt++;
                //std::cout<<"delt: "<<delt<<"\n";
                int8_t iter100=-127;
                for(int i=start_pos; i<=end_pos; i+=delt){
                        code_val_dict[iter100] = col[i];
                        reverse_code_val_dict[col[i]] = iter100;
                        iter100++;
                }
		std::vector<float> temp_rev;
		for(auto kv: reverse_code_val_dict){
			temp_rev.push_back(kv.first);
		}
		reverse_vec.push_back(temp_rev);
                //std::cout<<"iter; "<<+iter100<<"\n";
                code_val_dict_arr.push_back(code_val_dict);
                reverse_code_val_dict_arr.push_back(reverse_code_val_dict);

        }
        std::cout<<"CHECKPOINT 7!!!\n";
        fflush(stdout);

	/*
	for (auto i: reverse_vec[0]){
		std::cout<<i<<", ";
	}
        fflush(stdout);
	std::cout<<"*********************************************\n";

	for(auto kv: reverse_code_val_dict_arr[0]){
		std::cout<<kv.first<<": "<<kv.second<<"\n";
	}	
        fflush(stdout);

	std::cout<<"*********************************************\n";
        fflush(stdout);
	*/
	std::cout<<"NUM OF HYPERPLANES: "<<num_hyperplanes<<"\n";
	fflush(stdout);
        int flag;
        int num_met = 0;
        int met_list[num_hyperplanes+1];
        auto start = std::chrono::steady_clock::now();
        for (int j=0; j<num_hyperplanes; ++j){
		for(int i=0; i<dim; ++i){
                        if (i%2==0){
                                hyper_quant[j*64 + i/2] = getQValue(code_val_dict_arr[i], reverse_code_val_dict_arr[i], hyperplanes[j][i], reverse_vec[i], i);
                                //hyper_quant[j][i/2] = getQValue(code_val_dict_arr[i], reverse_code_val_dict_arr[i], hyperplanes[j][i], i);
                        }else{
                                hyper_quant[j*64 + (i-1)/2 + 32] = getQValue(code_val_dict_arr[i], reverse_code_val_dict_arr[i], hyperplanes[j][i], reverse_vec[i],i);
                                //hyper_quant[j][(i-1)/2 + tot_num_features] = getQValue(code_val_dict_arr[i], reverse_code_val_dict_arr[i], hyperplanes[j][i], i);
                        }
                }
		if(j%500000==0){
		const auto end = std::chrono::steady_clock::now();
		std::cout<<std::chrono::duration <double, std::milli> (end-start).count()<<",";
		fflush(stdout);
                start = std::chrono::steady_clock::now();
        	}
	}
        int obs_id=0;
        std::map<unsigned long long int, int> log_map;
        std::map<unsigned long long int, std::vector<int>> bucket_obsid_map;

        for(auto &dist: weighted_means){
                if (log_map.find(dist) == log_map.end()){
                        log_map[dist] = 0;
                        bucket_obsid_map[dist]  = std::vector<int>();
                }
                bucket_obsid_map[dist].push_back(obs_id);
                log_map[dist]++;
                obs_id++;
        }
        std::cout<<"CHECKPOINT 4!!!\n";
        fflush(stdout);

        int curr_card = 0;
        std::vector<unsigned long long int> temp2;
        std::vector<std::vector<unsigned long long int>> key_ranges;
        //temp2.push_back(0);
        for(auto &item: log_map){
                unsigned long long int key = item.first;
                int card = item.second;
                temp2.push_back(key);
                //std::sort(temp2.begin(), temp2.end());
                if(curr_card >5000){
                        key_ranges.push_back(temp2);
                        temp2.clear();
                        curr_card = 0;
                }
                curr_card+=card;
        }
	if(temp2.size()){
                        key_ranges.push_back(temp2);
                        temp2.clear();
	}
        int kiter=0;
        for(auto item: key_ranges){
                int siz = item.size();
                if (siz < 1)
                        continue;
                unsigned long long int range_lower = item[0];
                unsigned long long int range_upper = item[siz-1];
                int min_arr_index;
                min_arr_index = bucket_obsid_map[range_lower][0];
                siz = bucket_obsid_map[range_upper].size();
                int max_arr_index = bucket_obsid_map[range_upper][siz-1];
                int diff = max_arr_index - min_arr_index;
                if (max_arr_index - min_arr_index <0)
                        std::cout<<"range upper smaller than lower: "<<"\n";
                range_loc_map[range_lower] = std::pair<int, int> (min_arr_index, max_arr_index);
                kiter++;
        }
        for(auto &key: range_loc_map){
                range_arr.push_back(key.first);
        }


        std::cout<<"CHECKPOINT 6!!!\n";
        fflush(stdout);
        Config::setConfigItem(std::string("maxnum"), std::to_string(max_num));
        Config::setConfigItem(std::string("minnum"), std::to_string(min_num));
        std::cout<<"CHECKPOINT 7!!!\n";
        fflush(stdout);

}





void Model::readModel(){
	JSONReader json_obj = JSONReader(Config::getValue("polymodelfilename"));
	polytopes = json_obj.readPolytopesCSV();
	std::cout<<"reacjed jer \n";
	fflush(stdout);
	std::string mean_filename = Config::getValue("meanfilename");
	std::string pool_filename = Config::getValue("poolfilename");
	std::fstream fin;
	fin.open(mean_filename, std::ios::in);
	int i=0;
	std::string line, word, temp;
	std::vector<std::vector<float>> mean_new;
        while(getline(fin, line, '\n')){
                std::istringstream templine(line);
                std::string data;
                std::vector<float> temp_vector;
                while(getline(templine, data, ',')){
                        temp_vector.push_back(std::atof(data.c_str()));
                }
		mean_new.push_back(temp_vector);
		if (i<polytopes.size()){
			polytopes[i].setMean(temp_vector);
			i++;
		}else{
			Polytope temp_polytope(Polytope(polytopes.size(), temp_vector, temp_vector, 0, 0, 0));
			polytopes.push_back(temp_polytope);
		}
	}
	fin.close();
	std::cout<<"reacjed jeri2 \n";
	fflush(stdout);
	std::vector<float> pooled = readBbox(pool_filename);
	std::cout<<"reacjed jer3 \n";
	fflush(stdout);

	hyperplanes = json_obj.readHyperplanes();
	int dim = hyperplanes[0].size();
	for (auto ele: polytopes[0].getPool())
		std::cout<<ele<<"\n";
	polytopes[0].setPool(pooled);
	setPooledVarInv(polytopes[0].getPool());
	setPooledDetTerm(polytopes[0].getPoold());
	read_neighbor_array();
	std::cout<<"READ and pooled set!!\n";
	fflush(stdout);
	std::vector<float>pooled_var_ext = polytopes[0].getPool();
	for(int i=0; i<pooled_var_ext.size(); ++i){
		pooled_var_ext[i] = sqrt(pooled_var_ext[i]);
		//std::cout<<"pool: "<<pooled_var_ext[i]<<"\n";
	}
	pooled_var_ext.push_back(1);
	float max_obs = 0;
	for(auto poly: polytopes){
		float log_obs = 2*log(poly.getObs());
		if(log_obs > max_obs)
			max_obs = log_obs;
	}

	std::cout<<"pooled set2!!\n";
	fflush(stdout);

	fin.open(Config::getValue("featurefilename"));
	std::vector<std::string> row;
	std::vector<float> temp_vector;
	int num_obs = 0;
	while(getline(fin, line, '\n')){
		std::istringstream templine(line);
		std::string data;
		while(getline(templine, data, ',')){
			indices.push_back(std::atoi(data.c_str()));
		}
	}
	std::cout<<"features read\n";
	fflush(stdout);

	float min_num = 10000000;
	float max_num = -10000000;	


	std::vector<std::vector<float>> temp_weighted_means;
	for(auto &poly: polytopes){
		std::vector<float> external  = poly.getMean();
		external.push_back(sqrt(-2*log(poly.getObs()) + max_obs));
		poly.setNewTerm(sqrt(-2*log(poly.getObs()) + max_obs));
		std::vector<float> meanvar = mulVecs(external, pooled_var_ext);
		std::vector<float> newmeanvar;
		for(auto indx: indices){
			min_num = std::min(min_num, meanvar[indx]);
			max_num = std::max(max_num, meanvar[indx]);
			newmeanvar.push_back(meanvar[indx]);
		}
		temp_weighted_means.push_back(newmeanvar);
	}
	std::cout<<"CHECKPOINT 1!!!\n";
	fflush(stdout);
	int bin_size = pow(2, NUM_BITS) - 1;
	int it1=0;
	int num_in_range = 0;
	int new_range_begin = 0;
	int new_range_end = 0;
	int iter=0;
	
	double class_dict[2] = {0};
	
	//MortonND_4D::FieldBits = 9;
	//std::vector<std::pair<Polytope, unsigned long long int>> temp_pair_vec;
	for(auto row: temp_weighted_means){
		unsigned long long int temp1[4];
		it1 = 0;
		for(auto ele: row){
			unsigned long long int new_ele = (unsigned long long int)(bin_size*(ele - min_num) / (max_num - min_num) );
			//TODO: convert to hilbert

			temp1[it1] = new_ele;
			it1++;
		}
		//int f1 = (int) temp1[0];
		//int f2 = (int) temp1[1];
		//int f3 = (int) temp1[2];
		//int f4 = (int) temp1[3];
		//bitmask_ hilbert_c2i(unsigned nDims, unsigned nBits, bitmask_t const coord[])
		//unsigned long long int hilbert_index = (unsigned long long int)  MortonND_4D::Encode(f1, f2, f3, f4);
		unsigned long long int hilbert_index = hilbert_c2i(4, NUM_BITS, temp1);
		temp_pair_vec.push_back(std::pair<Polytope, unsigned long long int>(polytopes[iter] , hilbert_index));
		//weighted_means.push_back(hilbert_index);
		iter++;
	}
	std::cout<<"CHECKPOINT 2!!!\n";
	fflush(stdout);
	std::sort(temp_pair_vec.begin(), temp_pair_vec.end(), [](auto &left, auto &right) {return left.second < right.second;});
	polytopes.clear();

	int num_hyperplanes = hyperplanes.size();
	class_arr_top = new int[num_hyperplanes];
	
	int kit=0;
	for(auto &ele: temp_pair_vec){
		//ele.setNewTerm(2*ele.getDetTerm - ele.getLogNumObs())	
		polytopes.push_back(ele.first);
		class_arr_top[kit] = ele.first.getPredClass();
		weighted_means.push_back(ele.second);
		kit++;
	}
	std::cout<<"CHECKPOINT 3!!!\n";
	fflush(stdout);
	fflush(stdout);

	__m256i INF = _mm256_set1_epi8(0);	
	std::cout<<"CHECKPOINT 4!!!\n";
	fflush(stdout);
	fflush(stdout);
	hyperplanes_oned = new float [num_hyperplanes*64]();
	std::cout<<"CHECKPOINT 4.1!!!\n";
	fflush(stdout);
	fflush(stdout);
	int8_t test_var;

	hyper_quant = new (std::align_val_t(32))int8_t [num_hyperplanes * 64]();

	std::cout<<"CHECKPOINT 5!!!\n";
	fflush(stdout);

	i=0;
	int tot_num_features = (int) dim/2;
	std::vector<std::vector<float>> column_wise(dim);	
	int j=0;
	for(auto &ele: temp_pair_vec){
		int k=ele.first.getID();
		for(int i=0; i<dim; ++i){
			hyperplanes_oned[j*64 + i] = hyperplanes[k][i];
			column_wise[i].push_back(hyperplanes[k][i]);
		}
		j++;
	}
	std::vector<std::vector<float>> hyperplanes_temp;
	for(auto row: hyperplanes){
		std::vector<float> temp;
		for(auto j: row){
			temp.push_back(j);
		}
		hyperplanes_temp.push_back(temp);
	}
	
	hyperplanes.clear();	

	for(auto &ele: temp_pair_vec){
		int k=ele.first.getID();
		std::vector<float> temp2;
		for(int j=0; i<dim; ++j){
			temp2.push_back(hyperplanes_temp[k][j]);
		}
		hyperplanes.push_back(temp2);	
	}
		
		
		
		
	std::cout<<"CHECKPOINT 6!!!\n";
	fflush(stdout);
	//std::vector<std::map<int8_t, float>> code_val_dict_arr;
	//std::vector<std::map<float, int8_t>> reverse_code_val_dict_arr;
	for(auto &col: column_wise){
		std::sort(col.begin(), col.end());
		int start_pos=0;
		int end_pos=col.size()-1;
		int iter=0;
		int flag=0;
		std::map<int8_t, float> code_val_dict;
		std::map<float, int8_t> reverse_code_val_dict;
		for(auto ele: col){
			//std::cout<<ele<<",";
			if (ele<=-99999){
				start_pos=iter;
			}
			if(ele >= 99999){
				end_pos=iter;
				break;
			}
			iter++;
		}
		//std::cout<<"start_pos; "<<start_pos<<"\n";
		//std::cout<<"end_pos; "<<end_pos<<"\n";
		int delt = (end_pos-start_pos+1)/253;
		delt++;
		//std::cout<<"delt: "<<delt<<"\n";
		int8_t iter100=-127;
		for(int i=start_pos; i<=end_pos; i+=delt){
			code_val_dict[iter100] = col[i];
			reverse_code_val_dict[col[i]] = iter100;
			iter100++;
		}
		//std::cout<<"iter; "<<+iter100<<"\n";
		code_val_dict_arr.push_back(code_val_dict);
		reverse_code_val_dict_arr.push_back(reverse_code_val_dict);

	}
	std::cout<<"CHECKPOINT 7!!!\n";
	fflush(stdout);

	int flag;
	int num_met = 0;
	int met_list[num_hyperplanes+1];
	for (int j=0; j<num_hyperplanes; ++j){
		for(int i=0; i<dim; ++i){
			if (i%2==0){
				hyper_quant[j*64 + i/2] = getQValue(code_val_dict_arr[i], reverse_code_val_dict_arr[i], hyperplanes[j][i], reverse_vec[i], i);
				//hyper_quant[j][i/2] = getQValue(code_val_dict_arr[i], reverse_code_val_dict_arr[i], hyperplanes[j][i], i);
			}else{
				hyper_quant[j*64 + (i-1)/2 + 32] = getQValue(code_val_dict_arr[i], reverse_code_val_dict_arr[i], hyperplanes[j][i], reverse_vec[i], i);
				//hyper_quant[j][(i-1)/2 + tot_num_features] = getQValue(code_val_dict_arr[i], reverse_code_val_dict_arr[i], hyperplanes[j][i], i);
			}
		}
	}



	std::cout<<"CHECKPOINT 8!!!\n";
	fflush(stdout);


	/*for(auto ele: temp_pair_vec){
	  struct Poly t;
	  std::vector<float> meanv(ele.first.getMean());
	  std::vector<float> varv(ele.first.getVar());
	  float extra_term = -1*ele.first.getLogNumObs() + ele.first.getDetTerm();

	  for(auto et: meanv)
	  t.mean.push_back(et);
	  for(auto et: varv)
	  t.var.push_back(et);
	  t.extra_term = extra_term;

	  poly_eff.push_back(t);
	  }
	  */
	int tot_size = polytopes.size() * (polytopes[0].getMean().size()+1) * 2 ; 
	poly_eff_arr = new float[tot_size];
	classes = new short int [tot_size];
	iter = 0;
	int iter2 = 0;
	for(auto ele: temp_pair_vec){
		struct Poly t;
		std::vector<float> meanv(ele.first.getMean());
		std::vector<float> varv(ele.first.getVar());
		float extra_term = ele.first.getNewTerm();
		classes[iter2] = ele.first.getPredClass();
		iter2++;	

		for(auto et: meanv){
			poly_eff_arr[iter] = et;
			iter++;
		}
		poly_eff_arr[iter] = extra_term;
		iter++;
		for(auto et: varv){
			poly_eff_arr[iter] = et;
			iter++;
		}
		poly_eff_arr[iter] = 0;

	}



	//std::sort(weighted_means.begin(), weighted_means.end());
	int obs_id=0;
	std::vector<unsigned long long int> log_distances;
	std::map<unsigned long long int, int> log_map;
	std::map<unsigned long long int, std::vector<int>> bucket_obsid_map;

	for(auto &dist: weighted_means){
		if (log_map.find(dist) == log_map.end()){
			log_map[dist] = 0;
			bucket_obsid_map[dist]  = std::vector<int>();
		}	
		bucket_obsid_map[dist].push_back(obs_id);
		log_map[dist]++;
		log_distances.push_back(dist);
		obs_id++;
	}
	std::cout<<"CHECKPOINT 4!!!\n";
	fflush(stdout);

	int curr_card = 0;
	std::vector<unsigned long long int> temp2;
	std::vector<std::vector<unsigned long long int>> key_ranges;
	//temp2.push_back(0);
	for(auto &item: log_map){
		unsigned long long int key = item.first;
		int card = item.second;
		temp2.push_back(key);
		//std::sort(temp2.begin(), temp2.end());
		if(curr_card >10000){
			key_ranges.push_back(temp2);
			temp2.clear();
			curr_card = 0;
		}
		curr_card+=card;
	}
	std::cout<<"CHECKPOINT 5!!!\n";
	fflush(stdout);
	//range_loc_map
	int kiter=0;
	for(auto item: key_ranges){
		int siz = item.size();
		if (siz <= 1)
			continue;
		unsigned long long int range_lower = item[0];
		unsigned long long int range_upper = item[siz-1];
		int min_arr_index;
		//if (range_lower == 0 && bucket_obsid_map[range_lower].size()==0)
		//	min_arr_index = bucket_obsid_map[item[1]][0];
		//else
		min_arr_index = bucket_obsid_map[range_lower][0]; 
		siz = bucket_obsid_map[range_upper].size();
		int max_arr_index = bucket_obsid_map[range_upper][siz-1]; 
		int diff = max_arr_index - min_arr_index;
		if (max_arr_index - min_arr_index <0)
			std::cout<<"range upper smaller than lower: "<<"\n";
		range_loc_map[range_lower] = std::pair<int, int> (min_arr_index, max_arr_index);
		kiter++;
	}
	for(auto &key: range_loc_map){
		range_arr.push_back(key.first);
		//std::cout<<"size of key: "<<key.second.second - key.second.first<<"\n";	
	}


	std::cout<<"CHECKPOINT 6!!!\n";
	fflush(stdout);
	Config::setConfigItem(std::string("maxnum"), std::to_string(max_num));
	Config::setConfigItem(std::string("minnum"), std::to_string(min_num));
	std::cout<<"CHECKPOINT 7!!!\n";
	fflush(stdout);

}

void Model::writeTest(std::vector<std::vector<float>> observation){
	std::vector<std::vector<float>> samples;
	std::string fname = Config::getValue("writefname");
	std::fstream fout;
	fout.open(fname, std::ofstream::out | std::ofstream::app );
	std::vector<float> pool_var_vec = polytopes[0].getPool();
	std::cout<<"enter here size: "<<pool_var_vec.size()<<"\n";
	fflush(stdout);
	for(int i=0; i<pool_var_vec.size();++i){
		pool_var_vec[i] = sqrt(pool_var_vec[i]);
	}
	for(auto &obs: observation){

		obs.push_back(0);
		pool_var_vec.push_back(1.0);
		std::vector<float> meanvar = mulVecs(obs, pool_var_vec);
		for(auto ele: meanvar){
			fout<<ele<<",";
		}
		fout<<"\n";
	}
	fout.close();
}

int Model::findClosestPolytope(std::vector<float> observation){
	std::vector<float> observation2;
	int iter = 0;	
	/*for(auto i: observation){
	  observation2.push_back(i*pooled_var_inv[iter]);
	  iter++;
	  }
	  */
	if(Config::getValue("measure") == std::string("topone"))
		return findTopOne(observation);
	else
		return findTopKFilter(observation, pooled_var_inv);
	//return findTopK(observation);
}

int readGarbage(){
	std::fstream fi;
	int j;
	int sum;
	fi.open("/data/rand_file.txt");
	for(int i=0; i<70000000; ++i)
	{
		fi>>j;
		sum+=j;
	}
	fi.close();
	return sum;
}

int Model::findClosestPolytopeNew(std::vector<float> observation){
	std::cout<<"ch1\n";
	fflush(stdout);
	std::vector<float> observation2;
	unsigned long long int hilbert_cube[4];
	int iter = 0;
	float max_num = std::atof(Config::getValue("maxnum").c_str());		
	float min_num = std::atof(Config::getValue("minnum").c_str());		
	int it1=0;
	int bin_size = pow(2, NUM_BITS) - 1;
	//indices are the top 4 features split on. It's size is 4.

	//int gar = readGarbage();	
	std::vector<float> pooled_var_inv = getPooledVarInv();
	
	
	std::cout<<"Obs Indics: "<<"\n";
	fflush(stdout);
	for(auto idx : indices){
		std::cout<<idx<<", ";
	}
	std::cout<<"\n";
	fflush(stdout);


	
	std::cout<<"pooled var inv: "<<"\n";
	fflush(stdout);
	for(auto idx : pooled_var_inv){
		std::cout<<idx<<", ";
	}
	std::cout<<"\n";
	fflush(stdout);


	
	
	
	auto start = std::chrono::steady_clock::now();
	for(auto iter: indices){
		float item = observation[iter]*pooled_var_inv[iter];
		item = std::max(min_num, item);
		item = std::min(max_num, item);
		//TODO: convert to hilbert index
		unsigned long long int to_convert = (unsigned long long int) (bin_size * (item - min_num) / (max_num - min_num));
		hilbert_cube[it1] = to_convert;
		it1++;
		//observation2.push_back(hilbert_index);
	}
	unsigned long long int hilbert_indices_all_neighbors[1];

	int it2=0;
	//unsigned long long int hilbert_index = mortonnd::MortonNDBmi<4, uint32_t>::Encode(hilbert_cube[0], hilbert_cube[1], hilbert_cube[2], hilbert_cube[3]);
	//unsigned long long int hilbert_index = hilbert_c2i(4, NUM_BITS, hilbert_cube);
	//hilbert_indices_all_neighbors[0] = hilbert_index;
	//MortonND_4D::FieldBits = 9;
	auto start2 = std::chrono::steady_clock::now();
	for(auto neighbor: neighbor_arr){
		unsigned long long int cube[4];
		int flag = 1;
		for(int i=0; i<4; ++i){
			cube[i] = hilbert_cube[i] + neighbor[i];
			if (cube[i] < 0 || cube[i] >= bin_size){
				flag = -1;
				break;
			}
		}	
		if(flag > 0){
			//int f1 = (int) cube[0];
			//int f2 = (int) cube[1];
			//int f3 = (int) cube[2];
			//int f4 = (int) cube[3];
			//auto encoding = MortonND_4D::Encode(f1, f2, f3, f4);
			//hilbert_indices_all_neighbors[it2] = encoding;
			hilbert_indices_all_neighbors[it2]  = hilbert_c2i(4, NUM_BITS, cube);

			//unsigned long long int hilbert_index = mortonnd::MortonNDBmi<4, uint32_t>::Encode(cube[0], cube[1], cube[2], cube[3]);
			it2++;
		}
	}
	auto end2 = std::chrono::steady_clock::now();
	double elapsed2 = std::chrono::duration<double, std::milli>(end2 - start2).count();
	//elapsed_arr2.push_back(elapsed2);
	std::unordered_set<unsigned long long int> hilbert_set( hilbert_indices_all_neighbors, hilbert_indices_all_neighbors+it2 );
	std::cout<<"PRINTING LENGTH OF SET: "<<hilbert_set.size();
	//hilbert_indices_all_neighbors.erase( unique( hilbert_indices_all_neighbors, hilbert_indices_all_neighbors+it2 ), hilbert_indices_all_neighbors+it2 );
	//observation2.push_back(0);
	//TODO: uodate to list of hilbert indices (81 neighboring cubes)
	int ret;
	ret = filterHyper(hilbert_set, observation);
	auto end = std::chrono::steady_clock::now();
	//	double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
	//	elapsed_arr.push_back(elapsed);
	return ret;
}

template<typename T>
std::vector<int> argsort(const std::vector<T> &array) {
	std::vector<int> indices(array.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::sort(indices.begin(), indices.end(),
			[&array](int left, int right) -> bool {
			// sort indices according to corresponding array element
			return array[left] < array[right];
			});

	return indices;
}


void Model::read_neighbor_array(){
	std::string filename = std::string("neighbors_all.csv");
	std::fstream fin;
	fin.open(filename, std::ios::in);
	std::vector<std::string> row;
	std::string line, word, temp;
	std::vector<int> temp_vector;

	while(getline(fin, line, '\n')){


		std::istringstream templine(line);
		std::string data;
		while(getline(templine, data, ',')){
			temp_vector.push_back((unsigned long long int)std::atoi(data.c_str()));
		}
		int siz = temp_vector.size();
		neighbor_arr.push_back(temp_vector);
		temp_vector.clear();
	}
}

float ll_score(struct Poly poly_ele, std::vector<float> observation){
	int size = observation.size();
	float res = 0;
	for (int i=0; i<size; ++i){
		float t1;
	        
		t1= observation[i] - poly_ele.mean[i];
		res += (std::pow(t1,2) * poly_ele.var[i]);
	}
	res += poly_ele.extra_term;
	return res;
}

static inline __m128 masked_read(int d, const float *x) {
	assert (0 <= d && d < 4);
	__attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
	switch (d) {
		case 3:buf[2] = x[2];
		case 2:buf[1] = x[1];
		case 1:buf[0] = x[0];
	}
	return _mm_load_ps(buf);
}



float ll_score_new (const float *x, const float *y, const float *z, size_t d){
	float res = 0;
	for(int i=0; i<d; ++i){
		float sq = x[i] - y[i];
		res+= 0.5*sq*sq*z[i];
	}
	return res;
}

/*float* ll_score_simd2 (const float *x, const float *y, size_t num_ele, size_t d){
  float res = 0;
  int k=0;
  float *arr = new float[num_ele];
  for(int i=0; i<num_ele*d; ++i){

  res+=(x[i] - y[i])*z[i];
  num++;
  if (num==d){
  num=0;
  arr[k++]=res;
  res = 0;
  }
  }
  return arr;
  }*/


float ll_score_simd (const float *x, const float *y, const float *z, size_t d)
{
	/*__m512 msum1 = _mm512_setzero_ps();

	  while (d >= 16) {
	  __m512 mx = _mm512_loadu_ps (x); x += 16;
	  __m512 my = _mm512_loadu_ps (y); y += 16;
	  __m512 mz = _mm512_loadu_ps (z); z += 16;
	  const __m512 a_m_b1 = mx - my;
	  msum1 += a_m_b1 * a_m_b1 * mz ;
	  d -= 16;
	  }

	  __m256 msum2 = _mm512_extractf32x8_ps(msum1, 1);
	  msum2 +=       _mm512_extractf32x8_ps(msum1, 0);
	  */
	__m256 msum2 = _mm256_setzero_ps();
	while (d >= 8) {
		__m256 mx = _mm256_loadu_ps (x); x += 8;
		__m256 my = _mm256_loadu_ps (y); y += 8;
		__m256 mz = _mm256_loadu_ps (z); z += 8;
		const __m256 a_m_b1 = mx - my;
		msum2 += a_m_b1 * a_m_b1 * mz ;
		d -= 8;
	}

	__m128 msum3 = _mm256_extractf128_ps(msum2, 1);
	msum3 +=       _mm256_extractf128_ps(msum2, 0);

	if (d >= 4) {
		__m128 mx = _mm_loadu_ps (x); x += 4;
		__m128 my = _mm_loadu_ps (y); y += 4;
		__m128 mz = _mm_loadu_ps (z); z += 4;
		const __m128 a_m_b1 = mx - my;
		msum3 += a_m_b1 * a_m_b1 * mz ;
		d -= 4;
	}

	if (d > 0) {
		__m128 mx = masked_read (d, x);
		__m128 my = masked_read (d, y);
		__m128 mz = masked_read (d, z);
		__m128 a_m_b1 = mx - my;
		msum3 += a_m_b1 * a_m_b1 * mz ;
	}

	msum3 = _mm_hadd_ps (msum3, msum3);
	msum3 = _mm_hadd_ps (msum3, msum3);
	return  _mm_cvtss_f32 (msum3);
}

std::pair<int, int> Model::findEuclideanHyper(unsigned long long int observation){
	std::vector<std::pair<int, unsigned long long int>> euc;
	int it=0;
	std::cout<<"euc1\n";
	fflush(stdout);
	for(auto ele: temp_pair_vec){
		unsigned long long int temp_sub;
		if(ele.second > observation)
			temp_sub = ele.second - observation;
		else
			temp_sub = observation - ele.second;
		std::pair<int, unsigned long long int> tmp_pair(it, temp_sub);
		euc.push_back(tmp_pair);
		it++;
	}
	std::cout<<"euc2\n";
	fflush(stdout);
	std::sort(euc.begin(), euc.end(), [](auto &left, auto &right) {return left.second < right.second;});
	std::cout<<"euc3\n";
	fflush(stdout);
	std::vector<int> pos_index;
	for(int i=0; i<5000; ++i){
		pos_index.push_back(euc[i].first);
	};
	std::cout<<"euc4\n";
	fflush(stdout);
	int min_here = *std::min_element(pos_index.begin(), pos_index.end());
	int max_here = *std::max_element(pos_index.begin(), pos_index.end());
	std::cout<<"euc5\n";
	fflush(stdout);
       	return std::pair<int, int>(min_here, max_here);
}

std::vector<int> Model::findEuclideanHyper2(std::vector<float> observation){
	std::vector<std::pair<int, float>> euc;
	int it=0;
	std::cout<<"euc1\n";
	fflush(stdout);
	for(auto ele: temp_pair_vec2){
		float temp_sub;
		std::vector<float> inter = subtractVecs(observation, ele.second);
		float dist = dotVecs(inter, inter);

		std::pair<int, float> tmp_pair(it, dist);
		euc.push_back(tmp_pair);
		it++;
	}
	std::cout<<"euc2\n";
	fflush(stdout);
	std::sort(euc.begin(), euc.end(), [](auto &left, auto &right) {return left.second < right.second;});
	std::cout<<"euc3\n";
	fflush(stdout);
	std::vector<int> pos_index;
	for(int i=0; i<50000; ++i){
		pos_index.push_back(euc[i].first);
	};
       	return pos_index;
}

int Model::filterHyper(std::unordered_set<unsigned long long int> observation_vec, std::vector<float> orig_observation){
	std::vector<std::pair<float, int>> distance_index_arr;
	std::vector<unsigned long long int> index_actual_arr(2);
	int it=0;
	int num_obs_vec = observation_vec.size();

	int dim = 2*orig_observation.size();	
        int tot_num_features = (int) dim/2;  
       	int8_t *test_ele = new (std::align_val_t(32))int8_t[64]();


        for(int iter5=0; iter5<(int)dim/2; iter5++){
                        int8_t temp_c_lower = getQValue( code_val_dict_arr[2*iter5], reverse_code_val_dict_arr[2*iter5], orig_observation[iter5], reverse_vec[2*iter5],  2*iter5);
                        int8_t temp_c_upper = getQValue( code_val_dict_arr[2*iter5+1], reverse_code_val_dict_arr[2*iter5+1], orig_observation[iter5], reverse_vec[2*iter5+1], 2*iter5+1);
                        test_ele[iter5] = temp_c_lower;
                        test_ele[iter5+32] = temp_c_upper;
        }
	
	std::cout<<"before euc hyper\n";
	fflush(stdout);	
	std::vector<float> pooled_var_inv = getPooledVarInv();
	for(auto observation: observation_vec){
		auto upper = std::upper_bound(range_arr.begin(), range_arr.end(), observation);
		unsigned long long int index_actual;
		if (upper == range_arr.begin())
			index_actual = *upper;
		else
			index_actual = *(upper-1);
		index_actual_arr[it] = index_actual;
		it++;

	}
	sort( index_actual_arr.begin(), index_actual_arr.end() );
	index_actual_arr.erase( unique( index_actual_arr.begin(), index_actual_arr.end() ), index_actual_arr.end() );
	int num = 0;
	int num_indexes = index_actual_arr.size();
	
	std::pair<int, int> pos_res = findEuclideanHyper(*observation_vec.begin()); 
	//orig_observation.push_back(0);
	int obs_size = orig_observation.size();
	int real_index;
	double class_dict[2] = {0};
	double card_dict[2] = {0};
	float *obs_ptr = orig_observation.data();
	
	//for(int jj=0; jj<num_indexes; ++jj){
		int jj=0;
		auto index_actual = index_actual_arr[jj];
		//int lower_bound = range_loc_map[index_actual].first;
		//int upper_bound = range_loc_map[index_actual].second;
		
		int lower_bound = pos_res.first;
		int upper_bound = pos_res.second;
		std::cout<<"NUMBER of hyper: "<<upper_bound-lower_bound+1<<"\n";

		auto start3 = std::chrono::steady_clock::now();		
		__m256i srcvec_lower, srcvec_upper, lower, upper, result;
		bool res;
		srcvec_lower = _mm256_load_si256((__m256i*) &test_ele[0]);
		srcvec_upper = _mm256_load_si256((__m256i*) &test_ele[32]);
		int num_hyperplanes = upper_bound - lower_bound + 1;
		int flag;
        	int num_met = 0;
        	int met_list[num_hyperplanes+1];
		__m256i INF = _mm256_set1_epi8(0);
	
		for(int j=lower_bound*64; j<upper_bound*64; j+=64){
			flag = 0;

			lower = _mm256_load_si256((__m256i*) &hyper_quant[j]);
			upper = _mm256_load_si256((__m256i*) &hyper_quant[j+32]);
			result = range_compare(srcvec_lower, lower, upper,  srcvec_upper);
		
			res = (int)has_zero(result, INF);
			// class_dict[(int)class_arr_top[(int)(j/64)]]+=(1-res) ;
			if (res==0) {
				met_list[num_met] = (int)j/64;
				num_met++;
				//class_dict[(int)class_arr_top[(int)j/64]]+=1;
			}
		}
		std::cout<<"Num met: "<<num_met<<"\n";
		fflush(stdout);	
		auto end3 = std::chrono::steady_clock::now();
		double elapsed3 = std::chrono::duration<double, std::milli>(end3 - start3).count();
                int num_met2=0;
                const auto start = std::chrono::steady_clock::now();
                for(int jj=0; jj<num_met; ++jj){
                        flag = 0;
                        int j = met_list[jj];
                        for(int i=0; i<tot_num_features; ++i){
                                if(orig_observation[i] < hyperplanes_oned[j*64 + 2*i ]  || orig_observation[i] > hyperplanes_oned[j*64 + 2*i+1]){
                                        flag = 1;
                                        break;
                                }
                        }
                        if (flag == 0){
                                class_dict[(int)class_arr_top[j]]+=1;
                                card_dict[(int)class_arr_top[j]]+=card_arr_top[j];
                                num_met2++;
                        }

                }
		std::cout<<"Num met 2: "<<num_met2<<"\n";
		fflush(stdout);	
	//}
        const auto end2 = std::chrono::steady_clock::now();
        //std::cout<<std::chrono::duration <double, std::micro> (end2-start).count()<<",";
        int pred_val = 1;
	if(card_dict[0] > card_dict[1])
		pred_val=0;


	return pred_val;

}


//TODO: change observation to arr
int Model::filterNew(std::unordered_set<unsigned long long int> observation_vec, std::vector<float> orig_observation){
	
	std::vector<std::pair<float, int>> distance_index_arr;
	//	float struct_size = (float)sizeof(struct Poly) + ((float)sizeof(float)*2.0*(float)poly_eff[0].mean.size());

	std::vector<unsigned long long int> index_actual_arr(10);
	//std::vector<float>distance_arr;
	//distance_arr.reserve(1000);
	//float distance_arr[90000];
	struct DistInd distance_arr[60000];
	//__builtin_prefetch(distance_arr.data(), 0, 3);
	int it=0;
	int num_obs_vec = observation_vec.size();

	for(auto observation: observation_vec){
		auto upper = std::upper_bound(range_arr.begin(), range_arr.end(), observation);
		unsigned long long int index_actual;
		if (upper == range_arr.begin())
			index_actual = *upper;
		else
			index_actual = *(upper-1);
		index_actual_arr[it] = index_actual;
		it++;

	}
	struct DistInd smallest_dist[10] ;
	int largest_index =0;
	float largest_dist = 99999;
	sort( index_actual_arr.begin(), index_actual_arr.end() );
	index_actual_arr.erase( unique( index_actual_arr.begin(), index_actual_arr.end() ), index_actual_arr.end() );
	int pos = 0;
	int num = 0;
	int num_indexes = index_actual_arr.size();
	orig_observation.push_back(0);
	int obs_size = orig_observation.size();
	int real_index;

	float *obs_ptr = orig_observation.data();
	//for(int j=1; j<2; ++j){
	int a[10];	
	int j=1;
	auto index_actual = index_actual_arr[j];
	int lower_bound = range_loc_map[index_actual].first;
	int upper_bound = range_loc_map[index_actual].second;
	auto start3 = std::chrono::steady_clock::now();		
	int mean_size=0;
	for(int i=lower_bound; i<=upper_bound; ++i){
		//float curr_dist = polytopes[i].ll_score(orig_observation);
		real_index = i*mean_size;
		float curr_dist = ll_score_simd(&poly_eff_arr[real_index], obs_ptr, &poly_eff_arr[real_index+(int)mean_size/2], obs_size);
		//float* curr_dist = ll_score_simd2(&poly_eff_arr[lower_bound*mean_size], obs_ptr, upper_bound - lower_bound +1, obs_size); 
		distance_arr[num].dist = curr_dist;
		distance_arr[num].index = i;
		a[i%2] = std::abs((int)curr_dist) % 2;
		num++;
		//elapsed_arr3.push_back(elapsed3);
		__builtin_prefetch(&distance_arr[0], 0, 3);

		/*
		   index_actual = index_actual_arr[j+1];
		   lower_bound = range_loc_map[index_actual].first;
		//__builtin_prefetch(&poly_eff_arr[lower_bound*mean_size], 0, 3);
		real_index = upper_bound*mean_size;
		float curr_dist = ll_score_simd(&poly_eff_arr[real_index], obs_ptr, &poly_eff_arr[real_index+mean_size/2], obs_size);
		distance_arr[num].dist = curr_dist;
		distance_arr[num].index = upper_bound;
		*/
		//distance_arr[num] = curr_dist;

		}
		auto start4 = std::chrono::steady_clock::now();		
		for(int i=0; i<10; ++i){
			float curr_dist = distance_arr[i].dist;
			smallest_dist[i] = {99999, distance_arr[i].index};
			if (curr_dist > largest_dist){
				largest_dist = curr_dist;
				largest_index = num;
			}
		}
		for(int i=10; i<num; ++i){
			float curr_dist = distance_arr[i].dist;
			if (curr_dist < largest_dist){
				smallest_dist[largest_index] = {curr_dist, distance_arr[i].index};
				largest_dist = curr_dist;
				for(int k=0; k<10; ++k){
					if (smallest_dist[k].dist > largest_dist){
						largest_dist = smallest_dist[k].dist;
						largest_index = k;
					}
				}
			}
		}
		int cls[10]={0};
		for(int i=0; i<10; ++i){
			cls[classes[smallest_dist[i].index]]++;
		}
		int max_c = -1;
		int max_ind = -1;
		for (int i=0; i<10; ++i){
			if(cls[i] > max_c){
				max_c = cls[i];
				max_ind = i;
			}
		}
		auto end4 = std::chrono::steady_clock::now();
		double elapsed4 = std::chrono::duration<double, std::milli>(end4 - start4).count();
		elapsed_arr4.push_back(elapsed4);
		return max_ind;
}

int Model::findTopOne(std::vector<float> observation){
	std::map<int, std::pair<int, float>> min_dist_arr;
	std::map<int, std::vector<std::pair<int, float>> > euc_dist_arr;

	int num_threads = std::stoi(Config::getValue("numthreads"));
	int num_poly = polytopes.size();
#pragma omp parallel for num_threads(num_threads)
	for(int curr_index=0; curr_index<num_poly; ++curr_index){
		auto start = std::chrono::steady_clock::now();
		float curr_dist;
		if(Config::getValue("distance") == std::string("mahalanobis"))
			curr_dist = polytopes[curr_index].maha_dist(observation);
		else if(Config::getValue("distance") == std::string("eucmean"))
			curr_dist = polytopes[curr_index].euc_mean_dist(observation);
		else if(Config::getValue("distance") == std::string("eucvarmean"))
			curr_dist = polytopes[curr_index].euc_varmean_dist(observation);
		else
			curr_dist = polytopes[curr_index].ll_score(observation);

		int tn = polytopes[curr_index].getTreeNum();
		float euc_dist = polytopes[curr_index].euc_mean_dist(observation);
		if(euc_dist_arr.find(tn) == euc_dist_arr.end())
			euc_dist_arr[tn] = std::vector<std::pair<int, float>>();

		euc_dist_arr[tn].push_back(std::pair<int, float>(curr_index, euc_dist));
		if(min_dist_arr.find(tn) == min_dist_arr.end())
#pragma omp critical
			min_dist_arr[tn] = std::pair<int, float>(curr_index, curr_dist);
		else if(curr_dist < min_dist_arr[tn].second){
#pragma omp critical
			min_dist_arr[tn] = std::pair<int, float>(curr_index, curr_dist);
		}
		auto end = std::chrono::steady_clock::now();
		double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
		elapsed_arr.push_back(elapsed);
		//double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
		//std::cout<<"polytope "<<curr_index<<": "<<elapsed<<"\n";
	}
	std::map<int, int> tree_to_pos;
	for(auto &ele: euc_dist_arr){
		int to_search = min_dist_arr[ele.first].first;
		std::vector<std::pair<int, float>> distances = ele.second;
		std::sort(distances.begin(), distances.end(), [](auto &left, auto &right) {return left.second < right.second;});
		
		int iter = 0;
		for(auto x: distances){
			if (x.first == to_search){
				tree_to_pos[ele.first] = iter;
				break;
			}
			iter++;
		}
		//std::cout<<"smallest dist: "<<distances[0].second<<"\n";
		//std::cout<<"winning (ll) dist: "<<min_dist_arr[ele.first].second<<"\n";
		//std::cout<<"printing Winning polytope: \n";
		polytopes[min_dist_arr[ele.first].first].print();
	}
	int pred_class = 0;
	std::map<int, int> max_class_arr;
	for(auto ele: min_dist_arr){
		num_trees++;
		winning_indices.push_back(ele.second.first);
		int curr_class = polytopes[ele.second.first].getPredClass();
		if(max_class_arr.find(curr_class) == max_class_arr.end())
			max_class_arr[curr_class] = 0;
		max_class_arr[curr_class]++;
	}

	for(auto ele: tree_to_pos){
		std::cout<<ele.second<<", ";
	}
	std::cout<<"\n******************************************************\n";
	std::vector<std::pair<int, int>> v(max_class_arr.begin(), max_class_arr.end());

	auto result_it = std::max_element(v.begin(), v.end(), [](auto &left, auto &right) {return left.second < right.second;});
	return v[std::distance(v.begin(),result_it)].first;
}


int Model::findTopK(std::vector<float> observation){
	std::vector<std::pair<float, int>> distance_index_arr;
	int num_threads = std::stoi(Config::getValue("numthreads"));
	int num_poly = polytopes.size();
	/*for(auto poly: polytopes){
	  if(poly.getObs() != poly.getNumObs()){
	  std::cout<<"not equal"<<poly.getObs()<<", "<< poly.getNumObs()<<"\n";
	  std::vector<float>varv = poly.getVar();
	  for(auto ele: varv)
	  std::cout<<ele<<",";
	  std::cout<<"\n";
	  }
	  }*/
#pragma omp parallel for num_threads(num_threads)
	for(int curr_index=0; curr_index<num_poly; ++curr_index){
		float curr_dist;
		if(Config::getValue("distance") == std::string("mahalanobis"))
			curr_dist = polytopes[curr_index].maha_dist(observation);
		else if(Config::getValue("distance") == std::string("eucmean"))
			curr_dist = polytopes[curr_index].euc_mean_dist(observation);
		else if(Config::getValue("distance") == std::string("eucvarmean"))
			curr_dist = polytopes[curr_index].euc_varmean_dist(observation);
		else
			curr_dist = polytopes[curr_index].ll_score(observation);
#pragma omp critical
		distance_index_arr.push_back(std::pair<float, int>(curr_dist, curr_index));
	}
	int k = std::atoi(Config::getValue("topk").c_str());
	std::sort(distance_index_arr.begin(), distance_index_arr.end(), [](auto &left, auto &right) {return left.first < right.first;});
	//std::nth_element(distance_index_arr.begin(), distance_index_arr.begin()+k, distance_index_arr.end());
	//std::nth_element(distance_index_arr.begin(), distance_index_arr.begin()+k, distance_index_arr.end(), [](auto &left, auto &right) {return left.first > right.first;});
	int pred_class = 0;
	std::map<int, float> max_class_arr;
	for(int i=0; i<k; ++i){
		winning_indices.push_back(distance_index_arr[i].second);
		int curr_class = polytopes[distance_index_arr[i].second].getPredClass();
		if(max_class_arr.find(curr_class) == max_class_arr.end())
			max_class_arr[curr_class] = 0;
		max_class_arr[curr_class]++;
	}
	std::cout<<"\n*************************************************\n";
	std::vector<std::pair<int, float>> v(max_class_arr.begin(), max_class_arr.end());

	auto result_it = std::max_element(v.begin(), v.end(), [](auto &left, auto &right) {return left.second < right.second;});
	return v[std::distance(v.begin(),result_it)].first;
}


int Model::findTopKFilter(std::vector<float> observation, std::vector<float>weight){
	std::vector<std::pair<float, int>> distance_index_arr;
	std::vector<std::pair<float, int>> distance_index_arr2;
	int num_threads = std::stoi(Config::getValue("numthreads"));
	int num_poly = polytopes.size();
#pragma omp parallel for num_threads(num_threads)
	for(int curr_index=0; curr_index<num_poly; ++curr_index){
		float curr_dist;
		if(Config::getValue("distance") == std::string("mahalanobis"))
			curr_dist = polytopes[curr_index].maha_dist(observation);
		else if(Config::getValue("distance") == std::string("eucmean"))
			curr_dist = polytopes[curr_index].euc_mean_dist(observation);
		else if(Config::getValue("distance") == std::string("eucvarmean"))
			curr_dist = polytopes[curr_index].euc_varmean_dist(observation);
		else
			curr_dist = polytopes[curr_index].filter_dist(observation, weight );
#pragma omp critical
		distance_index_arr.push_back(std::pair<float, int>(curr_dist, curr_index));
	}
	int k = std::atoi(Config::getValue("topk").c_str());
	std::cout<<"k: "<<k<<"\n";
	std::sort(distance_index_arr.begin(), distance_index_arr.end(), [](auto &left, auto &right) {return left.first < right.first;});
	//std::nth_element(distance_index_arr.begin(), distance_index_arr.begin()+k, distance_index_arr.end());
	//std::nth_element(distance_index_arr.begin(), distance_index_arr.begin()+k, distance_index_arr.end(), [](auto &left, auto &right) {return left.first > right.first;});
	for(int i=0; i<k; ++i){
		int curr_index = distance_index_arr[i].second;
		float curr_dist = polytopes[curr_index].ll_score(observation );
		distance_index_arr2.push_back(std::pair<float, int>(curr_dist, curr_index));
	}


	std::sort(distance_index_arr2.begin(), distance_index_arr2.end(), [](auto &left, auto &right) {return left.first < right.first;});
	int pred_class = 0;
	std::map<int, int> max_class_arr;
	k = std::atoi(Config::getValue("topm").c_str());
	std::cout<<"m: "<<k<<"\n";
	for(int i=0; i<k; ++i){
		winning_indices.push_back(distance_index_arr2[i].second);
		int curr_class = polytopes[distance_index_arr2[i].second].getPredClass();
		int num_obs = polytopes[distance_index_arr2[i].second].getObs();
		int tree_num = polytopes[distance_index_arr2[i].second].getTreeNum();
		float num_obs_2 = polytopes[distance_index_arr2[i].second].getNumObs();
		std::vector<float> meanv = polytopes[distance_index_arr2[i].second].getMean();
		std::vector<float> varv = polytopes[distance_index_arr2[i].second].getVar();
		//std::cout<<tree_num<<","<<"\n";
		/*std::cout<<"distance: "<<distance_index_arr[i].first<<"\n";
		  std::cout<<"num obs: "<<num_obs_2<<"\n";
		  std::cout<<"old num obs: "<<num_obs<<"\n";
		  std::cout<<"printing mean vec: \n";
		  for(auto mm: meanv)
		  std::cout<<mm<<",";
		  std::cout<<"\n";
		  std::cout<<"printing var vec: \n";
		  for(auto vv: varv)
		  std::cout<<vv<<",";
		  std::cout<<"\n";
		  */
		if(max_class_arr.find(curr_class) == max_class_arr.end())
			max_class_arr[curr_class] = 0;

		max_class_arr[curr_class]++;
	}
	std::cout<<"\n*************************************************\n";
	std::vector<std::pair<int, int>> v(max_class_arr.begin(), max_class_arr.end());

	auto result_it = std::max_element(v.begin(), v.end(), [](auto &left, auto &right) {return left.second < right.second;});
	return v[std::distance(v.begin(),result_it)].first;
}

