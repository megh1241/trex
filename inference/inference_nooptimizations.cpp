//
// Test.cpp
//
// This is a direct port of the C version of the RTree test program.
//

#include "src/hilbert.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <immintrin.h>
#include <x86intrin.h>
#include <stdio.h>
#include <new>
#include <math.h>
#include <map>
#include <stdio.h>
#include <algorithm>
#include <unordered_set>

#define NUM_ELS 10000
#define SIMD_WIDTH 128
int order;
float min_num;
float max_num;
int range_arr_size;
int neighbor_arr_size;
unsigned long long int range_arr_ptr[1000];
int farray[4];
using namespace std;

bool has_zero(__m128i x, __m128i INF){
	//const __m128 SIGN_MASK = _mm128_set1_ps(-0.0);
	x = _mm_cmpeq_epi8(x, INF);
	return _mm_movemask_epi8(x) != 0;
}

bool has_zero256(__m256i x, __m256i INF2){

	//const __m128 SIGN_MASK = _mm128_set1_ps(-0.0);
	x = _mm256_cmpeq_epi8(x, INF2);
	return _mm256_movemask_epi8(x) != 0;
}

__m128i range_compare(__m128i a, __m128i b, __m128i c, __m128i d) {
	return (_mm_cmpgt_epi8(a, b) & _mm_cmpgt_epi8(c, d) ) ;
	//return (_mm_cmpgt_epi8(a, b) | _mm_cmpeq_epi8(a, b))  & (_mm_cmpgt_epi8(c, d) | _mm_cmpeq_epi8(c, d) ) ;
}

__m256i range_compare256(__m256i a, __m256i b, __m256i c, __m256i d) {
	return (_mm256_cmpgt_epi8(a, b) & _mm256_cmpgt_epi8(c, d) ) ;
	//return (_mm_cmpgt_epi8(a, b) | _mm_cmpeq_epi8(a, b))  & (_mm_cmpgt_epi8(c, d) | _mm_cmpeq_epi8(c, d) ) ;
}

std::vector<std::vector<float>> readHyper(std::string filename){
	std::fstream fin;
	fin.open(filename, std::ios::in);
	std::vector<std::string> row;
	std::string line, word, temp;
	std::vector<std::vector<float>> hyper;
	std::vector<float> temp_vector;
	int num_obs = 0;
	while(getline(fin, line, '\n')){
		std::istringstream templine(line);
		std::string data;
		while(getline(templine, data, ',')){
			temp_vector.push_back(std::atof(data.c_str()));
		}
		hyper.push_back(temp_vector);
		temp_vector.clear();
		num_obs++;
	}
	fin.close();
	return hyper;
}


std::vector<std::vector<float>> readTest(std::string filename, std::vector<float>&labels){
	std::fstream fin;
	fin.open(filename, std::ios::in);
	std::vector<std::string> row;
	std::string line, word, temp;
	std::vector<float> temp_vector;
	std::vector<std::vector<float>> test_data;
	int num_obs = 0;
	while(getline(fin, line, '\n')){
		std::istringstream templine(line);
		std::string data;
		while(getline(templine, data, ',')){
			temp_vector.push_back(std::atof(data.c_str()));
		}
		int siz = temp_vector.size();
		float last_ele = (float)(temp_vector.at(siz-1));
		labels.push_back(last_ele);
		temp_vector.pop_back();
		test_data.push_back(temp_vector);
		temp_vector.clear();
		num_obs++;
		if(num_obs>1000)
		break;
	}
	fin.close();
	return test_data;
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

std::vector<int> readFeatures(std::string filename){
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


std::vector<int> readInt(std::string filename){
	std::ifstream file_handler(filename);

	// use a std::vector to store your items.  It handles memory allocation automatically.
	std::vector<int> arr;
	int number;
	while (file_handler>>number) {
		arr.push_back(number);
		file_handler.ignore(std::numeric_limits<std::streamsize>::max(), ',');
	}
	return arr;
}

std::vector<float> readFloat(std::string filename){
	std::ifstream file_handler(filename);

	// use a std::vector to store your items.  It handles memory allocation automatically.
	std::vector<float> arr;
	int number;
	while (file_handler>>number) {
		arr.push_back(number);
		file_handler.ignore(std::numeric_limits<std::streamsize>::max(), ',');
	}
	return arr;
}


std::vector<int8_t> readInt8(std::string filename){
	std::ifstream file_handler(filename);

	// use a std::vector to store your items.  It handles memory allocation automatically.
	std::vector<int8_t> arr;
	char number;
	while (file_handler>>number) {
		arr.push_back((int8_t)number);
		file_handler.ignore(std::numeric_limits<std::streamsize>::max(), ',');
	}
	return arr;
}

int8_t* readHyperQuant(std::string filename, int num_hyperplanes){
	std::ifstream file_handler(filename);

	int num_ele = SIMD_WIDTH / 4;
	// use a std::vector to store your items.  It handles memory allocation automatically.
	int8_t* hyper_quant = new (std::align_val_t((int)num_ele/2))int8_t [num_hyperplanes * num_ele]();
	int number;
	int iter=0;
	while (file_handler>>number) {
		hyper_quant[iter] =  (int8_t)number;
		file_handler.ignore(std::numeric_limits<std::streamsize>::max(), ',');
		iter++;
	}
	return hyper_quant;
}

std::vector<float> readMinMax(std::string filename){
	std::ifstream file_handler(filename);
	float min_num, max_num;
	char temp;
	int the_order;
	std::vector<float> to_return;
	file_handler >> min_num;
	file_handler >> temp;
	file_handler >> max_num;
	file_handler >> temp;
	file_handler >> the_order;
	to_return.push_back(min_num);
	to_return.push_back(max_num);
	to_return.push_back(the_order);
	return to_return;
}


std::vector<std::map<float, int8_t>> readCodeDict(std::string filename){
	std::ifstream file_handler(filename);
	int num_cols;
	file_handler>>num_cols;
	std::vector<std::map<float, int8_t>> result_arr;
	for(int i=0; i<num_cols; ++i){
		std::map<float, int8_t> temp_map;
		int num_eles;
		file_handler>>num_eles;
		for(int j=0; j<num_eles; ++j){
			float k;
			int v;
			file_handler>>k;
			file_handler>>v;
			//std::cout<<k<<": "<<v<<"\n";
			temp_map[k] = (int8_t)v;
		}
		result_arr.push_back(temp_map);
	}
	return result_arr;
}

std::vector<unsigned long long int> readLong(std::string filename){
	std::ifstream file_handler(filename);

	// use a std::vector to store your items.  It handles memory allocation automatically.
	std::vector<unsigned long long int> arr;
	unsigned long long int number;
	while (file_handler>>number) {
		arr.push_back(number);
		file_handler.ignore(std::numeric_limits<std::streamsize>::max(), ',');
	}
	return arr;
}
std::map<unsigned long long int, std::pair<int, int>> readMap(std::string filename){
	std::ifstream file_handler(filename);
	int num_items;
	file_handler>>num_items;
	std::map<unsigned long long int, std::pair<int, int>> result_map;
	for(int i=0; i<num_items; ++i){
		unsigned long long int key;
		int beg_loc;
		int end_loc;
		file_handler>>key;
		file_handler>>beg_loc;
		file_handler>>end_loc;
		result_map[key] = std::pair<int, int>(beg_loc, end_loc);
	}
	return result_map;
}

std::vector<std::vector<int>> readNeighborArray(std::string filename){
	std::fstream fin;
	fin.open(filename, std::ios::in);
	std::vector<std::string> row;
	std::string line, word, temp;
	std::vector<std::vector<int>> neighbor_arr;
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
	return neighbor_arr;
}

double getAccuracy(const std::vector<int> &predicted, const std::vector<int> &labels){
	int wrong = 0;
	int siz = predicted.size();
	for(int i=0; i<siz; ++i){
		if(labels[i] != predicted[i]){
			wrong++;
		}
	}
	return (double)(siz - wrong) / (double)(siz);
}

int8_t getQValue(std::map<float, int8_t> reverse_dict, float hyper_ele, int i_dim){


	if(hyper_ele <= -99999)
		return (int8_t)-128;
	if(hyper_ele >= 99999)
		return (int8_t)127;
	std::vector<std::pair<float, int8_t>> reverse_dict_vector(reverse_dict.size());

	std::copy(reverse_dict.begin(), reverse_dict.end(), reverse_dict_vector.begin());
	int found=0;	
	if(i_dim%2){
		for(int i=0; i<reverse_dict_vector.size()-2; ++i){
			if(reverse_dict_vector[i].first <= hyper_ele && reverse_dict_vector[i+1].first > hyper_ele)
				return reverse_dict_vector[i].second;
		}
	}
	else{
		int siz = reverse_dict_vector.size();
		for(int i=0; i<siz-2; ++i){
			if(reverse_dict_vector[i].first <= hyper_ele && reverse_dict_vector[i+1].first > hyper_ele){
				int8_t final_to_return = reverse_dict_vector[i+1].second;
				return final_to_return;
			}
		}
	}
	return (int8_t)reverse_dict_vector[reverse_dict_vector.size()-1].second;
}

int8_t* quantizeSample_backup(std::vector<std::map<float, int8_t>> &reverse_code_val_dict_arr, std::vector<float>test){
	int num_ele = SIMD_WIDTH/4;
	int tot_num_features = std::min(num_ele/2, (int)test.size());
	int8_t *test_ele = new (std::align_val_t( (int) num_ele/2))int8_t[num_ele]();

	for(int iter5=0; iter5<tot_num_features; iter5++){
		int8_t temp_c_lower = getQValue(reverse_code_val_dict_arr[2*iter5], test[iter5], 2*iter5);
		int8_t temp_c_upper = getQValue(reverse_code_val_dict_arr[2*iter5+1], test[iter5], 2*iter5+1);
		test_ele[iter5] = temp_c_lower;
		test_ele[iter5+num_ele/2] = temp_c_upper;
	}
	return test_ele;

}

int8_t* quantizeSample(std::vector<std::map<float, int8_t>> &reverse_code_val_dict_arr, std::vector<float>test){
	int num_ele = SIMD_WIDTH/4;
	int tot_num_features = std::min(num_ele/2, (int)test.size());
	int8_t *test_ele = new (std::align_val_t( (int) num_ele/2))int8_t[num_ele]();
	int num_divisions = 252;
	float division_size = 1.0 / (float)num_divisions;

	for(int i=0; i<tot_num_features; i++){
		float hyper_ele = test[i];
		test_ele[i] = (int8_t)(std::floor(hyper_ele/division_size)-128+1);
		test_ele[i+num_ele/2] = (int8_t)(std::floor(hyper_ele/division_size)-128+1);
	}
	return test_ele;
}



unsigned long long int hilbertDecode(std::vector<float>test, std::vector<float> pooled, std::vector<int> features, float min_num, float max_num){
	unsigned long long int hilbert_cube[3];
	float bin_size = pow(2, order) - 1;
	int it1=0;
	for(auto f: features){
		float item = test[f] * pooled[f];
		item = std::max(min_num, item);
		item = std::min(max_num, item);
		//TODO: convert to hilbert index
		float conv = bin_size *(float) (item - min_num) / (float)(max_num - min_num);
		unsigned long long int to_convert = (unsigned long long int) ( (float)bin_size *(float) (item - min_num) / (float)(max_num - min_num));
		hilbert_cube[it1] = to_convert;
		it1++;
	}	
	unsigned long long int hilbert_index = hilbert_c2i(3, order, hilbert_cube);
	return hilbert_index;
}


std::pair<int, int> getSubset(unsigned long long int observation, std::vector<unsigned long long int> hilbert_vals){
	std::vector<std::pair<int, unsigned long long int>> euc;
	int it=0;
	for(auto ele: hilbert_vals){
		unsigned long long int temp_sub;
		if(ele > observation)
			temp_sub = ele - observation;
		else
			temp_sub = observation - ele;
		std::pair<int, unsigned long long int> tmp_pair(it, temp_sub);
		euc.push_back(tmp_pair);
		it++;
	}
	std::sort(euc.begin(), euc.end(), [](auto &left, auto &right) {return left.second < right.second;});
	std::vector<int> pos_index;
	int tot_num_hyper = hilbert_vals.size();
	for(int i=0; i<NUM_ELS; ++i){
		pos_index.push_back(euc[i].first);
	}
	int min_here = *std::min_element(pos_index.begin(), pos_index.end());
	int max_here = *std::max_element(pos_index.begin(), pos_index.end());
	return std::pair<int, int>(min_here, max_here);
}


std::vector<unsigned long long int> getSubsetNeighbors(std::vector<float> observation, std::vector<float> pooled_var_inv, std::vector<unsigned long long int> range_arr, std::vector<std::vector<int>> neighbor_arr, std::vector<int> features, float max_num, float min_num){
	int tot_num_neighbors = 82; 
	int num_dims = features.size();
	int hilbert_cube[num_dims] = {0};
	float bin_size = pow(2, order) - 1;
	int curr_index = 0;
	for(auto iter: features){
		float item = observation[iter] * pooled_var_inv[iter];
		int to_convert = (int) (bin_size * (item - min_num) / (max_num - min_num));
		hilbert_cube[curr_index] = to_convert;
		curr_index++;
	}
	unsigned long long int hilbert_indices_all_neighbors[tot_num_neighbors];
	int neighbor_id=0;
	for(auto neighbor : neighbor_arr){
		unsigned long long int cube[num_dims];
		int flag = 1;
		for(int i=0; i<num_dims; ++i){
			cube[i] = (unsigned long long int)(hilbert_cube[i] + neighbor[i]);
			if (cube[i] < 0 || cube[i] >= bin_size){
				flag = -1;
				break;
			}
		}
		if(flag > 0){
			hilbert_indices_all_neighbors[neighbor_id] = hilbert_c2i(3, order, cube);
			neighbor_id++;
		}
	}
	std::unordered_set<unsigned long long int> hilbert_set( hilbert_indices_all_neighbors, hilbert_indices_all_neighbors+neighbor_id );

	int it=0;
	std::vector<unsigned long long int> index_actual_arr(tot_num_neighbors);
	for(auto obs: hilbert_set){
		auto upper = std::upper_bound(range_arr.begin(), range_arr.end(), obs);
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
	return index_actual_arr;
}

int main()
{
	std::string dirname = "/data/hepmass_prune_transform_new/";
	std::string testname = "test_hepmass_t.csv";

	std::string test_fname = dirname + testname;
	std::string hyper_fname = dirname + "hyper.txt";
	std::string hyper_quant_fname = dirname  + "hyper_quant.txt";
	std::string pooled_fname = dirname + "pooled.csv";
	std::string classlist_fname = dirname + "classes.txt";
	std::string cardlist_fname = dirname + "card.txt";
	std::string hilbert_fname = dirname + "hilbert_vals.txt";
	std::string code_fname = dirname + "code_val_dict.txt";
	std::string maxmin_fname = dirname + "maxmin_order.txt";
	std::string features_fname = dirname + "features.csv";
	std::string range_arr_fname = dirname + "range_arr.txt";
	std::string range_loc_map_fname = dirname + "range_loc_map.txt";
	std::string neighbor_arr_fname = dirname + "neighbors_all.csv";
	std::cout<<"statt\n";
	fflush(stdout);
	std::vector<int> features = readFeatures(features_fname);

	std::vector<int> fourd_features = std::vector<int>(features.begin(), features.begin()+3);
	std::cout<<"printing features!\n";
	std::cout<<"*****************************************\n";
	for(auto ele: fourd_features)
		std::cout<<ele<<": ";
	std::cout<<"\n";
	std::cout<<"*****************************************\n";
	fflush(stdout);
	std::vector<float> labels;

	std::vector<float> min_max = readMinMax(maxmin_fname);
	float min_num = min_max[0];
	float max_num = min_max[1];
	order = (int)min_max[2];
	int num_classes=2;

	std::vector<float> pooled = readBbox(pooled_fname);

	std::vector<float> inv_sqrt_pooled;
	std::cout<<"printing pooled var inv: \n";
	for(auto ele: pooled){
		std::cout<<1.0 / sqrt(ele)<<",";
		inv_sqrt_pooled.push_back(1.0 );
		//inv_sqrt_pooled.push_back(1.0 / sqrt(ele));

	}
	std::cout<<"\n";

	std::cout<<"read pooled\n";
	fflush(stdout);

	std::vector<std::vector<float>> test_list = readTest(test_fname, labels);
	std::cout<<"read testn";
	fflush(stdout);
	std::vector<float> class_list = readFloat(classlist_fname);
	std::vector<int> card_list = readInt(cardlist_fname);
	std::cout<<"read class and card\n";
	fflush(stdout);
	std::vector<unsigned long long int> hilbert_vals = readLong(hilbert_fname);
	std::cout<<"read hilbert vals\n";
	fflush(stdout);

	std::vector<std::map<float, int8_t>> reverse_code_map_arr; 
	//std::vector<std::map<float, int8_t>> reverse_code_map_arr = readCodeDict(code_fname);
	std::cout<<"read code dict\n";
	fflush(stdout);

	std::vector<std::vector<float>> hyperplanes = readHyper(hyper_fname);

	std::vector<unsigned long long int> range_arr = readLong(range_arr_fname);
	std::map<unsigned long long int, std::pair<int, int>> range_loc_map = readMap(range_loc_map_fname);

	std::vector<std::vector<int>> neighbor_arr = readNeighborArray(neighbor_arr_fname);
	std::cout<<"read hyper\n";
	fflush(stdout);
	int dim = hyperplanes[0].size();
	int tot_num_features = dim/2;
	int num_hyperplanes = hyperplanes.size();
	std::cout<<"NUM hyperplanes: "<<num_hyperplanes<<"\n";
	int num_ele = SIMD_WIDTH/4;
	float *hyperplane_oned = new float [num_hyperplanes*dim]();
	for (int j=0; j<num_hyperplanes; ++j){
		for(int i=0; i<dim; ++i){
			hyperplane_oned[j*dim + i] = hyperplanes[j][i];
		}
	}

	int8_t *hyper_quant = readHyperQuant(hyper_quant_fname, num_hyperplanes);

	__m128i INF = _mm_set1_epi8(0);
	__m256i INF_256 = _mm256_set1_epi8(0);
	int tno=0;
	int correct=0;
	float mse = 0.0;
	int tot=0;
	int met_list[4000] = {0};
 std::fstream f;
        f.open(std::string("trex_new/noopt_hepmass.csv"), std::ios::out | std::ios::app);


	for(auto &test : test_list){
		int class_dict[2] = {0};
		unsigned long long int hilbert_test = hilbertDecode(test, inv_sqrt_pooled, fourd_features, min_num, max_num);
		//std::pair<int, int> location = getSubset(hilbert_test, hilbert_vals);
		std::vector<unsigned long long int> buckets = getSubsetNeighbors(test, inv_sqrt_pooled, range_arr, neighbor_arr, fourd_features, max_num, min_num);

		//int8_t *test_ele = quantizeSample(reverse_code_map_arr, test);
		int num_met2=0;
		int num_met = 0;
		//float predicted = 0.0;
		auto start_full = std::chrono::high_resolution_clock::now();
		for(auto bucket: buckets){	
			int lower_bound = range_loc_map[bucket].first;
			int upper_bound = range_loc_map[bucket].second;

			for(int j=lower_bound; j<upper_bound; ++j){
				int flag = 0;
				for(int i=0; i<tot_num_features; ++i){
					if(test[i] < hyperplane_oned[j*dim + 2*i ]  || test[i] > hyperplane_oned[j*dim + 2*i+1]){
						flag = 1;
						break;
					}

				}
				if (flag == 0){
					class_dict[(int)class_list[j]] += (int)card_list[j];
					num_met2++;

				}
			}
		}
		auto end_full = std::chrono::high_resolution_clock::now();
		int predicted = 0;
		if(class_dict[1] > class_dict[0])
			predicted = 1;
		//std::cout<<"num met: "<<num_met2<<"\n";	
		double elapsed_full = std::chrono::duration<double, std::micro>(end_full - start_full).count();
		//std::cout<<elapsed_full<<"\n";
		//fflush(stdout);

		f<<elapsed_full<<",";
		/*if(num_met2>0)
		  predicted = predicted / (float)num_met2;	
		  else
		  predicted = labels[tot];
		  */
		//std::cout<<"predicted: "<<predicted<<"\n";
		//std::cout<<"gt: "<<labels[tot]<<"\n";
		/*float deltaa = predicted - labels[tot];
		  mse += (deltaa*deltaa);
		  float mse_tot = (float)mse/(float)tot;
		  std::cout<<"mse: "<<mse_tot<<"\n";
		  */
		if (predicted == labels[tot])
			correct++;
		tot+=1;

	}
	std::cout<<"correct: "<<correct<<"\n";
	std::cout<<"tot: "<<tot<<"\n";
	//float mse_tot = (float)mse/(float)tot;
	//std::cout<<"mse: "<<mse_tot<<"\n";

}


