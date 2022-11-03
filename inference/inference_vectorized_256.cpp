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
#include <bitset>

#define NUM_ELS 10000
#define SIMD_SIZE 256


#define _mm_cmpge_epu8(a, b) \
	_mm_cmpeq_epi8(_mm_max_epu8(a, b), a)

#define _mm256_cmpge_epu8(a, b) \
	_mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a)

#define _mm_cmple_epu8(a, b) _mm_cmpge_epu8(b, a)

#define _mm_cmpgt_epu8(a, b) \
	_mm_xor_si128(_mm_cmple_epu8(a, b), _mm_set1_epi8(-1))

#define _mm_cmplt_epu8(a, b) _mm_cmpgt_epu8(b, a)
#include <stdint.h>
#include <string.h>

float min_num;
float max_num;
int range_arr_size;
int neighbor_arr_size;
unsigned long long int range_arr_ptr[1000];
int farray[4];
using namespace std;
void print128_num(__m128i var)
{
	uint8_t val[16];
	memcpy(val, &var, sizeof(val));
	for(int i=0; i<16; ++i){
		std::cout<<std::bitset<8>(val[i])<<", ";
	}
	std::cout<<"\n";
	fflush(stdout);
	//printf("Numerical: %i %i %i %i %i %i %i %i \n", 
	//		val[0], val[1], val[2], val[3], val[4], val[5], 
	//		val[6], val[7]);
}

int serial_compare(__m128i var1, __m128i var2){
	uint8_t val1[16];
	memcpy(val1, &var1, sizeof(val1));
	uint8_t val2[16];
	memcpy(val2, &var2, sizeof(val2));
	for(int i=0; i<16; ++i)
	{
		if (val2[i] > val1[i])
			return 0;
	}
	return 1;
}
int serial_compare256(__m256i var1, __m256i var2){
	uint8_t val1[32];
	memcpy(val1, &var1, sizeof(val1));
	uint8_t val2[32];
	memcpy(val2, &var2, sizeof(val2));
	for(int i=0; i<32; ++i)
	{
		if (val2[i] > val1[i])
			return 0;
	}
	return 1;
}
int serial_has_zero(__m128i var){
	uint8_t val[16];
	memcpy(val, &var, sizeof(val));
	for(int i=0; i<16; ++i){
		if(val[i] == 0)
			return 0;
	}
	return 1;	
}

int serial_has_zero2(__m128i var){
	uint8_t val[16];
	memcpy(val, &var, sizeof(val));
	for(int i=0; i<16; ++i){
		if(val[i] == 0)
			return 0;
	}
	return 1;	
}

int order;


__m128i range_compare(__m128i a, __m128i b){//, __m128i c, __m128i d) {
	//return (_mm_cmpgt_epi8(a, b) & _mm_cmpgt_epi8(c, d) ) ;
	return _mm_cmpge_epu8(a, b);  //& _mm_cmpge_epu8(c, d); 
}



__m256i range_compare256(__m256i a, __m256i b) {
	return _mm256_cmpge_epu8(a, b);// & _mm256_cmpgt_epu8(c, d)  ;
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


std::vector<std::vector<int>> readHyperInt(std::string filename){
	std::fstream fin;
	fin.open(filename, std::ios::in);
	std::vector<std::string> row;
	std::string line, word, temp;
	std::vector<std::vector<int>> hyper;
	std::vector<int> temp_vector;
	int num_obs = 0;
	while(getline(fin, line, '\n')){
		std::istringstream templine(line);
		std::string data;
		while(getline(templine, data, ',')){
			temp_vector.push_back(std::atoi(data.c_str()));
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
		if(num_obs >  1000)
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

	int num_ele = SIMD_SIZE / 4;
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

uint8_t* quantizeSample(std::vector<float>&test, bool shift){
	uint8_t *test_ele = new (std::align_val_t(64))uint8_t[SIMD_SIZE/8]();
	int num_divisions = 14;
	float division_size = 1.0 / (float)num_divisions;
	int test_s = std::min((int)test.size(), (int)SIMD_SIZE/8);
	for(int i=0; i<test_s; i++){
		float hyper_ele = test[i];
		if(hyper_ele == 1)
			test_ele[i] = 15;
		else if (hyper_ele == 0)
			test_ele[i] = 0;
		else
			test_ele[i] = (uint8_t)(std::floor(hyper_ele/division_size)) + 1 ;

		if (shift==true)
			test_ele[i] = test_ele[i]<<4;
	}
	return test_ele;
}

uint8_t* readHyperBin(int n, std::string filename){
	std::fstream ifs;
	ifs.open(filename, std::ios::in|std::ios::binary);
	uint8_t* buffer = new (std::align_val_t(64)) uint8_t[n](); 
	ifs.read((char*)(buffer), sizeof(uint8_t) * n); // <---
	return buffer;
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


std::vector<unsigned long long int> getSubsetNeighbors(std::vector<float> observation  ,std::vector<unsigned long long int> range_arr, std::vector<std::vector<int>> neighbor_arr, std::vector<int> features, float max_num, float min_num){
	int tot_num_neighbors = 82; 
	int num_dims = features.size();
	int hilbert_cube[num_dims] = {0};
	float bin_size = pow(2, order) - 1;
	int curr_index = 0;
	for(auto iter: features){
		float item = observation[iter] ; 
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
	//std::string dirname = "/data/higgs_prune_transform/";
	//std::string dirname = "/data/higgs_prune_transform_new_4bitsimd/";
	//std::string testname = "test_HIGGS_t.csv";
	std::string testname = "test_hepmass_t.csv";

	std::string test_fname = dirname + testname;
	std::string hyper_fname = dirname + "hyper.txt";
	std::string lut_fname = dirname + "lut.txt";
	std::string hyperquant_lut_fname = dirname + "hyperquant_lut.bin";
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
	__m128i INF = _mm_setr_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
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
	float min_num = 0;
	float max_num = 1;
	order = (int)min_max[2];
	int num_classes=2;

	std::fstream f;
	f.open(std::string("trex_new/hepmass_latency.csv"), std::ios::out | std::ios::app);

	std::vector<std::vector<float>> test_list = readTest(test_fname, labels);
	std::cout<<"read testn";
	fflush(stdout);
	std::vector<int> class_list = readInt(classlist_fname);
	std::vector<int> card_list = readInt(cardlist_fname);
	std::cout<<"read class and card\n";
	fflush(stdout);
	std::vector<unsigned long long int> hilbert_vals = readLong(hilbert_fname);

	std::vector<unsigned long long int> range_arr = readLong(range_arr_fname);
	std::cout<<"read range aarr\n";
	fflush(stdout);
	std::map<unsigned long long int, std::pair<int, int>> range_loc_map = readMap(range_loc_map_fname);
	std::cout<<"read map\n";
	fflush(stdout);
	uint8_t* hyper_bin_lut = readHyperBin(hilbert_vals.size()*SIMD_SIZE/8, hyperquant_lut_fname);
	std::cout<<"read hyper bin\n";
	fflush(stdout);

	std::cout<<"read hyper bin lut\n";
	fflush(stdout);
	std::vector<std::vector<int>> neighbor_arr = readNeighborArray(neighbor_arr_fname);
	int dim = 2*test_list[0].size();
	int tot_num_features = SIMD_SIZE/8;
	int num_hyperplanes = hilbert_vals.size();


	int tno=0;
	int correct=0;
	float mse = 0.0;
	int tot=0;
	uint32_t mask;
	std::cout<<"before test\n";
	fflush(stdout);
	__m256i hyper_vec, qv_odd_shift, qv_evn_shift, test_vec_shifted;
	__m256i test_vec_old, test_vec;
	__m256i result, inter_vec;
	const __m256i hi_mask_08   = _mm256_setr_epi8(15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15);
	const __m256i hi_mask_08_2   = _mm256_setr_epi8(240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240);
	//const __m128i qv_odd_dirty, qv_odd_shift, qv_evn_shift;
	uint8_t flag;
	int index;
	uint8_t res;
	std::vector<float> test;
	for(auto &test_old : test_list){
		test.clear();
		//for(int i=0; i<24; ++i)
		for(int i=0; i<test_old.size(); ++i)
			test.emplace_back(test_old[features[i]]);
		int class_dict[2] = {0};
		auto start_full = std::chrono::high_resolution_clock::now();
		std::vector<unsigned long long int> buckets = getSubsetNeighbors(test_old,  range_arr, neighbor_arr, fourd_features, max_num, min_num);
		uint8_t *test_ele = quantizeSample(test, false);
		uint8_t *test_ele_shifted = quantizeSample(test, true);
		test_vec = _mm256_load_si256((__m256i*) &test_ele[0]);
		test_vec_shifted = _mm256_load_si256((__m256i*) &test_ele_shifted[0]);

		int num_met = 0;

		for(auto &bucket: buckets){	
			int lower_bound = range_loc_map[bucket].first;
			int upper_bound = range_loc_map[bucket].second;
			//std::cout<<lower_bound<<": "<<upper_bound<<"\n";
			//fflush(stdout);
			__builtin_prefetch(&hyper_bin_lut[lower_bound*index], 0, 3);
			for(int i=lower_bound; i<upper_bound; ++i){
				index = i*tot_num_features;
				hyper_vec = _mm256_load_si256((__m256i*) &hyper_bin_lut[index]);
				qv_odd_shift = _mm256_and_si256(hi_mask_08, hyper_vec);	
				qv_evn_shift = _mm256_and_si256(hi_mask_08_2, hyper_vec);	
				//int res = serial_compare256(qv_odd_shift, test_vec) & serial_compare256( test_vec_shifted, qv_evn_shift);
				inter_vec = _mm256_and_si256(range_compare256(qv_odd_shift, test_vec), range_compare256( test_vec_shifted, qv_evn_shift));	
				mask = _mm256_movemask_epi8(inter_vec);
				res = (mask == 0xffffffff);
				class_dict[(int)class_list[i]] += res*(card_list[i]);	
			}
		}
		auto end_full = std::chrono::high_resolution_clock::now();
		double elapsed_full = std::chrono::duration<double, std::micro>(end_full - start_full).count();
		std::cout<<elapsed_full<<"\n";
		fflush(stdout);
		int predicted = 0;

		f<<elapsed_full<<",";
		if(class_dict[1] > class_dict[0])
			predicted = 1;
		//if(num_met > 0)
		//	std::cout<<"num met: "<<num_met<<"\n";	

		if (predicted == labels[tot])
			correct++;
		tot+=1;
		//std::cout<<"num correct; "<<correct<<"\n";
		//std::cout<<"total ; "<<tot<<"\n";
		//fflush(stdout);

	}
	f.close();
	std::cout<<"correct: "<<correct<<"\n";
	std::cout<<"tot: "<<tot<<"\n";
	return correct;
}


