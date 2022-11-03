#include "model.h"
#include "polytope.h"
#include "config.h"
#include "../rapidjson/include/rapidjson/document.h"
#include "../rapidjson/include/rapidjson/writer.h"
#include "../rapidjson/include/rapidjson/stringbuffer.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <iostream>
#include <chrono>
#include <assert.h>
#include <algorithm>
#include "json_reader.h"
#include "utils.h"
#include<cmath>
#include <sstream>
#include <string>
using namespace rapidjson;

/*void normalizeObs(std::vector<float> input){
  std::ifstream in(filename);
  std::string contents((std::istreambuf_iterator<char>(in)),
  std::istreambuf_iterator<char>());
  const char *json = contents.data();
  auto start = std::chrono::steady_clock::now();
  Document d;
  d.Parse(json);
  assert(d.IsObject());
  const Value& maps = d;
  SizeType num_polytopes = d.Size();
  std::vector<Polytope> polytopes;
  int count_one=0;
  int total=0;
  int numrep = 0;
  std::map<std::vector<float>, std::vector<float>> mean_dict;
  std::map<std::vector<float>, std::vector<float>> var_dict;


  const Value& scale_mean = maps[0];
  const Value& scale_var = maps[1];
  }*/

void writePoly(std::vector<Polytope> polytopes){

	std::vector<std::vector<float>> samples;
	float max_obs = 0;
	for(auto poly: polytopes){
		float log_obs = 2*log(poly.getObs());
		if(log_obs > max_obs)
			max_obs = log_obs;
	}	
	
	std::string fname = Config::getValue("writefname");
	std::fstream fout;
	fout.open(fname,  std::ios::out );
	std::vector<float>pooled_var_ext = polytopes[0].getPool();
	for(int i=0; i<pooled_var_ext.size(); ++i){
		pooled_var_ext[i] = sqrt(pooled_var_ext[i]);
	}
	pooled_var_ext.push_back(1);
	
	for(auto poly: polytopes){
		std::vector<float> external  = poly.getMean();
		external.push_back(sqrt(-2*log(poly.getObs()) + max_obs));
		std::vector<float> meanvar = mulVecs(external, pooled_var_ext);
		for(auto ele: meanvar){
			fout<<ele<<",";
		}
		fout<<"\n";
	}
	fout.close();
	
	fname = Config::getValue("writefname2");
	fout.open(fname,  std::ios::out );
	for(auto poly: polytopes){
		std::vector<float> mean  = poly.getMean();
		for(auto j: mean)
			fout<<j<<",";
		std::vector<float> var  = poly.getVar();
		for(auto j: var)
			fout<<j<<",";
		fout<<poly.getDetTerm()<<",";
		fout<<poly.getObs()<<",";
		fout<<poly.getPredClass()<<",";
		fout<<"\n";
	}
	fout.close();
}


void writeCsv(std::vector<Polytope> polytopes){
	std::vector<std::vector<float>> oversample;
	std::vector<std::vector<float>> undersample;

	for(auto poly: polytopes){
		std::vector<float>temp;
		std::vector<float> mean = poly.getMean();
		std::vector<float> var = poly.getVar();
		int tree_num = poly.getTreeNum();
		float det_term = poly.getDetTerm();
		float num_obs = poly.getObs();
		float class_t = (float)poly.getPredClass();
		temp.push_back(class_t);
		temp.push_back(tree_num);
		temp.push_back(det_term);
		temp.push_back(num_obs);
		for(auto i: mean)
			temp.push_back(i);
		for(auto i: var)
			temp.push_back(i);
		if(poly.getObs() < 18){
			undersample.push_back(temp);	
		}else{
			oversample.push_back(temp);
		}
	}
	std::string fname_undersample = Config::getValue("undersample");
	std::string fname_oversample= Config::getValue("oversample");
                        //Write the nodes
        std::fstream fout;
        fout.open(fname_undersample,  std::ios::out );
        for(auto temp: undersample){
        	for(auto ele: temp){
			fout<<ele<<",";
                }
		fout<<"\n";
        }
        fout.close();
        
	fout.open(fname_oversample,  std::ios::out );
        for(auto temp: oversample){
        	for(auto ele: temp){
			fout<<ele<<",";
                }
		fout<<"\n";
        }
        fout.close();



}

void imputeVar(std::vector<Polytope> &polytopes){
        std::vector<std::vector<float>> pooled_var;
        std::vector<std::vector<float>> inverse_pooled_var;
        std::vector<float> temp;

        int n=0;
        int num_classes = 2;

        for(int i=0; i<num_classes; ++i) {
                pooled_var.push_back(temp);
                inverse_pooled_var.push_back(temp);
        }
        std::vector<float> n_vec(num_classes, 0);

        std::vector<int> indices_replace;
        std::vector<int> indices_one;
        int iter = 0;
        int siz = polytopes[0].getVar().size();
        std::vector<float> tot_var(siz, 0);
        std::vector<float> inverse_tot_var;
	int tot_n = 0;
        float tot_det_term = 0;
        for(auto poly: polytopes){
                std::vector<float> varv = poly.getVar();
                int pred_class = poly.getPredClass();
		int nobs = poly.getObs() - 1;
                if((poly.getObs() >= 18) && std::find(varv.begin(), varv.end(), 0) == varv.end()){
                        if(pooled_var[pred_class].size() == 0){
                                for(int i=0; i<varv.size(); ++i){
                                        pooled_var[pred_class].push_back((float)1/varv[i]);
                                }

                        }
                        else{
                                for(int i=0; i<varv.size(); ++i){
                                        pooled_var[pred_class][i] += ((float)1/varv[i]);
                                }
                        }
                        n_vec[pred_class]++;
                        for(int i=0; i<siz; ++i){
                                tot_var[i] += ((float)(1)/varv[i]);
			}
			tot_n+=nobs;
                        n++;
                }else{
                        if (poly.getObs() < 18)
                                indices_replace.push_back(iter);
                        else
                                indices_one.push_back(iter);
                }
                iter++;
        }

        float det_term=0;
        std::vector<float> det_term_vec;

        for(int i=0; i<num_classes; ++i){
                for(auto &ele: pooled_var[i]){
                        ele = ele/(float)n_vec[i];
                        det_term += log(sqrt(ele));
                        inverse_pooled_var[i].push_back((float)1/ele);
                }
                det_term_vec.push_back(det_term);
                det_term = 0;
        }
        /*
        float det_term2=0;
        for(auto &ele: pooled_var[1]){
                ele = ele/(float)n_vec[1];
                det_term2 += log(sqrt(ele));
                inverse_pooled_var[1].push_back((float)1/ele);
        }
        */
        for(auto &ele: tot_var)
        {
                ele  = ele/(float)n;
                tot_det_term += log(sqrt(ele));
                inverse_tot_var.push_back((float)1/ele);
        }


        //for(int i=0; i< polytopes.size(); ++i){
        for(auto i: indices_replace){
                int pred_class = polytopes[i].getPredClass();
                polytopes[i].setVar(inverse_tot_var);
                polytopes[i].setDetTerm(tot_det_term);
                //polytopes[i].setVar(inverse_pooled_var[pred_class]);
                //polytopes[i].setDetTerm(det_term_vec[pred_class]);
        }

        for(auto i: indices_one){
                std::vector<float> varv = polytopes[i].getVar();
                float det_term = polytopes[i].getDetTerm();
                int siz = varv.size();
                int pred_class = polytopes[i].getPredClass();
                for(int j=0; j<siz; ++j){
                        if(varv[j] == 0){
                                varv[j] = inverse_tot_var[j];
                                det_term += log(sqrt(tot_var[j]));
                                //varv[j] = inverse_pooled_var[pred_class][j];
                                //det_term += log(sqrt(pooled_var[pred_class][j]));
                        }
                }
                polytopes[i].setVar(varv);
                polytopes[i].setDetTerm(det_term);
        }
        polytopes[0].setPool(inverse_tot_var);
        polytopes[0].setPoold(tot_det_term);


}
/*
            temp_list = []
            temp_list.extend( np.mean(leaf_obs_map[leaf_id], axis=0).tolist() )
            temp_list.extend( inverse(var).tolist() )
            temp_list.append( float(term1) )
            temp_list.append( num_obs )
            temp_list.append(leaf_class_map[leaf_id])
*/

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

std::vector<float> readBbox1(std::string filename){
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



std::vector<std::vector<float>> readTest(std::string filename, std::vector<int>&labels){
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
                int last_ele = (int)(temp_vector.at(siz-1));
                labels.push_back(last_ele);
                temp_vector.pop_back();
                test_data.push_back(temp_vector);
                temp_vector.clear();
                num_obs++;
        }
        fin.close();
        return test_data;
}
std::vector<Polytope> JSONReader::readPolytopesCSV(){
	std::vector<Polytope> polytopes;
	std::string filename = Config::getValue("polymodelfilename");
	std::string hyper_filename = Config::getValue("hyperfilename");
    	std::fstream fin;
    	fin.open(filename, std::ios::in);
    	std::vector<std::string> row;
    	std::string line, word, temp;
    	int num_obs = 0;
        std::vector<float> tot_var;
        std::vector<float> inverse_tot_var;
	std::cout<<"done0\n";
fflush(stdout);

/*
 *             temp_list.extend( np.mean(leaf_obs_map[leaf_id], axis=0).tolist() )
            temp_list.extend( inverse(var).tolist() )
            temp_list.append( float(term1) )
            temp_list.append( num_obs )
            temp_list.append(leaf_class_map[leaf_id]
 */

	while(getline(fin, line, '\n')){
        	std::istringstream templine(line);
        	std::string data;
    		std::vector<float> temp_vector;
        	while(getline(templine, data, ',')){
            		temp_vector.push_back(std::atof(data.c_str()));
        	}
		int siz = temp_vector.size();
		int class_num = (int)temp_vector[siz-1];
		temp_vector.pop_back();
		int num_o = (int)temp_vector[temp_vector.size()-1];
		temp_vector.pop_back();
		float det_term = temp_vector[temp_vector.size()-1];
		temp_vector.pop_back();
		int mid = temp_vector.size() / 2;
		std::vector<float>mean_vec = std::vector<float>(temp_vector.begin(), temp_vector.begin()+mid);
		std::vector<float>var_vec = std::vector<float>(temp_vector.begin()+mid, temp_vector.end());

		siz = var_vec.size();
		if(tot_var.size() == 0){
			for(int i=0; i<siz; ++i){
				tot_var.push_back((float)(1)/var_vec[i]);
			}
		}else{
			for(int i=0; i<siz; ++i){
                     		tot_var[i] += ((float)(1)/var_vec[i]);
                	}
		}
		
		num_obs++;
		//int tno, std::vector<float> mv, std::vector<float> emv, std::vector<float> vv, std::vector<float> evv, float term1, int no, int pc
    		polytopes.push_back(Polytope(0, mean_vec, var_vec, det_term, num_o, class_num));
		int ps = polytopes.size() -1; 
		polytopes[ps].setObs(num_o);
	}
    	fin.close();
	std::cout<<"done 1\n";
	fflush(stdout);
	float tot_det_term = 0;
        for(auto &ele: tot_var)
        {
		if (ele <=0){
			std::cout<<"zero encountered!!!!!\n";
		fflush(stdout);
		}
                ele  = ele/(float)num_obs;
                tot_det_term += log(sqrt(ele));
                inverse_tot_var.push_back((float)1/ele);
        }
        polytopes[0].setPool(inverse_tot_var);
        polytopes[0].setPoold(tot_det_term);

	return polytopes;

}


std::vector<std::vector<float>> JSONReader::readHyperplanes(){
        //TODO: fill
	std::string hyper_fname = Config::getValue("hyperfilename");
	std::vector<std::vector<float>> hyperplanes = readHyper(hyper_fname);
	return hyperplanes;
}


std::vector<Polytope> JSONReader::readPolytopes(){
	bool write_csv = false;
	bool write_polytopes = false;
	std::ifstream in(filename);
	std::string contents((std::istreambuf_iterator<char>(in)),
			std::istreambuf_iterator<char>());
	const char *json = contents.data();
	auto start = std::chrono::steady_clock::now();
	Document d;
	d.Parse(json);
	assert(d.IsObject());
	const Value& maps = d;
	SizeType num_polytopes = d.Size();
	std::vector<Polytope> polytopes;
	int count_one=0;
	int total=0;
	int numrep = 0;
	std::map<std::vector<float>, std::vector<float>> mean_dict;
	std::map<std::vector<float>, std::vector<float>> ext_mean_dict;
	std::map<std::vector<float>, std::vector<float>> var_dict;
	std::map<std::vector<float>, std::vector<float>> ext_var_dict;

	/*
	   leaf_json_map_arr.append(X.mean(axis = 0).tolist())
	   leaf_json_map_arr.append(X.var(axis = 0).tolist())
	   leaf_json_map_arr.append(X.std(axis = 0).tolist())

	   const Value& scale_mean = maps[0];
	   const Value& scale_var = maps[1];
	   const Value& scale_stdv = map[2];
	   */
	int k;
	for(SizeType j=0; j<num_polytopes; ++j){
		std::vector<float> mean;
		std::vector<float> var;
		std::vector<float> ext_mean;
		std::vector<float> ext_var;
		const Value& tree_map = maps[j];
		int treenum = tree_map["treenum"].GetInt();
		int leafid = tree_map["leafid"].GetInt();
		const Value& mean_vec = tree_map["mean"];
		const Value& var_vec = tree_map["var"];
		k = (int) mean_vec.Size(); 
		float term1 = tree_map["term1"].GetDouble();
		float numobs = tree_map["numobs"].GetDouble();
		float tr_class = tree_map["class"].GetDouble();

		for(int i=0; i<k; ++i) {
			mean.push_back(mean_vec[i].GetDouble());
			ext_mean.push_back(mean_vec[i].GetDouble());
			var.push_back(var_vec[i].GetDouble());
			ext_var.push_back(var_vec[i].GetDouble());
		}
		mean.push_back(tr_class);
                mean.push_back(numobs);
		mean.push_back(j);
		if(mean_dict.find(mean) != mean_dict.end() ){

			if ( mean_dict[mean][3] != tr_class){
				std::cout<<"not equal";
				std::cout<<"old num obs: "<<mean_dict[mean][4]<<"\n";
				std::cout<<"new num obs: "<< numobs<<"\n";
				fflush(stdout);
			}

			//std::cout<<"number reps: "<<numrep<<"\n";
			mean_dict[mean][2]+=numobs;
			numrep++;
			continue;
		}

		std::vector<float> tempvec;
		tempvec.push_back(treenum);
		tempvec.push_back(term1);
		tempvec.push_back(numobs);
		tempvec.push_back(tr_class);
		tempvec.push_back(numobs);
		mean_dict[mean] = tempvec;
		var_dict[mean] = var;
		ext_mean.push_back( -2*term1 + 2*log(float(numobs))   );
		ext_var.push_back(1.0);
		ext_mean_dict[mean] = ext_mean;
		ext_var_dict[mean] = ext_var;
		//polytopes.push_back(Polytope(treenum, mean, ext_mean, var, ext_var, term1, (int)numobs, (int)tr_class));	
		//polytopes[i].print();
	}

	const Value& scale_var = maps[1];
	std::vector<float> inv_comm_var;
	float det_term2=0;
	for(int i=0; i<k; ++i){
		float newvar = scale_var[i].GetDouble();
		det_term2 += log(sqrt(newvar));
		inv_comm_var.push_back((float)1);
	}

	for(auto const &ele: mean_dict){
		std::vector<float> temp_mean = ele.first;
		std::vector<float> other_ele = ele.second;
		std::vector<float> realvar = var_dict[temp_mean];
		std::vector<float> mean_ext_vec2 = ext_mean_dict[temp_mean];
		std::vector<float> var_ext_vec2 = ext_var_dict[temp_mean];
		temp_mean.pop_back();
		temp_mean.pop_back();
		temp_mean.pop_back();	
		if(other_ele[4] < 18)
			continue;
		if (write_polytopes)
			polytopes.push_back(Polytope(other_ele[0], temp_mean, mean_ext_vec2, realvar, var_ext_vec2, other_ele[1], (int)other_ele[2], (int)other_ele[3] ));
		else
			polytopes.push_back(Polytope(other_ele[0], temp_mean, realvar, other_ele[1], (int)other_ele[2], (int)other_ele[3] ));
		int psiz = polytopes.size() - 1;
		polytopes[psiz].setObs(other_ele[4]);
		//polytopes[psiz].setVote(other_ele[2]);
		count_one++;
	}
	imputeVar(polytopes);
	if(write_polytopes){
		writePoly(polytopes);
		//exit(0);
		return polytopes;
	}

	/*for(auto &poly: polytopes){
	  if(poly.getObs() <= 1.0){
	  poly.setVar(inv_comm_var);
	  poly.setDetTerm(0);
	  count_one++;
	  }
	  }*/
	if(write_csv){
		writeCsv(polytopes);
		exit(0);
	}
	std::cout<<"Count one: "<<count_one<<"\n";
	return polytopes;
}

