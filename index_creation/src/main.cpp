#include <iostream>
#include <string>
#include <cstring>
#include <unordered_set>
#include <omp.h>
#include "config.h"
#include "polydb.h"
#include "polytope.h"
#include "utils.h"
#include <omp.h>
#include <thread>
#include <iostream>
#include <chrono>

const int min_num_cmd_args = 3;

static void showUsage(){
    std::cout<<"Usage: "
        "./exe "
	"--numthreads <number of threads>"
        "--polymodelfilename <polytope json filename>"
        "--datafilename <data filename>"
	"--distance <mahalanobis/loglikelihood"
	"--measure <topk/topone"
	"--topk <k>"
	"--percmean <k>"
	"--percdet <k>"
	"\n";
}

/*
       JSONReader json_obj = JSONReader(Config::getValue("polymodelfilename"));
               std::string mean_filename = Config::getValue("meanfilename");
	               std::string pool_filename = Config::getValue("poolfilename");
		               std::string class_filename = Config::getValue("classfilename");
			               std::string top_features_filename = Config::getValue("featurefilename");
*/
static void parseArgs(int argc, char* argv[]){
	const std::unordered_set<std::string> cmdline_args = 
	{"--hyperfilename", "--cardfilename", "--help", "--newfilename", "--poolfilename", "--classfilename", "--meanfilename", "--altfilename", "--writefname", "--featurefilename", "--writefname2",   "--oversample", "--undersample", "--topm", "--numthreads", "--percobs", "--polymodelfilename", "--datafilename", "--distance", "--measure", "--topk", "--percmean", "--percdet"};
	if (argc < min_num_cmd_args){
        	std::cerr<<"Invalid set of commandline args!\n";
        	showUsage();
        	exit(-1);
	}
	for(int i=1; i<argc; i+=2){
		if(strcmp(argv[i], "--help") == 0)
            		showUsage();
        	else if(cmdline_args.find(argv[i]) != cmdline_args.end()){
            		std::string config_key = std::string(argv[i]).erase(0, 2);
            		Config::setConfigItem(config_key, argv[i+1]);
        	}
        	else{
            		std::cerr<<"Unknown commandline args: "<<argv[i]<<"\n";
            		showUsage();
            		exit(-1);
        	}
	}
	if(Config::getValue("numthreads") == std::string("notfound")){
		int proc_count = (int)std::thread::hardware_concurrency();
		Config::setConfigItem(std::string("numthreads"), std::to_string(proc_count));
	}
	if(Config::getValue("distance") == std::string("notfound")){
		Config::setConfigItem(std::string("distance"), std::string("mahalanobis"));
	}
	if(Config::getValue("measure") == std::string("notfound")){
		Config::setConfigItem(std::string("measure"), std::string("topone"));
	}
	if(Config::getValue("topk") == std::string("notfound")){
		Config::setConfigItem(std::string("topk"), std::string("100"));
	}
	if(Config::getValue("topm") == std::string("notfound")){
		Config::setConfigItem(std::string("topk"), std::string("25"));
	}
	if(Config::getValue("percmean") == std::string("notfound")){
		Config::setConfigItem(std::string("percmean"), std::string("1"));
	}
	if(Config::getValue("percdet") == std::string("notfound")){
		Config::setConfigItem(std::string("percdet"), std::string("1"));
	}
	omp_set_num_threads(std::atoi(Config::getValue("numthreads").c_str()));
}


int main(int argc, char* argv[]){
	parseArgs(argc, argv);
	std::cout<<"Args parsed\n";
	fflush(stdout);
	
	
	
	std::vector<std::vector<float>> test_X;
	std::vector<int> test_y;
	std::vector<int> pred_y;
	auto end2 = std::chrono::steady_clock::now();
	loadTestData(test_X, test_y);
	auto end3 = std::chrono::steady_clock::now();
	std::cout<<"Test data loaded\n";
	fflush(stdout);
	double elapsed2 = std::chrono::duration<double, std::milli>(end3 - end2).count();
	std::cout<<"time: "<<elapsed2<<"\n";
	fflush(stdout);
	
	Model poly_model = Model();
	auto start = std::chrono::steady_clock::now();
	//poly_model.readModel();
	poly_model.readModelPooledHyper(test_X);
	auto end = std::chrono::steady_clock::now();
	std::cout<<"Model read\n";
	fflush(stdout);
	double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout<<"time: "<<elapsed<<"\n";
	
	
	
	int iter = 0;
	int correct = 0;
	int pred_cl;
	//int max_obs = 20;
	int obs = 0;
	//poly_model.writeTest(test_X);
	//exit(0);
	for(auto single_obs: test_X){
		auto end4 =  std::chrono::steady_clock::now();
		if (Config::getValue("distance") == std::string("db")){
			PolyDB polydb_obj = PolyDB();
			polydb_obj.constructDB(poly_model, single_obs, false);
			pred_cl = polydb_obj.intersectionExperiment();
			std::cout<<pred_cl<<",";
			fflush(stdout);
		}
		else if (Config::getValue("distance") == std::string("ext")){
			//PolyDB polydb_obj = PolyDB();
			//polydb_obj.constructDB(poly_model, single_obs, false);
			pred_cl = poly_model.findClosestPolytopeNew(single_obs);
			//pred_cl = polydb_obj.intersectionExperimentExt();
			std::cout<<pred_cl<<",";
			fflush(stdout);
		}
		else{
		}
		//std::cout<<"predicted: "<<pred_cl<<"\n";
		//std::cout<<"actual: "<<test_y[iter]<<"\n";
		if (pred_cl == test_y[iter])
			correct++;
		std::cout<<"num correct: "<<correct<<"\n";
		iter++;
		std::cout<<"total: "<<iter<<"\n";
		fflush(stdout);
		pred_y.push_back(pred_cl);
		auto end5 = std::chrono::steady_clock::now();
		double elapsed3 = std::chrono::duration<double, std::milli>(end5 - end4).count();
		obs++;
		//if (obs > 100){
		//}
	}
			//poly_model.writeTimeToFile();
			//exit(0);
	double acc = getAccuracy(pred_y, test_y);
	std::cout<<"Accuracy : "<<acc<<"\n";
	
	}
