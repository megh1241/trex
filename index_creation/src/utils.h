#include<iostream>
#include<stdlib.h>
#include <vector>

std::vector<float> subtractVecs(std::vector<float> a, std::vector<float> b);
std::vector<float> addVecs(std::vector<float> a, std::vector<float> b);
std::vector<float> mulVecs(std::vector<float> a, std::vector<float> b);
float dotVecs(std::vector<float> a, std::vector<float> b);
std::vector<float> inverseVec(std::vector<float> a);
void loadTestData(std::vector<std::vector<float>>& test_data, std::vector<int>& labels);
void loadTestData(std::vector<std::vector<float>>& test_data, std::vector<double>& labels);
double getAccuracy(const std::vector<int> &predicted, const std::vector<int> &labels);
double getAccuracy(const std::vector<double> &predicted, const std::vector<double> &labels);
