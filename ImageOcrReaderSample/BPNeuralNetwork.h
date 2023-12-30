#ifndef BP_NEURAL_NETWORK_H
#define BP_NEURAL_NETWORK_H

#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>

/* Based on BackProp.h */

class ThreadSafeRandom {
public:
	ThreadSafeRandom() : dre(std::random_device{}()), dis(-1.0,+1.0) { }

	// Get random number
	double getNext() {
		std::lock_guard<std::mutex> lock(m);
		return dis(dre) / 1000;
	}

private:
	std::mt19937 dre;
	//std::uniform_int_distribution<int> dis;
	std::uniform_real_distribution<double> dis;
	std::mutex m;
};

class BPNeuralNetwork{

	//neuron outputs
	double **out;

	//neuron error differences
	double **delta;

	//weights
	double ***weight;

	//layers of the network
	int layers;

	//layer' sizes
	int *layerSpec;

	double learnRate;

	double momentum;

	double ***weightChange;

	bool sigmoidOutput;

	double transferFunction(double in, bool outputLayer)
	{
		if (outputLayer && sigmoidOutput)
		{ //sigmoid for the output layer
			return (double)(1 / (1 + exp(-in)));
		}
		else
		{ //relu for the inner layers
			if (sigmoidOutput) {
				if (in >= 0) {
					return in;
				} else {
					return 0;
				}
			} else { //linear output - for image processing
				if (in >= 0) {
					return in;
				} else {
					return in * 0.1;
				}
			}
		}
	}

	double transferFunctionDerivative(double in, bool outputLayer)
	{
		if (outputLayer && sigmoidOutput)
		{ // sigmoid derivative for the output layer
			return (double)((1.0 - transferFunction(in, outputLayer)) * transferFunction(in, outputLayer)); //sigmoid derivative
		}
		else
		{ //relu derivative for the inner layers
			if (sigmoidOutput) {
				if (in >= 0) {
					return 1;
				} else {
					return 0;
				}
			} else { //linear output derivative
				if (in >= 0) {
					return 1;
				} else {
					return 0.1;
				}
			}
		}
	}

	std::vector<std::string> split(const std::string& s, char delimiter) {
		std::vector<std::string> tokens;
		std::string token;
		std::istringstream tokenStream(s);
		while (std::getline(tokenStream, token, delimiter)) {
			tokens.push_back(token);
		}
		return tokens;
	}

	void writeStringToFile(const std::string& filename, const std::string& content) {
		std::ofstream outFile(filename, std::ios::trunc);
		if (outFile.is_open()) {
			outFile << content;
			outFile.close();
		}
		else {
			std::cout << "Unable to open file for writing.";
		}
	}

	std::string readStringFromFile(const std::string& filename) {
		std::ifstream inFile(filename);
		std::stringstream strStream;
		if (inFile.is_open()) {
			strStream << inFile.rdbuf(); // read the file
			inFile.close();
		}
		else {
			std::cout << "Unable to open file for reading.";
		}
		return strStream.str(); // str holds the content of the file
	}

public:

	~BPNeuralNetwork()
	{
		for (int i = 0; i < layers; i++)
			delete[] out[i];
		delete[] out;

		for (int i = 1; i < layers; i++)
			delete[] delta[i];
		delete[] delta;

		for (int i = 1; i < layers; i++)
			for (int j = 0; j < layerSpec[i]; j++)
				delete[] weight[i][j];

		for (int i = 1; i < layers; i++)
			delete[] weight[i];
		delete[] weight;

		for (int i = 1; i < layers; i++)
			for (int j = 0; j < layerSpec[i]; j++)
				delete[] weightChange[i][j];

		for (int i = 1; i < layers; i++)
			delete[] weightChange[i];
		delete[] weightChange;

		delete[] layerSpec;
	}


	BPNeuralNetwork(
		int _layers,
		int *_layerSpec,
		double _learnRate,
		double _momentum)
	{
		//Use linear activation for better image interpretation
		sigmoidOutput = false;
		
		learnRate = _learnRate;
		momentum = _momentum;

		layers = _layers;
		layerSpec = new int[layers];

		int i;

		for (i = 0; i < layers; i++) {
			layerSpec[i] = _layerSpec[i];
		}

		out = new double* [layers + 1];
		for (i = 0; i < layers; i++) {
			out[i] = new double[layerSpec[i]];
			if (i == layers - 1) {
				out[i + 1] = new double[layerSpec[i]];
			}
		}

		delta = new double* [layers];
		for (i = 1; i < layers; i++) {
			delta[i] = new double[layerSpec[i]];
		}

		weight = new double** [layers];
		for (i = 1; i < layers; i++) {
			weight[i] = new double* [layerSpec[i]];
		}
		for (i = 1; i < layers; i++) {
			for (int j = 0; j < layerSpec[i]; j++) {
				weight[i][j] = new double[layerSpec[i - 1] + 1];
			}
		}

		weightChange = new double** [layers];
		for (i = 1; i < layers; i++) {
			weightChange[i] = new double* [layerSpec[i]];
		}
		for (i = 1; i < layers; i++) {
			for (int j = 0; j < layerSpec[i]; j++) {
				weightChange[i][j] = new double[layerSpec[i - 1] + 1];
			}
		}

		ThreadSafeRandom* tsr = new ThreadSafeRandom();
		
		//srand((unsigned)(time(NULL)));
		for (i = 1; i < layers; i++)
			for (int j = 0; j < layerSpec[i]; j++)
				for (int k = 0; k < layerSpec[i - 1] + 1; k++)
					weight[i][j][k] = tsr->getNext(); //(double) - 1.0 + ((double)(rand()) / (32767 / 2));

		delete tsr;
		tsr = nullptr;

		for (i = 1; i < layers; i++)
			for (int j = 0; j < layerSpec[i]; j++)
				for (int k = 0; k < layerSpec[i - 1] + 1; k++)
					weightChange[i][j][k] = (double)0.0;

	}

	void backPropagate(double *in,double *tgt)
	{
		double sum;

		feedForward(in);

		//calculate error difference for the output layer
		for (int i = 0; i < layerSpec[layers - 1]; i++) {
			delta[layers - 1][i] = transferFunctionDerivative(out[layers - 1][i], true) * (tgt[i] - out[layers - 1][i]);
		}

		//calculate error difference for the other layers
		for (int i = layers - 2; i > 0; i--) {
			for (int j = 0; j < layerSpec[i]; j++) {
				sum = 0.0;
				for (int k = 0; k < layerSpec[i + 1]; k++) {
					sum += delta[i + 1][k] * weight[i + 1][k][j];
				}

				delta[i][j] = transferFunctionDerivative(out[i][j], false) * sum;
			}
		}

		//add momentum if any
		for (int i = 1; i < layers; i++) {
			for (int j = 0; j < layerSpec[i]; j++) {
				for (int k = 0; k < layerSpec[i - 1]; k++) {
					weight[i][j][k] += momentum * weightChange[i][j][k];
				}
				weight[i][j][layerSpec[i - 1]] += momentum * weightChange[i][j][layerSpec[i - 1]];
			}
		}

		//update weights using learnRate	
		for (int i = 1; i < layers; i++) {
			for (int j = 0; j < layerSpec[i]; j++) {
				for (int k = 0; k < layerSpec[i - 1]; k++) {
					weightChange[i][j][k] = learnRate * delta[i][j] * out[i - 1][k];
					weight[i][j][k] += weightChange[i][j][k];
				}
				weightChange[i][j][layerSpec[i - 1]] = learnRate * delta[i][j];
				weight[i][j][layerSpec[i - 1]] += weightChange[i][j][layerSpec[i - 1]];
			}
		}
	}

	void feedForward(double *in)
	{
		double sum;

		//input to input the layer
		for (int i = 0; i < layerSpec[0]; i++)
			out[0][i] = in[i];

		//calculate output for the other layers
		for (int i = 1; i < layers; i++) {
			for (int j = 0; j < layerSpec[i]; j++) {
				sum = 0.0;
				for (int k = 0; k < layerSpec[i - 1]; k++) {
					sum += out[i - 1][k] * weight[i][j][k];
				}

				//Add BIAS with (1.0 * weight) too
				sum += weight[i][j][layerSpec[i - 1]];

				out[i][j] = transferFunction(sum, i == (layers - 1));

				if (i == layers - 1) {
					out[i + 1][j] = exp(sum);
				}
			}

			if (i == layers - 1) {
				double theExpSum = 0.0;
				for (int j = 0; j < layerSpec[i]; j++){
					theExpSum += out[i + 1][j];
				}

				for (int j = 0; j < layerSpec[i]; j++) {
					out[i + 1][j] = out[i + 1][j] / theExpSum;
				}
			}
		}
	}

	double meanSquareError(double *tgt) const
	{
		double mse = 0;

		for (int i = 0; i < layerSpec[layers - 1]; i++) {
			mse += (tgt[i] - out[layers - 1][i]) * (tgt[i] - out[layers - 1][i]);
		}

		//return sum * 1/n
		return mse / (layerSpec[layers - 1]);
	}	
	
	double outValue(int outPosition) const
	{
		return out[layers - 1][outPosition];
	}
	
	double outSoftMaxValue(int outPosition) const
	{
		return out[layers][outPosition];
	}

	std::string getNetWeights()
	{
		std::string weightString = "";

		for (int i = 1; i < layers; i++)
			for (int j = 0; j < layerSpec[i]; j++)
				for (int k = 0; k < layerSpec[i - 1] + 1; k++) {
					weightString.append(std::to_string(weight[i][j][k]));
					weightString.append(";");
				}

		return weightString;
	}

	void setNetWeights(std::string weightString)
	{
		std::vector<std::string> weightStrings = split(weightString, ';');

		long nextStringValue = 0;

		for (int i = 1; i < layers; i++)
			for (int j = 0; j < layerSpec[i]; j++)
				for (int k = 0; k < layerSpec[i - 1] + 1; k++)
					weight[i][j][k] = std::stod(weightStrings[nextStringValue++]);

	}

	void loadNet(std::string fileName)
	{
		setNetWeights(readStringFromFile(fileName));
	}

	void saveNet(std::string fileName)
	{
		writeStringToFile(fileName, getNetWeights());
	}

};

#endif //BP_NEURAL_NETWORK_H 

