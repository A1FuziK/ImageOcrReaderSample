#include <iostream>
#include <string>
#include <vcruntime.h>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "BPNeuralNetwork.h"

//This is the side size of the char image
int image_factor = 48;

int layers = 3, layerSpec[3] = {image_factor * image_factor,image_factor * image_factor,64};
double learnRate = 0.01, momentum = 0.5, errorThreshHold = 0.0001, dTemp = 0.0;

//This should be enough in the train process
long maxIterations = 2000;

BPNeuralNetwork* oNet = nullptr;

int iFuto = 0;

std::string chars_on_image = "";


std::vector<unsigned char> LoadPngImage(std::string fileName, int& imageWidth, int& imageHeight)
{
    cv::Mat img = cv::imread(fileName, cv::IMREAD_COLOR);

    if (img.empty())
    {
        std::cout << "Could not open or find the image" << std::endl;
        return {};
    }

    cv::Mat bw_image;
    cv::cvtColor(img, bw_image, cv::COLOR_BGR2GRAY);
    
    std::vector<unsigned char> bytes;
    bytes.assign(bw_image.datastart, bw_image.dataend);

    imageWidth = bw_image.cols;
    imageHeight = bw_image.rows;
    return bytes;
}


void SavePngImage(std::string fileName, int imageWidth, int imageHeight,  std::vector<unsigned char> imageBytes)
{
    cv::Mat bw_image(imageHeight, imageWidth, CV_8UC1, &imageBytes[0]);
    cv::imwrite(fileName, bw_image);
}


void rescale_line(unsigned char* Source,int SrcWidth, unsigned char * Target, int TgtWidth)
{
	int NumPixels = TgtWidth;
	int IntPart = SrcWidth / TgtWidth;
	int FractPart = SrcWidth % TgtWidth;
	int E = 0;

	while (NumPixels-- > 0) {
		*Target++ = *Source;
		Source += IntPart;
		E += FractPart;

		if (E >= TgtWidth) {
			E -= TgtWidth;
			Source++;
		} /* if */
	} /* while */
}


void rescale_buffer(unsigned char *Source, int SrcWidth, int SrcHeight , unsigned char *Target, int TgtWidth, int TgtHeight)
{
	int NumPixels = TgtHeight;
	int IntPart = (SrcHeight / TgtHeight) * SrcWidth;
	int FractPart = SrcHeight % TgtHeight;
	int E = 0;

	unsigned char *PrevSource = NULL;

	while (NumPixels-- > 0) {
		if (Source == PrevSource) {
			memcpy(Target, Target-TgtWidth, TgtWidth*sizeof(*Target));
		} else {
			rescale_line(Source, SrcWidth ,Target, TgtWidth);
			PrevSource = Source;
		} /* if */

		Target += TgtWidth;
		Source += IntPart;
		E += FractPart;

		if (E >= TgtHeight) {
			E -= TgtHeight;
			Source += SrcWidth;
		} /* if */
	} /* while */
}


std::vector<unsigned char> rescale_image(int imageWidth, int imageHeight,  std::vector<unsigned char> imageBytes)
{
	std::vector<unsigned char> retBytes;
	
	unsigned char* pImage = imageBytes.data();
	unsigned char* pOutImage = (unsigned char*) malloc(image_factor * image_factor);
	memset(pOutImage,255,image_factor * image_factor);

	rescale_buffer(
		pImage,imageWidth,imageHeight,
		pOutImage,image_factor,image_factor);

	for (int n = 0 ; n < (image_factor*image_factor) ; n++)
		retBytes.push_back(pOutImage[n]);

	free(pOutImage);
	pOutImage = nullptr;

	return retBytes;
}


char InterpretOCRCharOnNeuralNet(int imageWidth, int imageHeight,  std::vector<unsigned char> imageBytes , int& iConfidence)
{
	char c_ret = '*';

	iConfidence = 0;

	std::vector<unsigned char> rescaledImage = rescale_image(imageWidth, imageHeight, imageBytes);

	iFuto++;
	std::string sFileName = "Work\\Samples\\" + std::to_string(iFuto) + ".png";
	SavePngImage(sFileName, image_factor, image_factor, rescaledImage);
	
	std::vector<double> imageInput;
	
	for (int n = 0 ; n < (image_factor * image_factor) ; n++) {
		if ( rescaledImage[n] <= 160) {
			imageInput.push_back(1.0);
		} else {
			imageInput.push_back(0.0);
		}
	}

	oNet->feedForward(imageInput.data());

	int iResChar = -1;
	double dStoredVal = 0.0;

	for (int iOut = 0 ; iOut < 64 ; iOut++) {
		if (oNet->outValue(iOut) > dStoredVal) {
			dStoredVal = oNet->outValue(iOut);
			iResChar = iOut;
		}
	}


	if (iResChar > -1) {
		char crBuf[] = {
			'<','0','1','2','3','4','5','6','7','8','9','>',
			'q','w','e','r','t','z','u','i','o','p','a','s','d','f','g','h','j','k','l','y','x','c','v','b','n','m',
			'Q','W','E','R','T','Z','U','I','O','P','A','S','D','F','G','H','J','K','L','Y','X','C','V','B','N','M'
			};

		c_ret = crBuf[iResChar];

		iConfidence = static_cast<int>(dStoredVal * 100);
		
		if (iConfidence > 99) {
			iConfidence = 99;
		}
	}


	return c_ret;
}


int _cutandinterpretchar(int imageWidth, int imageHeight,  std::vector<unsigned char> imageBytes, unsigned long Left, unsigned long Right, unsigned long Top, unsigned long Bottom, char* ocr, char* con)
{
	int iRet = 0;

	int line = 0;
	int pixel = 0;

	int pre_weight = 0;
	int weight = 0;

	int iAdatKezd = 0;

	for (line = Top; line < Bottom; line++)
	{
		pre_weight = weight;

		weight = 0;

		for (pixel = Left; pixel - Right; pixel++)
		{
			if (imageBytes[(line * imageWidth) + pixel] <= 160)
			{
				weight++;
			}
		}

		if (weight >= 2)
		{
			if (iAdatKezd == 0)
			{
				iAdatKezd = line;
			}
		}
		else if (weight <= 2 && pre_weight <= 2 && iAdatKezd && (line > Top + (Bottom - Top)/2))
		{
			break;
		}
	}

	if (iAdatKezd)
	{
		Top = iAdatKezd - 1;
	}

	Bottom = line;

	std::vector<unsigned char> cutBytes;
	int cutImageHeight = (int)(Bottom - Top);
	int cutImageWidth = (int)(Right - Left);

	for (unsigned long line = Top; line < Bottom; line++)
	{
		for (unsigned long pixel = Left; pixel < Right; pixel++)
		{
			cutBytes.push_back(imageBytes[(line * imageWidth) + pixel]);
		}
	}
	
	iFuto++;
	std::string sFileName = "Work\\Samples\\" + std::to_string(iFuto) + ".png";
	SavePngImage(sFileName, cutImageWidth, cutImageHeight, cutBytes);

	int iConfidence = 0;

	sprintf_s(ocr,2048, "%c", InterpretOCRCharOnNeuralNet(cutImageWidth, cutImageHeight,cutBytes, iConfidence));

	sprintf_s(con,2048, "%d", (iConfidence - 1) / 10);

	std::cout <<"$REALCHARINFO;"
	<< Left << ":" << Right << ":"
	<< Top << ":" << Bottom << ":"
	<< ocr << ":" << iConfidence << "\n";


	return iRet;
}


//Primitive function to find the characters in the line
int _messureonchars(int imageWidth, int imageHeight,  std::vector<unsigned char> imageBytes, int data_line, unsigned long sor_kezd, unsigned long sor_veg)
{
	if (sor_kezd < 0)
		sor_kezd = 0;

	if (sor_veg > imageHeight)
		sor_veg = imageHeight - 1;

	unsigned long ures_oszlopok = 0;
	unsigned long karakter = 0;

	bool bKarakter = false;
	unsigned long karakter_kezd = 0;
	unsigned long karakter_veg = 0;
	unsigned long karakter_ures_oszlopok = 0;

	unsigned long karakter_szel_gyujto = 0;

	unsigned long pixel;
	for (pixel = 0; pixel < imageWidth; pixel++)
	{
		unsigned long weight = 0;
		unsigned long line;
		for (line = sor_kezd; line < sor_veg; line++)
		{
			if (imageBytes[(line * imageWidth) + pixel] <= 160)
			{
				weight += 1;
			}
		}

		if (weight >= 2 && pixel < imageWidth - 1)
		{
			if (!bKarakter)
			{
				bKarakter = true;
				karakter_kezd = pixel;
				karakter_veg = pixel;
				karakter_ures_oszlopok = 0;
			}
			else
			{
				karakter_veg = pixel;
			}
		}
		else
		{
			if (bKarakter)
			{ // Ha van nyitott karakter
				if (karakter > 0)
				{
					int ures_hely = (ures_oszlopok + 1) / (karakter_szel_gyujto / karakter);
					for (int iBesz = 0 ; iBesz < ures_hely ; iBesz++)
						chars_on_image.append(" ");
				}
				
				ures_oszlopok = 0;
				
				karakter_ures_oszlopok++;
				if (karakter_ures_oszlopok > 3 || pixel == imageWidth - 1)
				{
					bKarakter = false;
					if ((karakter_veg - karakter_kezd) < 2)
					{
						//"ERROR: Character too narrow!\n");
					}
					else
					{
						char ocr[2048];
						char con[2048];
						memset(ocr, 0, sizeof(ocr));
						memset(con, 0, sizeof(con));

						_cutandinterpretchar(
							imageWidth,
							imageHeight,
							imageBytes,
							karakter_kezd - 1,
							karakter_veg + 2,
							sor_kezd,
							sor_veg,
							ocr,
							con);
						
						/*std::cout << "$CHARINFO;"
						<< karakter_kezd - 1 << ":"
						<< karakter_veg + 2 << ":"
						<< sor_kezd << ":"
						<< sor_veg << ":"
						<< ocr << ":"
						<< con << "\n";*/

						chars_on_image.append(ocr);

						karakter++;
						karakter_szel_gyujto += karakter_veg - karakter_kezd;
					}
				}
			}
			else
			{
				ures_oszlopok++;
			}
		}
	}

	return karakter;
}


//Primitive function to find the character - lines in the document
int _messureonlines(int imageWidth, int imageHeight,  std::vector<unsigned char> imageBytes)
{
	int fellelt_sorok = 0;

	bool bSor = false;
	unsigned long sor_kezd = 0;
	unsigned long sor_veg = 0;

	unsigned long line;
	for (line = 0; line < imageHeight; line++)
	{
		unsigned long pixel;
		unsigned long weight = 0;
		for (pixel = 0; pixel < imageWidth; pixel++)
		{
			if (imageBytes[(line * imageWidth) + pixel] <= 160)
			{
				weight += 1;
			}
		}

		if (weight > 5)
		{ // It may have some character(s) / thing(s)
			if (!bSor)
			{
				bSor = true;
				sor_kezd = line;
			}
		}
		else
		{ // Valószínűleg nincsen adat benne
			if (bSor)
			{ // Van nyitott sor
				bSor = false;
				sor_veg = line;

				/////////////////////////////////////////////
				unsigned long sor_vastagsag = sor_veg - sor_kezd;

				{
					std::cout << "LINE LOCATED: " << sor_kezd << " -- " << sor_veg << " -- " << sor_vastagsag << "\n";

					_messureonchars(
						imageWidth,
						imageHeight,
						imageBytes,
						fellelt_sorok,
						sor_kezd - 10,
						sor_veg + 10);

					chars_on_image.append("\n");

					fellelt_sorok++;
					
					line += 10;
				}
				/////////////////////////////////////////////
			}
		}
	}

	return fellelt_sorok;
}

void ProcessBWImageBytes(int imageWidth, int imageHeight,  std::vector<unsigned char> imageBytes)
{
	chars_on_image = "";
	
	_messureonlines(imageWidth, imageHeight, imageBytes);

	std::cout << "Recognized:" << "\n" << chars_on_image << "\n";
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


std::vector<std::string> GetSampleFiles(std::string samplesDir)
{
	std::vector<std::string> files;
	for (const auto & entry : std::filesystem::directory_iterator(samplesDir))
		files.push_back(entry.path().string());
	return files;
}


struct charsample
{
	int imageWidth;
	int imageHeight;
	std::vector<unsigned char> imageBytes;
	
	std::vector<double> inputValues;
	std::vector<double> requestedOutValues;
};


void TrainNeuralNetwork()
{
	std::vector<std::string> samples = GetSampleFiles("Work\\Train\\");

	std::vector<charsample> colCharSamples;
	
	for (const auto & pathString : samples)
	{
		std::string fileName = std::filesystem::path(pathString).filename().string();
		std::string fileRoot = split(split(fileName,'.')[0],'_')[0];

		charsample oCharSample;

		for (int i = 0; i < 64 ; i++)
		{
			if (i == std::stoi(fileRoot) - 1)
				oCharSample.requestedOutValues.push_back(1.0);
			else
				oCharSample.requestedOutValues.push_back(0.0);
		}

		int imageWidth;
		int imageHeight;
		std::vector<unsigned char> imageBytes;
		imageBytes = LoadPngImage(pathString,imageWidth,imageHeight);
		std::vector<unsigned char> rescaledBytes;
		rescaledBytes = rescale_image(imageWidth, imageHeight, imageBytes);

		oCharSample.imageBytes = rescaledBytes;
		oCharSample.imageHeight = oCharSample.imageWidth = image_factor;

		for (int i = 0; i < image_factor * image_factor ; i++)
		{
			if (oCharSample.imageBytes[i] <= 160)
				oCharSample.inputValues.push_back(1.0);
			else
				oCharSample.inputValues.push_back(0.0);
		}

		colCharSamples.push_back(oCharSample);
	}

	std::cout << "Let's train the network..." << "\n";

	long iteration;
	bool bTrained = false;

	long trainCycles = 0;
	double comulatedDistance = 0.0;

	for (iteration = 0; iteration < maxIterations && bTrained == false; iteration++)
	{
		bTrained = true;
		for (int iBuffLearn = 0; iBuffLearn < colCharSamples.size(); iBuffLearn++)
		{
			for (int iInnerLeanCycle = 0; iInnerLeanCycle < 1; iInnerLeanCycle++)
			{
			
				oNet->backPropagate(
					colCharSamples[iBuffLearn].inputValues.data(),
					colCharSamples[iBuffLearn].requestedOutValues.data());
			
				dTemp = oNet->meanSquareError(colCharSamples[iBuffLearn].requestedOutValues.data());

				comulatedDistance += dTemp;
				trainCycles++;
				if (dTemp <= (comulatedDistance / trainCycles) || dTemp < errorThreshHold)
					break;
			}

			if (dTemp < errorThreshHold)
			{
			
			}
			else
			{
				bTrained = false;
			}
		}

		if (iteration % (maxIterations / 100) == 0)
		{
			std::cout << "Still training... last meanSquareError: " << dTemp
				<< " average meanSquareError: " << (comulatedDistance / trainCycles) << "\n";
		}
	}

	std::cout << trainCycles << " train cycles completed... in " << iteration << " main iterations... "
		<< " average meanSquareError: " << (comulatedDistance / trainCycles)
		<< " last meanSquareError: " << dTemp << "\n";
}


int main(int argc, char* argv[])
{
	//Create the network
	oNet = new BPNeuralNetwork(layers, layerSpec, learnRate, momentum);

	//Train the network
	TrainNeuralNetwork();

	//Save the weights
	oNet->saveNet("Work\\trained.net");
	delete oNet;

	//return 0;

	//Recreate the network
	oNet = new BPNeuralNetwork(layers, layerSpec, learnRate, momentum);

	//Load the weights
	oNet->loadNet("Work\\trained.net");
	
    int imageWidth = 0;
    int imageHeight = 0;
	
	std::vector<unsigned char> imgBytes;
    
    imgBytes = LoadPngImage("Work\\minta.png", imageWidth, imageHeight);
	ProcessBWImageBytes(imageWidth, imageHeight, imgBytes);
	
	imgBytes = LoadPngImage("Work\\teszt.png", imageWidth, imageHeight);
	ProcessBWImageBytes(imageWidth, imageHeight, imgBytes);
	
    return 0;
}