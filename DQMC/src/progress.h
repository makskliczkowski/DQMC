#pragma once
#include "common.h"

#ifndef PROGRESS_H
#define PROGRESS_H
// -------------------------------------------------------- PROGRESS BAR --------------------------------------------------------
class pBar {
public:
	void update(double newProgress) {
		currentProgress += newProgress;
		amountOfFiller = (int)((currentProgress / neededProgress) * (double)pBarLength);
	}
	void print() {
		currUpdateVal %= pBarUpdater.length();
		stout << "\r";															            // Bring cursor to start of line
		stout << firstPartOfpBar;												            // Print out first part of pBar
		for (int a = 0; a < amountOfFiller; a++) {								            // Print out current progress
			stout << pBarFiller;                                                            // By filling the output
		}
		stout << pBarUpdater[currUpdateVal];                                                
		for (int b = 0; b < pBarLength - amountOfFiller; b++) {					            // Print out spaces
			stout << " ";
		}
		stout << lastPartOfpBar;												            // Print out last part of progress bar
		stout << " (" << (int)(100 * (currentProgress / neededProgress)) << "%)";	        // This just prints out the percent
		stout << std::flush;
		currUpdateVal += 1;
	}
	
	void printWithTime(std::string message) {
#pragma omp critical
		{
			stout << "\t\t\t\t-> time: " << tim_s(timer) << message << " : \n";
			this->print();
			stout << std::endl;
		}
		this->update(percentage);
	}
	


	pBar() : timer(std::chrono::high_resolution_clock::now()) { };							// constructor
	pBar(double percentage, int discreteSteps)
		: percentage(percentage)
		, timer(std::chrono::high_resolution_clock::now())
		, percentageSteps(static_cast<int>(percentage* discreteSteps / 100.0))
	{};
private:
	// --------------------------- STRING ENDS
	std::string firstPartOfpBar = "\t\t\t\t[";
	std::string lastPartOfpBar = "]";
	std::string pBarFiller = "|";
	std::string pBarUpdater = "|\\/";
	// --------------------------- PROGRESS
	clk::time_point timer;														            // inner clock
	int amountOfFiller = 0;															            // length of filled elements
	int pBarLength = 50;														            // length of a progress bar
	int currUpdateVal = 0;														            //
	double currentProgress = 0;													            // current progress
	double neededProgress = 100;												            // final progress
public:
	auto get_start_time()																	const RETURNS(this->timer);
	double percentage = 34;																	// print percentage
	int percentageSteps = 1;
};


#endif // !PROGRESS_H