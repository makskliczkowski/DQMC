#pragma once
#include "common.h"

#ifndef PROGRESS.H
#define PROGRESS.H
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
	void printWithTime(const std::string& message, double percentage) {
#pragma omp critical
		{
			stout << "\t\t\t\t-> time: " << tim_s(timer) << message << " : \n";
			this->print();
			stout << std::endl;
		}
		this->update(percentage);
	}
	pBar() : timer(std::chrono::high_resolution_clock::now()), amountOfFiller(0) {};	    // constructor
private:
	// --------------------------- STRING ENDS
	std::string firstPartOfpBar = "\t\t\t\t[";
	std::string lastPartOfpBar = "]";
	std::string pBarFiller = "|";
	std::string pBarUpdater = "/-\\|";
	// --------------------------- PROGRESS
	clk::time_point timer;														            // inner clock
	int amountOfFiller;															            // length of filled elements
	int pBarLength = 50;														            // length of a progress bar
	int currUpdateVal = 0;														            //
	double currentProgress = 0;													            // current progress
	double neededProgress = 100;												            // final progress
};


#endif // !PROGRESS.H