#pragma once
#include <string>
#include <vector>
#include <algorithm> // for std::ranges::copy depending on lib support
#include <iostream>
#include <ios>
#include <iomanip>
#include <thread>
#include <sstream>
#include <cmath>
#include <filesystem>
#include <complex>

// -------------------------------------------------------- DEFINITIONS --------------------------------------------------------

#define stout std::cout << std::setprecision(8) << std::fixed						// standard out
//#define im cpx(0.0,1.0)

static const char* kPSep =
#ifdef _WIN32
"\\";
#else
"/";
#endif

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;
using cpx = std::complex<double>;

constexpr long double PI = 3.141592653589793238462643383279502884L;			// it is me, pi
constexpr long double TWO = 2 * 3.141592653589793238462643383279502884L;	// it is me, 2pi
constexpr long double PI_half = PI / 2.0;

// -------------------------------------------------------- ALGORITHMS FOR MC --------------------------------------------------------

/// <summary>
/// Here we will state all the already implemented definitions that will help us building the user interfrace
/// </summary>
namespace impDef {
	/// <summary>
	/// Different Monte Carlo algorithms that can be provided inside the classes (for simplicity in enum form)
	/// </summary>
	enum class algMC {
		metropolis,
		heat_bath,
		self_learning
	};
	/// <summary>
	/// Types of implemented lattice types
	/// </summary>
	enum class lattice_types {
		square
		//triangle,
		//hexagonal
	};
}

// -------------------------------------------------------- COMMON UTILITIES --------------------------------------------------------
#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

template<class T>
using v_3d = std::vector<std::vector<std::vector<T>>>;				// 3d double vector
template<class T>
using v_2d = std::vector<std::vector<T>>;							// 2d double vector
template<class T>
using v_1d = std::vector<T>;										// 1d double vector

// ----------------------------------------------------------------------------- TIME FUNCTIONS -----------------------------------------------------------------------------
inline double tim_s(clk::time_point start) {
	return double(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration(\
		std::chrono::high_resolution_clock::now() - start)).count()) / 1000.0;
}

// ----------------------------------------------------------------------------- TOOLS -----------------------------------------------------------------------------

/// <summary>
/// 
/// </summary>
/// <param name="filename"></param>
/// <param name="mode"></param>
/// <returns></returns>
template <typename T>
inline void openFile(T& file, std::string filename, std::ios_base::openmode mode = std::ios::out) {
	file.open(filename, mode);
	if (!file.is_open()) throw "couldn't open a file: " + filename + "\n";
}


/// <summary>
/// check the sign of a value
/// </summary>
/// <param name="val">value to be checked</param>
/// <returns>sign of a variable</returns>
template <typename T>
inline int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

/// <summary>
/// Defines an euclidean modulo denoting also the negative sign
/// </summary>
/// <param name="a">left side of modulo</param>
/// <param name="b">right side of modulo</param>
/// <returns>euclidean a%b</returns>
inline int myModuloEuclidean(int a, int b)
{
	int m = a % b;
	if (m < 0) {
		m = (b < 0) ? m - b : m + b;
	}
	return m;
}

/// <summary>
/// 
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="output"></param>
/// <param name="elements"></param>
/// <param name="separtator"></param>
template <typename T>
inline void printSeparated(std::ostream& output, std::string separtator = "\t", std::initializer_list<T> elements = {}) {
	const int elements_size = elements.size();
#pragma omp critical
	for (auto elem : elements) {
		output << elem << separtator;
	}
#pragma omp critical
	output << std::endl;
}

/// <summary>
/// Print vector separated by commas
/// </summary>
template <typename T>
std::ostream& operator<< (std::ostream& out, const v_1d<T>& v) {
    if ( !v.empty() ) {
        out << '[';
        for(int i = 0; i < v.size(); i++)
			out << v[i] << ", ";
        out << "\b\b]"; // use two ANSI backspace characters '\b' to overwrite final ", "
    }
    return out;
}

template <typename T>
std::ostream& operator<< (std::ostream& out, const v_1d<v_1d<T>>& v) {
	if ( !v.empty() ) {
		for (auto it : v) {
			out << "\t\t\t\t";
			for (int i = 0; i < it.size(); i++)
				out << it[i] << "\t";
			out << "\n";
		}
    }
    return out;

}
// -------------------------------------------------------- STRING RELATED FUNCTIONS --------------------------------------------------------

/// <summary>
/// Changes a value to a string with a given precision
/// </summary>
/// <param name="a_value">Value to be transformed</param>
/// <param name="n">Precision</param>
/// <returns>String of a value</returns>
template <typename T>
inline std::string to_string_prec(const T a_value, const int n = 3) {
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << a_value;
	return out.str();
}

v_1d<std::string> split_str(const std::string& s, std::string delimiter = "\t");

v_1d<std::string> change_input_to_vec_of_str(int argc, char** argv);


class pBar {
public:
	void update(double newProgress) {
        currentProgress += newProgress;
        amountOfFiller = (int)((currentProgress / neededProgress)*(double)pBarLength);
    }
	void print() {
        currUpdateVal %= pBarUpdater.length();
        stout << "\r";															// Bring cursor to start of line
		stout << firstPartOfpBar;												// Print out first part of pBar
        for (int a = 0; a < amountOfFiller; a++) {								// Print out current progress
            stout << pBarFiller;
        }
        stout << pBarUpdater[currUpdateVal];
        for (int b = 0; b < pBarLength - amountOfFiller; b++) {					// Print out spaces
            stout << " ";
        }
        stout << lastPartOfpBar;												// Print out last part of progress bar
        stout << " (" << (int)(100*(currentProgress/neededProgress)) << "%)";	// This just prints out the percent
        stout << std::flush;
        currUpdateVal += 1;
    }
private:
	// --------------------------- STRING ENDS
	std::string firstPartOfpBar = "\t\t\t\t[";
    std::string lastPartOfpBar = "]";
    std::string pBarFiller = "|";
    std::string pBarUpdater = "/-\\|";
	// --------------------------- PROGRESS
	int amountOfFiller;															// length of filled elements
	int pBarLength = 50;														// length of a progress bar
    int currUpdateVal = 0;														// 
	double currentProgress = 0;													// current progress
    double neededProgress = 100;												// final progress

};



#endif // COMMON_UTILS_H
