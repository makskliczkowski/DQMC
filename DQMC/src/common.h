#pragma once
#include "../include/random.h"
#include <string>
#include <vector>
#include <algorithm> // for std::ranges::copy depending on lib support
#include <iostream>
#include <ios>
#include <iomanip>
#include <thread>
#include <sstream>
#include <cmath>
#include <complex>
// filesystem for directory creation
#ifdef __has_include
#  if __has_include(<filesystem>)
#    include <filesystem>
#    define have_filesystem 1
namespace fs = std::filesystem;
#  elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
#    define have_filesystem 1
#    define experimental_filesystem
namespace fs = std::experimental::filesystem;
#  else
#    define have_filesystem 0
#  endif
#endif

// armadillo
#define ARMA_64BIT_WORD // enabling 64 integers in armadillo obbjects
#define ARMA_BLAS_LONG_LONG // using long long inside LAPACK call
#define ARMA_USE_OPENMP
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

// -------------------------------------------------------- DEFINITIONS --------------------------------------------------------

#define stout std::cout << std::setprecision(8) << std::fixed						// standard out
//#define im cpx(0.0,1.0)

static const char* kPSep =
#ifdef _WIN32
R"(\)";
#else
"/";
#endif

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;
using cpx = std::complex<double>;
using uint = unsigned int;
using ulong = unsigned long;
using ull = unsigned long long;

constexpr long double PI = 3.141592653589793238462643383279502884L;			// it is me, pi
constexpr long double TWOPI = 2 * PI;										// it is me, 2pi
constexpr long double PI_half = PI / 2.0;
constexpr cpx im_num = cpx(0, 1);
const std::string kPSepS = std::string(kPSep);

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

//v_1d<double> fourierTransform(std::initializer_list<const arma::mat&> matToTransform, std::tuple<double,double,double> k, std::tuple<int,int,int> L);

// ----------------------------------------------------------------------------- MATRIX MULTIPLICATION

void setMatrixFromSubmatrix(arma::mat& M2Set, const arma::mat& MSet, uint row, uint col, uint Nrows, uint Ncols, bool update = true, bool minus = false);
void setSubmatrixFromMatrix(arma::mat& M2Set, const arma::mat& MSet, uint row, uint col, uint Nrows, uint Ncols, bool update = true, bool minus = false);

arma::mat inv_left_plus_right_qr(arma::mat& Ql, arma::mat& Rl, arma::umat& Pl, arma::mat& Tl, arma::vec& Dl, arma::mat& Qr, arma::mat& Rr, arma::umat& Pr, arma::mat& Tr, arma::vec& Dr, arma::vec& Dtmp);

/// <summary>
///
/// </summary>
/// <param name="mat_to_multiply"></param>
/// <param name="Q"></param>
/// <param name="R"></param>
/// <param name="P"></param>
/// <param name="T"></param>
/// <param name="D"></param>
void inline setUDTDecomp(const arma::mat& mat_to_multiply, arma::mat& Q, arma::mat& R, arma::umat& P, arma::mat& T, arma::vec& D) {
	if (!arma::qr(Q, R, P, mat_to_multiply)) throw "decomposition failed\n";
	for (int i = 0; i < R.n_rows; i++) {
		D(i) = 1.0 / R(i, i);
	}
	T = ((diagmat(D) * R) * P.t());
}

/// <summary>
///
/// </summary>
/// <param name="mat_to_multiply"></param>
/// <param name="Q"></param>
/// <param name="R"></param>
/// <param name="P"></param>
/// <param name="T"></param>
void inline multiplyMatricesQrFromRight(const arma::mat& mat_to_multiply, arma::mat& Q, arma::mat& R, arma::umat& P, arma::mat& T, arma::vec& D) {
	if (!arma::qr(Q, R, P, (mat_to_multiply * Q) * diagmat(R))) throw "decomposition failed\n";
	for (int i = 0; i < R.n_rows; i++) {
		D(i) = 1.0 / R(i, i);
	}
	T = ((diagmat(D) * R) * P.t()) * T;
}

/// <summary>
///
/// </summary>
/// <param name="R"></param>
/// <param name="D"></param>
void inline makeTwoScalesFromUDT(arma::mat& R, arma::vec& D) {
	for (int i = 0; i < R.n_rows; i++)
	{
		if (abs(R(i, i)) > 1) {
			R(i, i) = 1;
		}
		else {
			D(i) = 1;
		}
	}
}

/// <summary>
///
/// </summary>
/// <param name="R"></param>
/// <param name="D"></param>
void inline makeTwoScalesFromUDT_full(arma::mat& R, arma::vec& D) {
	for (int i = 0; i < R.n_rows; i++)
	{
		if (abs(R(i, i)) > 1.0) {
			D(i) = R(i, i);
			R(i, i) = 1;
		}
		else {
			D(i) = 1;
		}
	}
}

// ----------------------------------------------------------------------------- FILE AND STREAMS

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

template <typename T>
inline T variance(T value, T average, int norm) {
	return std::sqrt((value / norm - average * average) / norm);
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
inline void printSeparated(std::ostream& output, std::string separtator = "\t", std::initializer_list<T> elements = {}, arma::u16 width = 8, bool endline = true) {
	for (auto elem : elements) {
		output.width(width); output << elem << separtator;
	}
	if (endline) output << std::endl;
}

/// <summary>
/// Print vector separated by commas
/// </summary>
template <typename T>
std::ostream& operator<< (std::ostream& out, const v_1d<T>& v) {
	if (!v.empty()) {
		out << '[';
		for (int i = 0; i < v.size(); i++)
			out << v[i] << ", ";
		out << "\b\b]"; // use two ANSI backspace characters '\b' to overwrite final ", "
	}
	return out;
}

template <typename T>
std::ostream& operator<< (std::ostream& out, const v_1d<v_1d<T>>& v) {
	if (!v.empty()) {
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
inline std::string to_string_prec(const T a_value, const int n = 2) {
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
		amountOfFiller = (int)((currentProgress / neededProgress) * (double)pBarLength);
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
		stout << " (" << (int)(100 * (currentProgress / neededProgress)) << "%)";	// This just prints out the percent
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
	// constructor
	pBar() {
		timer = std::chrono::high_resolution_clock::now();
		amountOfFiller = 0;
	}
private:
	// --------------------------- STRING ENDS
	std::string firstPartOfpBar = "\t\t\t\t[";
	std::string lastPartOfpBar = "]";
	std::string pBarFiller = "|";
	std::string pBarUpdater = "/-\\|";
	// --------------------------- PROGRESS
	clk::time_point timer;														// inner clock
	int amountOfFiller;															// length of filled elements
	int pBarLength = 50;														// length of a progress bar
	int currUpdateVal = 0;														//
	double currentProgress = 0;													// current progress
	double neededProgress = 100;												// final progress
};

#endif // COMMON_UTILS_H
