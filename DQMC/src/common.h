#pragma once
/// user includes
#include "../include/random.h"

#include <string>
#include <vector>
#include <algorithm> 													// for std::ranges::copy depending on lib support
#include <iostream>
#include <ios>
#include <iomanip>
#include <thread>
#include <sstream>
#include <cmath>
#include <complex>

/// filesystem for directory creation
#ifdef __has_include
#  if __has_include(<filesystem>)
#    include <filesystem>
#    define have_filesystem 1
namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;
#  elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
#    define have_filesystem 1
#    define experimental_filesystem
namespace fs = std::experimental::filesystem;
using clk = std::chrono::system_clock;
#  else
#    define have_filesystem 0
#  endif
#endif

static const char* kPSep =
#ifdef _WIN32
R"(\)";
#else
"/";
#endif

/// armadillo
//#define ARMA_64BIT_WORD // enabling 64 integers in armadillo obbjects
//#define ARMA_BLAS_LONG_LONG // using long long inside LAPACK call
//#define ARMA_USE_OPENMP
//#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

// -------------------------------------------------------- DEFINITIONS --------------------------------------------------------

#define stout std::cout << std::setprecision(8) << std::fixed				// standard out
#define el std::endl
#define str std::to_string
#define PRNT(name) valueEquals(#name,(name))
#define PRT(name,prec) valueEquals(#name,(name),prec)

/// using types
using cpx = std::complex<double>;
using uint = unsigned int;
using ul = unsigned long;
using ull = unsigned long long;

/// constexpressions
constexpr long double PI = 3.141592653589793238462643383279502884L;			// it is me, pi
constexpr long double TWOPI = 2 * PI;										// it is me, 2pi
constexpr long double PI_half = PI / 2.0;
constexpr cpx imn = cpx(0, 1);
const std::string kPS = std::string(kPSep);

// -------------------------------------------------------- ALGORITHMS FOR MC --------------------------------------------------------

/*
/// Here we will state all the already implemented definitions that will help us building the user interfrace
*/
namespace impDef {
	/*
	/// Different Monte Carlo algorithms that can be provided inside the classes (for simplicity in enum form)
	*/
	enum class algMC {
		metropolis,
		heat_bath,
		self_learning
	};
	/*
	/// Types of implemented lattice types
	*/
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
/*
* return the duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline double tim_s(clk::time_point start) {
	return double(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration(\
		std::chrono::high_resolution_clock::now() - start)).count()) / 1000.0;
}

// ----------------------------------------------------------------------------- TOOLS -----------------------------------------------------------------------------

//v_1d<double> fourierTransform(std::initializer_list<const arma::mat&> matToTransform, std::tuple<double,double,double> k, std::tuple<int,int,int> L);

// ----------------------------------------------------------------------------- MATRIX MULTIPLICATION

/*
* Puts the given matrix MSet(smaller) to a specific place in the M2Set (bigger) matrix
* @param M2Set (bigger) matrix to find the submatrix in and set it's elements
* @param MSet (smaller) matrix to be put in the M2Set
* @param row row of the left upper element (row,col) of M2Set
* @param col col of the left upper element (row,col) of M2Set
* @param update if we shall add or substract MSet elements from M2Set depending on minus parameter
* @param minus substract?
*/
void setSubmatrixFromMatrix(arma::mat& M2Set, const arma::mat& MSet, uint row, uint col, uint Nrows, uint Ncols, bool update = true, bool minus = false);

/*
* Uses the given matrix MSet (bigger) to set the M2Set (smaller) matrix
* @param M2Set (smaller) matrix to find the submatrix in and set it's elements
* @param MSet (bigger) matrix to be put in the M2Set
* @param row row of the left upper element (row,col) of MSet
* @param col col of the left upper element (row,col) of MSet
* @param update if we shall add or substract MSet elements from M2Set depending on minus parameter
* @param minus substract?
*/
void setMatrixFromSubmatrix(arma::mat& M2Set, const arma::mat& MSet, uint row, uint col, uint Nrows, uint Ncols, bool update = true, bool minus = false);

/*
* Is used to calculate the equation of the form (U_l * D_l * T_l + U_r * D_r * T_r).
* @details UDT we get from QR decomposition with column pivoting
* @param Ql
* @param Rl
* @param Pl
* @param Tl
* @param Dl
* @param Qr
* @param Rr
* @param Pr
* @param Tr
* @param Dr
* @param Dtmp
* @warning Uses the UDT decomposition from QR with column pivoting
*/
arma::mat inv_left_plus_right_qr(arma::mat& Ql, arma::mat& Rl, arma::umat& Pl, arma::mat& Tl, arma::vec& Dl, arma::mat& Qr, arma::mat& Rr, arma::umat& Pr, arma::mat& Tr, arma::vec& Dr, arma::vec& Dtmp);

/*
* Creates the UDT decomposition using QR decomposition. WITH INVERSION OF R DIAGONAL ALREADY
* @cite doi:10.1016/j.laa.2010.06.023
* </summary>
* @param mat 
* @param Q unitary Q matrix
* @param R right triangular matrix
* @param P permutation matrix
* @param T upper triangular matrix
* @param D diagonal vector -> saves the inverse already
*/
void inline setUDTDecomp(const arma::mat& mat, arma::mat& Q, arma::mat& R, arma::umat& P, arma::mat& T, arma::vec& D) {
	if (!arma::qr(Q, R, P, mat)) throw "decomposition failed\n";
	// inverse during setting
	for (int i = 0; i < R.n_rows; i++)
		D(i) = 1.0 / R(i, i);
	T = ((diagmat(D) * R) * P.t());
}


/*
* Using ASvQRD - Accurate Solution via QRD with column pivoting to multiply the QR on the right and multiply new matrix mat_to_multiply on the left side.
* @cite doi:10.1016/j.laa.2010.06.023
* @param mat_to_multiply (left) matrix to multiply by the QR decomposed stuff (on the right)
* @param Q unitary Q matrix
* @param R right triangular matrix
* @param P permutation matrix
* @param T upper triangular matrix
* @param D inverse of the new R diagonal
*/
void inline multiplyMatricesQrFromRight(const arma::mat& mat_to_multiply, arma::mat& Q, arma::mat& R, arma::umat& P, arma::mat& T, arma::vec& D) {
	if (!arma::qr(Q, R, P, (mat_to_multiply * Q) * diagmat(R))) throw "decomposition failed\n";
	// inverse during setting
	for (int i = 0; i < R.n_rows; i++)
		D(i) = 1.0 / R(i, i);
	// premultiply old T by new T from left
	T = ((diagmat(D) * R) * P.t()) * T;
}

/*
* Loh's decomposition to two scales in UDT QR decomposition. One is lower than 0 and second higher. Uses R again to save memory
* @param R the R matrix from QR decompositon. As it's diagonal is mostly not used anymore it will be used to store (<= 1) elements of previous R
* @param D vector to store (> 1) elements of previous R
*/
void inline makeTwoScalesFromUDT(arma::mat& R, arma::vec& D) {
	for (int i = 0; i < R.n_rows; i++)
	{
		if (abs(R(i, i)) > 1)
			R(i, i) = 1;
		else
			D(i) = 1;
	}
}

/*
* Loh's decomposition to two scales in UDT QR decomposition. One is lower than 0 and second higher. Uses two new vectors
* @param R the R matrix from QR decompositon. As it's diagonal is mostly not used anymore it will be used to store (<= 1) elements of previous R
* @param D vector to store (> 1) elements of previous R
*/
void inline makeTwoScalesFromUDT(const arma::mat& R, arma::vec& Db,arma::vec& Ds) {
	Db.ones();
	Ds.ones();
	for (int i = 0; i < R.n_rows; i++)
	{
		if (abs(R(i, i)) > 1)
			Db(i) = R(i,i);
		else
			Ds(i) = R(i,i);
	}
}

// ----------------------------------------------------------------------------- FILE AND STREAMS

/*
* Opens a file
* @param filename filename
* @param mode std::ios_base::openmode
*/
template <typename T>
inline void openFile(T& file, std::string filename, std::ios_base::openmode mode = std::ios::out) {
	file.open(filename, mode);
	if (!file.is_open()) throw "couldn't open a file: " + filename + "\n";
}


/*
* printing the separated number of variables using the variadic functions initializer
*@param output output stream
*@param elements initializer list of the elements to be printed
*@param separator to be used @n default "\\t"
*@param width of one element column for printing
*@param endline shall we add endline at the end?
 */
template <typename T>
inline void printSeparated(std::ostream& output, char separtator = '\t', std::initializer_list<T> elements = {}, arma::u16 width = 8, bool endline = true) {
	for (auto elem : elements) {
		output.width(width); output << elem << std::string(separtator);
	}
	if (endline) output << std::endl;
}

/*
* printing the separated number of variables using the variadic functions initializer
*@param output output stream
*@param separator to be used
*@param width of one element column for printing
*@param endline shall we add endline at the end?
*@param elements at the very end we give any type of variable to the function
*/
template <typename... Types>
inline void printSeparated(std::ostream& output, char separtator, arma::u16 width, bool endline, Types... elements) {
	for (auto elem : elements) {
		output.width(width); output << elem << std::string(separtator);
	}
	if (endline) output << std::endl;
}

/*
* printing the initializer list to a string of format "{"name=value"separator"...""}"
*@param separator to be used
*@param elements initializer list of the elements to be printed
*@param prec precision of a string formatting
*@refitem PRT compiler definition
*/
template <typename T>
inline void printSeparated(char separtator = '\t', std::initializer_list<T> elements = {}, uint prec = 2) {
	std::string tmp = ""
	for (auto elem : elements) {
		tmp += PRT(elem,prec) + std::string(separtator);
	}
	tmp.pop_back();
	return tmp
}

/*
* printing the variadic list to a string of format "{"name=value"separator"...""}"
*@param separator to be used
*@param elements initializer list of the elements to be printed
*@param prec precision of a string formatting
*@refitem PRT compiler definition
*/
template <typename... Types>
inline void printSeparated(char separtator, uint prec, Types... elements) {
	std::string tmp = ""
	for (auto elem : elements) {
		tmp += PRT(elem,prec) + std::string(separtator);
	}
	tmp.pop_back();
	return tmp
}

/*
*Overwritten standard stream redirection operator for 2D vectors separated by commas
*@param out outstream to be used
*@param v 1D vector
*/
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

/*
* Overwritten standard stream redirection operator for 2D vectors
* @param out outstream to be used
* @param v 2D vector
*/
template <typename T>
std::ostream& operator << (std::ostream& out, const v_2d<T>& v ) {
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
// ----------------------------------------------------------------------------- HELPERS

template <typename T>
inline T variance(T value, T average, int norm) {
	return std::sqrt((value / norm - average * average) / norm);
}

/*
* check the sign of a value
* @param val value to be checked
* @returns sign of a variable
*/
template <typename T>
inline int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

/*
* Defines an euclidean modulo denoting also the negative sign
* @param a left side of modulo
* @param b right side of modulo
* @returns euclidean a%b
* @link https://en.wikipedia.org/wiki/Modulo_operation
*/
inline int myModuloEuclidean(int a, int b)
{
	int m = a % b;
	if (m < 0) m = (b < 0) ? m - b : m + b;
	return m;
}

/*
* given the char* name it prints its value in a format "name=val"
*@param name name of the variable
*@param value of the variable
*@returns "name=val" string
*/
template <typename T>
inline std::string ValEquals(char* name, T value, int prec = 2){
	return str(name)+"="+str_p(value,prec);
}

// -------------------------------------------------------- STRING RELATED FUNCTIONS --------------------------------------------------------

/*
*Changes a value to a string with a given precision
*@param a_value Value to be transformed
*@param n Precision @n default 2
*@returns String of a value
*/
template <typename T>
inline std::string str_p(const T a_value, const int n = 2) {
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << a_value;
	return out.str();
}

/*
* Splits string according to the delimiter
* @param s a string to be split
* @param delimiter a delimiter. Default = '\t'
* <returns></returns>
*/
v_1d<std::string> split_str(const std::string& s, std::string delimiter = "\t");

/*
* We want to handle files so let's make the c-way input a string. This way we will parse the command line arguments
* @param argc number of main input arguments 
* @param argv main input arguments 
* @returns vector of strings with the arguments from command line
*/
inline v_1d<std::string> change_input_to_vec_of_str(int argc, char** argv){
	// -1 because first is the name of the file
	std::vector<std::string> tmp(argc - 1, "");										
	for (int i = 0; i < argc - 1; i++)
		tmp[i] = argv[i + 1];
	return tmp;
};
#endif // COMMON_UTILS_H
