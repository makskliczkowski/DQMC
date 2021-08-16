#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <ios>
#include <sstream>

/// <summary>
/// Define a path separator for Unix and Windows systems
/// </summary>
static const char* kPSep =
#ifdef _WIN32
"\\";
#else
"/";
#endif

constexpr double PI = 3.14159265359;
constexpr double PI_half = PI / 2.0;
constexpr double TWO_PI = PI * 2;




#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H


template<class T>
using v_3d = std::vector<std::vector<std::vector<T>>>;				// 3d double vector
template<class T>
using v_2d = std::vector<std::vector<T>>;							// 2d double vector
template<class T>
using v_1d = std::vector<T>;										// 1d double vector


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

template <typename T>
inline void printSeparated(std::ostream& output, std::vector<T> elements, std::string separtator = "\t"){
	const int elements_size = elements.size();
#pragma omp critical
	for (int i = 0; i < elements_size - 1; i++) {
		output << elements[i] << separtator;
	}
#pragma omp critical
	output << elements[elements_size - 1] << std::endl;
}
/* STRING RELATED FUNCTIONS */

/// <summary>
/// Changes a value to a string with a given precision
/// </summary>
/// <param name="a_value">Value to be transformed</param>
/// <param name="n">Precision</param>
/// <returns>String of a value</returns>
template <typename T>
inline std::string to_string_prec(const T a_value, const int n = 3){
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << a_value;
	return out.str();
}

v_1d<std::string> split_str(const std::string& s, std::string delimiter = "\t");

v_1d<std::string> change_input_to_vec_of_str(int argc, char** argv);




#endif // COMMON_UTILS_H
