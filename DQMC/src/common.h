#pragma once
#include <string>
#include <vector>

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



int myModuloEuclidean(int a, int b);														// Euclidean modulo function denoting also the negative sign

/* STRING RELATED FUNCTIONS */
template <typename T>
std::string to_string_prec(const T a_value, const int n = 3);
std::vector<std::string> split_str(const std::string& s, std::string delimiter = "\t");






#endif // COMMON_UTILS_H
