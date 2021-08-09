#include "src/common.h"

/// <summary>
/// Defines an euclidean modulo denoting also the negative sign
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <returns></returns>
int myModuloEuclidean(int a, int b)
{
	int m = a % b;
	if (m < 0) {
		m = (b < 0) ? m - b : m + b;
	}
	return m;
}


/* STRING RELATED HELPERS */

/// <summary>
/// Changes a value to a string with a given precision
/// </summary>
/// <param name="a_value">Value to be transformed</param>
/// <param name="n">Precision</param>
/// <returns>String of a value</returns>
template <typename T>
std::string to_string_prec(const T a_value, const int n)
{
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << a_value;
	return out.str();
}
/// <summary>
/// Splits string according to the delimiter
/// </summary>
/// <param name="s">A string to be split</param>
/// <param name="delimiter">A delimiter. Default = '\t'</param>
/// <returns></returns>
std::vector<std::string> split_str(const std::string& s, std::string delimiter) {
	size_t pos_start = 0, pos_end, delim_len = delimiter.length();
	std::string token;
	std::vector<std::string> res;

	while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
		token = s.substr(pos_start, pos_end - pos_start);
		pos_start = pos_end + delim_len;
		res.push_back(token);
	}

	res.push_back(s.substr(pos_start));
	return res;
}
