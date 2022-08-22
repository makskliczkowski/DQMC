#include <string>
#include <vector>
#ifdef __has_include
#  if __has_include(<format>)
#    include <format>
#    define HAS_FORMAT 1
#	 define strf std::format
#  else
#    define HAS_FORMAT 0
#  endif
#endif

template<class T>
using v_3d = std::vector<std::vector<std::vector<T>>>;				// 3d double vector
template<class T>
using v_2d = std::vector<std::vector<T>>;							// 2d double vector
template<class T>
using v_1d = std::vector<T>;										// 1d double vector

template<class T>
using t_3d = std::tuple<T, T, T>;									// 3d tuple
template<class T>
using t_2d = std::pair<T, T>;										// 2d tuple - pair


//! -------------------------------------------------------- STRING RELATED FUNCTIONS --------------------------------------------------------

/*
* @brief Splits string according to the delimiter
* @param s a string to be split
* @param delimiter a delimiter. Default = '\\t'
* @return splitted string
*/
inline v_1d<std::string> split_str(const std::string& s, std::string delimiter) {
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

/*
* We want to handle files so let's make the c-way input a string. This way we will parse the command line arguments
* @param argc number of main input arguments 
* @param argv main input arguments 
* @returns vector of strings with the arguments from command line
*/
inline v_1d<std::string> changeInpToVec(int argc, char** argv) {
	// -1 because first is the name of the file
	v_1d<std::string> tmp(argc - 1, "");
	for (int i = 0; i < argc - 1; i++)
		tmp[i] = argv[i + 1];
	return tmp;
};
