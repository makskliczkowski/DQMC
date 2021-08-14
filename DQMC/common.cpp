#include "src/common.h"




/* STRING RELATED HELPERS */
/// <summary>
/// We want to handle files so let's make the c-way input a string
/// </summary>
/// <param name="argc"> number of main input arguments </param>
/// <param name="argv"> main input arguments </param>
/// <returns></returns>
v_1d<std::string> change_input_to_vec_of_str(int argc, char** argv)
{
	std::vector<std::string> tmp(argc-1,"");															// -1 because first is the name of the file
	for(int i = 1; i <argc;i++ ){
		tmp[i] = argv[i];
	}
	return tmp;
}
/// <summary>
/// Splits string according to the delimiter
/// </summary>
/// <param name="s">A string to be split</param>
/// <param name="delimiter">A delimiter. Default = '\t'</param>
/// <returns></returns>
v_1d<std::string> split_str(const std::string& s, std::string delimiter) {
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
