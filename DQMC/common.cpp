#include "src/common.h"



// -------------------------------------------------------- MATRIX MULTIPLICATION AND ARMA STUFF --------------------------------------------------------
/// <summary>
/// 
/// </summary>
/// <param name="M2Set"></param>
/// <param name="MSet"></param>
/// <param name="row"></param>
/// <param name="col"></param>
/// <param name="update"></param>
/// <param name="minus"></param>
void setSubmatrixFromMatrix(arma::mat& M2Set,  const arma::mat& MSet, uint row, uint col, uint Nrows, uint Ncols,bool update, bool minus) {
	//stout << "\t\tNrows=" << Nrows << ", Ncols=" << Ncols << "\t" << row << "," << col << std::endl;
	//if(row + Nrows > M2Set.n_rows ||  col + Ncols > M2Set.n_cols) throw "incompatible matrix dimensions\n";
	if (update) {
		if (!minus) {
			for (int a = 0; a < Nrows; a++) {
				for (int b = 0; b < Ncols; b++) {
					M2Set(row + a, col + b) += MSet(a, b) ;
				}
			}
		}
		else {
			for (int a = 0; a < Nrows; a++) {
				for (int b = 0; b < Ncols; b++) {
					M2Set(row + a, col + b) -= MSet(a, b) ;
				}
			}
		}
	}
	else {
		for (int a = 0; a < Nrows; a++) {
			for (int b = 0; b < Ncols; b++) {
				M2Set(row + a, col + b)= MSet(a, b) ;
			}
		}
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="M2Set"></param>
/// <param name="MSet"></param>
/// <param name="row"></param>
/// <param name="col"></param>
/// <param name="update"></param>
/// <param name="minus"></param>
void setMatrixFromSubmatrix(arma::mat& M2Set, const arma::mat& MSet, uint row, uint col, uint Nrows, uint Ncols, bool update, bool minus) {
	//if(row + Nrows > MSet.n_rows ||  col + Ncols > MSet.n_cols) throw "incompatible matrix dimensions\n";
	if (update) {
		if (!minus) {
			for (int a = 0; a < Nrows; a++) {
				for (int b = 0; b < Ncols; b++) {
					M2Set(a, b) += MSet(row + a, col + b);
				}
			}
		}
		else {
			for (int a = 0; a < Nrows; a++) {
				for (int b = 0; b < Ncols; b++) {
					M2Set(a, b) -= MSet(row + a, col + b);
				}
			}
		}
	}
	else {
		for (int a = 0; a < Nrows; a++) {
			for (int b = 0; b < Ncols; b++) {
				M2Set(a, b) = MSet(row + a, col + b);
			}
		}
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="Ql"></param>
/// <param name="Rl"></param>
/// <param name="Pl"></param>
/// <param name="Tl"></param>
/// <param name="Dl"></param>
/// <param name="Qr"></param>
/// <param name="Rr"></param>
/// <param name="Pr"></param>
/// <param name="Tr"></param>
/// <param name="Dr"></param>
/// <param name="Dtmp"></param>
arma::mat inv_left_plus_right_qr(arma::mat & Ql, arma::mat & Rl, arma::umat & Pl, arma::mat & Tl, arma::vec & Dl, arma::mat & Qr, arma::mat & Rr, arma::umat & Pr, arma::mat & Tr, arma::vec & Dr, arma::vec & Dtmp)
{
	// using loh
	makeTwoScalesFromUDT(Rl, Dl);															// remember D already inversed!
	makeTwoScalesFromUDT(Rr, Dr);															// remember D already inversed and we use tmp for help!
	
	setUDTDecomp(diagmat(Rl) * Tl * arma::inv(Tr) * diagmat(Dr)\
		+ diagmat(Dl) * Ql.t() * Qr * diagmat(Rr)\
		, Qr, Rl, Pl, Tl, Dtmp);
	setUDTDecomp(diagmat(Dr) * arma::inv(Tl) * diagmat(Dtmp) * Qr.t() * diagmat(Dl), Qr, Rl, Pl, Tl, Dtmp);
	return (arma::inv(Tr)*Qr) * arma::diagmat(Rl) * (Tl * Ql.t());
}


// -------------------------------------------------------- STRING RELATED HELPERS --------------------------------------------------------

/// <summary>
/// We want to handle files so let's make the c-way input a string
/// </summary>
/// <param name="argc"> number of main input arguments </param>
/// <param name="argv"> main input arguments </param>
/// <returns></returns>
v_1d<std::string> change_input_to_vec_of_str(int argc, char** argv)
{
	std::vector<std::string> tmp(argc - 1, "");															// -1 because first is the name of the file
	for (int i = 0; i < argc - 1; i++) {
		tmp[i] = argv[i + 1];
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