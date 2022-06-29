#include "../src/common.h"

/*v_1d<double> fourierTransform(std::initializer_list<const arma::mat&> matToTransform, std::tuple<double, double, double> k, std::tuple<int, int, int> L) {
	const auto [Lx,Ly,Lz] = L;
	const auto [kx,ky,kz] = k;
	v_1d<double> sum(matToTransform.size(), 0);
	for (int zz = -Lz + 1; zz < Lz; zz++) {
		for (int yy = -Ly + 1; yy < Ly; yy++) {
			const auto yelem = (yy)+Ly - 1;
			for (int xx = -Lx + 1; xx < Lx; xx++) {
				const auto xelem = xx + Lx - 1;
				const cpx exponential = std::exp(im_num * (kx * double(xx) + ky * double(yy) + kz * double(zz)));
				int counter = 0;
				for (auto& elem : matToTransform) {
					sum[counter] += (exponential * elem(xelem, yelem)).real();
					counter++;
				}
			}
		}
	}
	return sum;
}
*/

// -------------------------------------------------------- MATRIX MULTIPLICATION AND ARMA STUFF --------------------------------------------------------

void setSubmatrixFromMatrix(arma::mat& M2Set, const arma::mat& MSet, uint row, uint col, uint Nrows, uint Ncols, bool update, bool minus) {
	//stout << "\t\tNrows=" << Nrows << ", Ncols=" << Ncols << "\t" << row << "," << col << std::endl;
	//if(row + Nrows > M2Set.n_rows ||  col + Ncols > M2Set.n_cols) throw "incompatible matrix dimensions\n";
	if (update)
		if (!minus)
			for (int a = 0; a < Nrows; a++)
				for (int b = 0; b < Ncols; b++)
					M2Set(row + a, col + b) += MSet(a, b);
		else
			for (int a = 0; a < Nrows; a++)
				for (int b = 0; b < Ncols; b++)
					M2Set(row + a, col + b) -= MSet(a, b);
	else
		for (int a = 0; a < Nrows; a++)
			for (int b = 0; b < Ncols; b++)
				M2Set(row + a, col + b) = MSet(a, b);
}

void setMatrixFromSubmatrix(arma::mat& M2Set, const arma::mat& MSet, uint row, uint col, uint Nrows, uint Ncols, bool update, bool minus) {
	//if(row + Nrows > MSet.n_rows ||  col + Ncols > MSet.n_cols) throw "incompatible matrix dimensions\n";
	if (update)
		if (!minus)
			for (int a = 0; a < Nrows; a++)
				for (int b = 0; b < Ncols; b++)
					M2Set(a, b) += MSet(row + a, col + b);
		else 
			for (int a = 0; a < Nrows; a++) 
				for (int b = 0; b < Ncols; b++) 
					M2Set(a, b) -= MSet(row + a, col + b);
	else
		for (int a = 0; a < Nrows; a++)
			for (int b = 0; b < Ncols; b++)
				M2Set(a, b) = MSet(row + a, col + b);
}

/**
 * @brief 
 * 
 * R diagonal has elements smaller than one -> D_m
 * D is already inversed and has elements bigg
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
 * @return arma::mat 
 */
arma::mat inv_left_plus_right_qr(arma::mat& Ql, arma::mat& Rl, arma::umat& Pl, arma::mat& Tl, arma::vec& Dl, arma::mat& Qr, arma::mat& Rr, arma::umat& Pr, arma::mat& Tr, arma::vec& Dr, arma::vec& Dtmp)
{
	const auto loh = false;
	if (loh) {
		// using loh

		makeTwoScalesFromUDT(Rl, Dl);																								// remember D already inversed!
		makeTwoScalesFromUDT(Rr, Dr);																								// remember D already inversed!
		//! D_lm*D_rp^{-1} * X_l * X_r^{-1} + U_l^{-1} * U_r * D_rm * D_lp^{-1}
		setUDTDecomp(
			(DIAG(Rl) * DIAG(Dr)) * Tl * arma::inv(Tr) +
			Ql.t() * Qr * (DIAG(Dl) * DIAG(Rr)),
			Qr, Rl, Pl, Tl, Dtmp);
		//! D_rp^{-1}
		setUDTDecomp(diagmat(Dr) * arma::inv(Tl) * diagmat(Dtmp) * Qr.t() * diagmat(Dl), Qr, Rl, Pl, Tl, Dtmp);
		//? direct inversion
		//setUDTDecomp(DIAG(Dr) * arma::inv(Qr * DIAG(Rl) * Tl) * DIAG(Dl), Qr, Rl, Pl, Tl);
		return (arma::inv(Tr) * Qr) * DIAG(Rl) * (Tl * Ql.t());
	}
	else {
		setUDTDecomp(
			DIAG(Rl) * Tl * arma::inv(Tr) +
			Ql.t() * Qr * DIAG(Rr),
			Qr, Rl, Pl, Tl, Dtmp);
		return arma::inv(Tl * Tr) * DIAG(Dtmp) * arma::inv(Ql * Qr);
	}
}
