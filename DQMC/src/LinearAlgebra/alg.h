#pragma once
#ifndef ALG_H
#define ALG_H

// armadillo flags:
#define ARMA_USE_LAPACK                                                                     
//#define ARMA_BLAS_LONG_LONG                                                                 // using long long inside LAPACK call
//#define ARMA_DONT_USE_FORTRAN_HIDDEN_ARGS
////#define ARMA_DONT_USE_WRAPPER
#define ARMA_USE_MKL_ALLOC
#define ARMA_USE_MKL_TYPES
#define ARMA_WARN_LEVEL 1
#define ARMA_DONT_USE_OPENMP
////#define ARMA_USE_OPENMP
//#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#define DIAG arma::diagmat
#define EYE(X) arma::eye(X,X)
#define ZEROV(X) arma::zeros(X)
#define ZEROM(X) arma::zeros(X,X)

using uint = unsigned int;
// ----------------------------------------------------------------------------- MATRIX MULTIPLICATION
/*
* @brief Allows to calculate the matrix consisting of Column vector times row vector
* @param setMat matrix to set the elements onto
* @param setVec column vector to set the elements from
*/
template <typename _type>
inline void setColumnTimesRow(arma::Mat<_type>& setMat, const arma::Col<_type>& setVec) {
#pragma omp parallel for
	for (auto i = 0; i < setMat.n_rows; i++)
		for (auto j = 0; j < setMat.n_cols; j++)
			setMat(i, j) = conj(setVec(i)) * setVec(j);
}

/*
* @brief Allows to calculate the matrix consisting of Column vector times row vector but updates it instead of overwritng
* @param setMat matrix to set the elements onto
* @param setVec column vector to set the elements from
* @param plus if add or substract
*/
template <typename _type>
inline void setColumnTimesRow(arma::Mat<_type>& setMat, const arma::Col<_type>& setVec, bool plus) {
	if (plus)
#pragma omp parallel for
		for (auto i = 0; i < setMat.n_rows; i++)
			for (auto j = 0; j < setMat.n_cols; j++)
				setMat(i, j) += conj(setVec(i)) * setVec(j);
	else
#pragma omp parallel for
		for (auto i = 0; i < setMat.n_rows; i++)
			for (auto j = 0; j < setMat.n_cols; j++)
				setMat(i, j) -= conj(setVec(i)) * setVec(j);
}



/*
* @brief Allows to calculate the constant times column vector but updates it instead of overwritng
* @param setCol column to set the elements onto
* @param v value to multiply by
* @param multCol column vector to set the elements from
* @param conjug if we shall conjugate the column
*/
template <typename _type, typename _type2>
inline void setConstTimesCol(arma::Col<_type>& setCol, _type2 v, const arma::Col<_type>& multCol, bool conjug) {
	if (conjug)
#pragma omp parallel for
		for (auto i = 0; i < setCol.n_elem; i++)
			setCol(i) = v * conj(multCol(i));
	else
#pragma omp parallel for
		for (auto i = 0; i < setCol.n_elem; i++)
			setCol(i) = v * multCol(i);
}

/*
* @brief Allows to calculate the constant times column vector but updates it instead of overwritng
* @param setCol column to set the elements onto
* @param v value to multiply by
* @param multCol column vector to set the elements from
* @param conjug if we shall conjugate the column
*/
template <typename _type, typename _type2>
inline void setConstTimesCol(arma::Col<_type>& setCol, _type2 v, const arma::subview_col<_type>& multCol, bool conjug) {
	if (conjug)
#pragma omp parallel for
		for (auto i = 0; i < setCol.n_elem; i++)
			setCol(i) = v * conj(multCol(i));
	else
#pragma omp parallel for
		for (auto i = 0; i < setCol.n_elem; i++)
			setCol(i) = v * multCol(i);
}


/*
* @brief Allows to calculate the constant times column vector but updates it instead of overwritng
* @param setCol column to set the elements onto
* @param v value to multiply by
* @param multCol column vector to set the elements from
* @param plus if add or substract
* @param conjug if we shall conjugate the column
*/
template <typename _type, typename _type2>
inline void setConstTimesCol(arma::Col<_type>& setCol, _type2 v, const arma::Col<_type>& multCol, bool plus, bool conjug) {
	if (plus && conjug)
#pragma omp parallel for
		for (auto i = 0; i < setCol.n_elem; i++)
			setCol(i) += v * conj(multCol(i));
	else if (plus && !conjug)
#pragma omp parallel for
		for (auto i = 0; i < setCol.n_elem; i++)
			setCol(i) += v * multCol(i);
	else if (!plus && conjug)
#pragma omp parallel for
		for (auto i = 0; i < setCol.n_elem; i++)
			setCol(i) -= v * conj(multCol(i));
	else
#pragma omp parallel for
		for (auto i = 0; i < setCol.n_elem; i++)
			setCol(i) -= v * multCol(i);
}

/*
* @brief Allows to calculate the constant times column vector but updates it instead of overwritng
* @param setCol column to set the elements onto
* @param v value to multiply by
* @param multCol column vector to set the elements from
* @param plus if add or substract
* @param conjug if we shall conjugate the column
*/
template <typename _type, typename _type2>
inline void setConstTimesCol(arma::Col<_type>& setCol, _type2 v, const arma::subview_col<_type>& multCol, bool plus, bool conjug) {
	if (plus && conjug)
#pragma omp parallel for
		for (auto i = 0; i < setCol.n_elem; i++)
			setCol(i) += v * conj(multCol(i));
	else if (plus && !conjug)
#pragma omp parallel for
		for (auto i = 0; i < setCol.n_elem; i++)
			setCol(i) += v * multCol(i);
	else if (!plus && conjug)
#pragma omp parallel for
		for (auto i = 0; i < setCol.n_elem; i++)
			setCol(i) -= v * conj(multCol(i));
	else
#pragma omp parallel for
		for (auto i = 0; i < setCol.n_elem; i++)
			setCol(i) -= v * multCol(i);
}




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
* @brief Uses the given matrix MSet (bigger) to set the M2Set (smaller) matrix
* @param M2Set (smaller) matrix to find the submatrix in and set it's elements
* @param MSet (bigger) matrix to be put in the M2Set
* @param row row of the left upper element (row,col) of MSet
* @param col col of the left upper element (row,col) of MSet
* @param update if we shall add or substract MSet elements from M2Set depending on minus parameter
* @param minus substract?
*/
void setMatrixFromSubmatrix(arma::mat& M2Set, const arma::mat& MSet, uint row, uint col, uint Nrows, uint Ncols, bool update = true, bool minus = false);

/*
* @brief Is used to calculate the equation of the form (U_l * D_l * T_l + U_r * D_r * T_r).
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
* @brief Creates the UDT decomposition using QR decomposition. WITH INVERSION OF R DIAGONAL ALREADY
* @cite doi:10.1016/j.laa.2010.06.023
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
	T = ((DIAG(D) * R) * P.t());
}

/*
* @brief Creates the UDT decomposition using QR decomposition. WITHOUT D VECTOR
* @cite doi:10.1016/j.laa.2010.06.023
* @param mat
* @param Q unitary Q matrix
* @param R right triangular matrix
* @param P permutation matrix
* @param T upper triangular matrix
*/
void inline setUDTDecomp(const arma::mat& mat, arma::mat& Q, arma::mat& R, arma::umat& P, arma::mat& T) {
	if (!arma::qr(Q, R, P, mat)) throw "decomposition failed\n";
	// inverse during setting
	T = ((arma::inv(DIAG(R)) * R) * P.t());
}


/*
* @brief Calculate the multiplication of two matrices with numerical stability. USING SVD.
* SciPost Phys. Core 2, 011 (2020)
* @param right right matrix of multiplication
* @param left left matrix of multiplication
* @param Ql
* @param Dl
* @param Tl
* @param Qr
* @param Dr
* @param Tr
* @return matrix after multiplication
*/
arma::mat inline stableMultiplication(const arma::mat& left, const arma::mat& right,
	arma::mat& Ql, arma::vec& Dl, arma::mat& Tl,
	arma::mat& Qr, arma::vec& Dr, arma::mat& Tr)
{	

	// use SVD decomposition for stable multiplication
	arma::svd(Ql, Dl, Tl, left, "std");
	arma::svd(Qr, Dr, Tr, right, "std");
	// decompose the inner side
	arma::svd(Qr, Dr, Tl, Dl * ((Tl * Qr) * Dr), "std");
	return (Ql * Qr) * (Dr) * (Tl * Tr);

}

/*
* @brief Calculate the multiplication of two matrices with numerical stability. USING QR.
* SciPost Phys. Core 2, 011 (2020)
* @param right right matrix of multiplication
* @param left left matrix of multiplication
* @param Ql
* @param Rl
* @param Pl
* @param Tl
* @param Qr
* @param Rr
* @param Pr
* @param Tr
* @return matrix after multiplication
*/
arma::mat inline stableMultiplication(const arma::mat& left, const arma::mat& right,
	arma::mat& Ql, arma::mat& Rl, arma::umat& Pl, arma::mat& Tl,
	arma::mat& Qr, arma::mat& Rr, arma::umat& Pr, arma::mat& Tr)
{

	// use QR decomposition for stable multiplication
	setUDTDecomp(left, Ql, Rl, Pl, Tl);
	setUDTDecomp(right, Qr, Rr, Pr, Tr);

	// decompose the inner side
	setUDTDecomp(DIAG(Rl) * ((Tl * Qr) * DIAG(Rr)), Qr, Rr, Pr, Tl);

	return (Ql * Qr) * (DIAG(Rr)) * (Tl * Tr);
}




/*
* @brief Using ASvQRD - Accurate Solution via QRD with column pivoting to multiply the QR on the right and multiply new matrix mat_to_multiply on the left side.
* @cite doi:10.1016/j.laa.2010.06.023
* @param mat_to_multiply (left) matrix to multiply by the QR decomposed stuff (on the right)
* @param Q unitary Q matrix
* @param R right triangular matrix
* @param P permutation matrix
* @param T upper triangular matrix
* @param D inverse of the new R diagonal
*/
void inline multiplyMatricesQrFromRight(const arma::mat& mat_to_multiply, arma::mat& Q, arma::mat& R, arma::umat& P, arma::mat& T, arma::vec& D) {
	if (!arma::qr(Q, R, P, (mat_to_multiply * Q) * arma::diagmat(R))) throw "decomposition failed\n";
	// inverse during setting
	for (int i = 0; i < R.n_rows; i++)
		D(i) = 1.0 / R(i, i);
	// premultiply old T by new T from left
	T = ((DIAG(D) * R) * P.t()) * T;
}


void inline multiplyMatricesSVDFromRight(const arma::mat& mat_to_multiply, arma::mat& U, arma::vec& s, arma::mat& V, arma::mat& tmpV) {
	svd(U, s, tmpV, mat_to_multiply * U * DIAG(s));
	V = V * tmpV;
}
/*
* @brief Loh's decomposition to two scales in UDT QR decomposition. One is lower than 0 and second higher. Uses R again to save memory
* @param R the R matrix from QR decompositon. As it's diagonal is mostly not used anymore it will be used to store (<= 1) elements of previous R
* @param D vector to store (> 1) elements of previous R -> IT IS ALREADY INVERSE OF R DIAGONAL
*/
void inline makeTwoScalesFromUDT(arma::mat& R, arma::vec& D) {
	for (int i = 0; i < R.n_rows; i++)
	{
		if (abs(R(i, i)) > 1)
			R(i, i) = 1;				// min(1,R(i,i))
			// R(i,i) = 1
			// D(i,i) = 1/R(i,i)
		else
			D(i) = 1;					// inv of max(1,R(i,i))
			// R(i,i) = R(i,i)
			// D(i,i) = 1
	}
}

/*
* @brief Loh's decomposition to two scales in UDT QR decomposition. One is lower than 0 and second higher. Uses two new vectors
* @param R the R matrix from QR decompositon. As it's diagonal is mostly not used anymore it will be used to store (<= 1) elements of previous R
* @param D vector to store (> 1) elements of previous R
*/
void inline makeTwoScalesFromUDT(const arma::mat& R, arma::vec& Db, arma::vec& Ds) {
	Db.ones();
	Ds.ones();
	for (int i = 0; i < R.n_rows; i++)
	{
		if (abs(R(i, i)) > 1)
			Db(i) = R(i, i);
		else
			Ds(i) = R(i, i);
	}
}



#endif