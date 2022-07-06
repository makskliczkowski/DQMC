#include "include/hubbard_dqmc_qr.h"
using namespace arma;

void hubbard::HubbardQR::initializeMemory()
{	
	/// hopping exponent
	this->hopping_exp.zeros(this->Ns, this->Ns);

	/// interaction for all times
	this->int_exp_down.ones(this->Ns, this->M);															// for storing M interaction exponents for down spin
	this->int_exp_up.ones(this->Ns, this->M);															// for storing M interaction exponents for up spin

	/// all times exponents multiplication
	this->b_mat_up = v_1d<mat>(this->M, ZEROM(this->Ns));												// for storing M B up matrices
	this->b_mat_down = v_1d<mat>(this->M, ZEROM(this->Ns));												// for storing M B down matrices
	this->b_mat_up_inv = v_1d<mat>(this->M, ZEROM(this->Ns));											// for storing M B up matrices inverses
	this->b_mat_down_inv = v_1d<mat>(this->M, ZEROM(this->Ns));											// for storing M B down matrices inverses

	this->b_up_condensed = v_1d<mat>(this->p, ZEROM(this->Ns));											// for storing the precalculated multiplications of B up matrices series
	this->b_down_condensed = v_1d<mat>(this->p, ZEROM(this->Ns));										// for storing the precalculated multiplications of B down matrices series

	/// all times hs fields for real spin up and down
	this->hsFields.ones(this->M, this->Ns);																// for storing the Hubbard-Strattonovich auxliary fields

	/// Green's function matrix
	this->green_up.zeros(this->Ns, this->Ns);															// for storing equal time Green up matrix
	this->green_down.zeros(this->Ns, this->Ns);															// for storing equal time Green down matrix
	this->tempGreen_up.zeros(this->Ns, this->Ns);														// for storing temporary Green up matrices
	this->tempGreen_down.zeros(this->Ns, this->Ns);														// for storing temporary Green down matrices

	/// decomposition stuff
	this->Q_up.zeros(this->Ns, this->Ns);
	this->Q_down.zeros(this->Ns, this->Ns);
	this->P_up.zeros(this->Ns, this->Ns);
	this->P_down.zeros(this->Ns, this->Ns);
	this->R_up.zeros(this->Ns, this->Ns);
	this->R_down.zeros(this->Ns, this->Ns);
	this->D_down.zeros(this->Ns);
	this->D_up.zeros(this->Ns);
	this->D_tmp.zeros(this->Ns);
	this->T_down.zeros(this->Ns, this->Ns);
	this->T_up.zeros(this->Ns, this->Ns);

	/// set equal time Green's functions vectors
	this->g_down_eq = v_1d<mat>(this->M, EYE(this->Ns));
	this->g_up_eq = v_1d<mat>(this->M, EYE(this->Ns));

#ifdef CAL_TIMES
	//! B matrices inverses
	this->b_up_inv_cond = v_1d<mat>(this->M, ZEROM(this->Ns));
	this->b_down_inv_cond = v_1d<mat>(this->M, ZEROM(this->Ns));

	//! Big Green's functions
	this->g_up_time.eye(Ns * M, Ns * M);
	this->g_down_time.eye(Ns * M, Ns * M);
#endif // CAL_TIMES

}


//! -------------------------------------------------------- B MATS --------------------------------------------------------

/*
* @brief Precalculate the multiplications of B matrices according to M0 stable ones
* @param which_sector
*/
void hubbard::HubbardQR::cal_B_mat_cond(int which_sector)
{
	auto tim = which_sector * this->M_0;
	this->b_down_condensed[which_sector] = this->b_mat_down[tim];
	this->b_up_condensed[which_sector] = this->b_mat_up[tim];
	//#pragma omp parallel for num_threads(this->inner_threads)
	for (int i = 1; i < this->M_0; i++) {
		tim++;
		this->b_down_condensed[which_sector] = this->b_mat_down[tim] * this->b_down_condensed[which_sector];
		this->b_up_condensed[which_sector] = this->b_mat_up[tim] * this->b_up_condensed[which_sector];
	}
}

/*
* @brief We use UDT QR decomposition to decompose the chain multiplication of toMultUp abd toMultDown matrices by B matrices of connected spin from left.
* @param l_start firts time for B matrix to multiply
* @param l_end ending time for the B matrices chain multiplication
* @param toMultUp spin up matrix to multiply from the left
* @param toMultDown spin down matrix to multiply from the right
* @param toSetUp where to set the up multiplication
* @param toSetDown where to set the down multiplication
*/
void hubbard::HubbardQR::b_mat_mult_left(int l_start, int l_end, const mat& toMultUp,const mat& toMultDown, mat& toSetUp, mat& toSetDown)
{
	assert("the hell? they should be different, those times" && l_start != l_end);
	int timer = l_start;
	int step = l_start > l_end ? -1 : 1;

	const int how_many = abs(l_end - l_start);
#ifdef USE_QR
	toSetUp = stableMultiplication(toMultUp, this->b_mat_up[timer], Q_up, R_up, P_up, T_up, Q_down, R_down, P_down, T_down);
	toSetDown = stableMultiplication(toMultDown, this->b_mat_down[timer], Q_up, R_up, P_up, T_up, Q_down, R_down, P_down, T_down);
#elif defined USE_SVD
	toSetUp = stableMultiplication(toMultUp, this->b_mat_up[timer], Q_up, D_up, T_up, Q_down, D_down, T_down);
	toSetDown = stableMultiplication(toMultDown, this->b_mat_down[timer], Q_up, D_up, T_up, Q_down, D_down, T_down);
#else
	toSetUp = toMultUp * this->b_mat_up[timer];
	toSetDown = toMultDown * this->b_mat_down[timer];
#endif // USE_QR
	for(int i = 1; i < how_many; i++){
		const auto prev = timer;
		timer = l_start + step * i;
#ifdef USE_QR
		toSetUp = stableMultiplication(this->b_mat_up[prev], this->b_mat_up[timer], Q_up, R_up, P_up, T_up, Q_down, R_down, P_down, T_down);
		toSetDown = stableMultiplication(this->b_mat_down[prev], this->b_mat_down[timer], Q_up, R_up, P_up, T_up, Q_down, R_down, P_down, T_down);
#elif defined USE_SVD
		toSetUp = stableMultiplication(this->b_mat_up[prev], this->b_mat_up[timer], Q_up, D_up, T_up, Q_down, D_down, T_down);
		toSetDown = stableMultiplication(this->b_mat_down[prev], this->b_mat_down[timer], Q_up, D_up, T_up, Q_down, D_down, T_down);
#else
		toSetUp = this->b_mat_up[prev] * this->b_mat_up[timer];
		toSetDown = this->b_mat_down[prev] * this->b_mat_down[timer];
#endif
	
	}
}

/*
* @brief We use UDT QR decomposition to decompose the chain multiplication of toMultUp abd toMultDown matrices by B_INV matrices of connected spin from left.
* @param l_start firts time for B_INV matrix to multiply
* @param l_end ending time for the B_INV matrices chain multiplication
* @param toMultUp spin up matrix to multiply from the left
* @param toMultDown spin down matrix to multiply from the right
* @param toSetUp where to set the up multiplication
* @param toSetDown where to set the down multiplication
*/
void hubbard::HubbardQR::b_mat_mult_left_inv(int l_start, int l_end, const mat& toMultUp,const mat& toMultDown, mat& toSetUp, mat& toSetDown)
{
	assert("the hell? they should be different, those times" && l_start != l_end);
	int timer = l_start;
	int step = l_start > l_end ? -1 : 1;

	const int how_many = abs(l_end - l_start);
	toSetUp = stableMultiplication(toMultUp, this->b_mat_up_inv[timer], Q_up, R_up, P_up, T_up, Q_down, R_down, P_down, T_down);
	toSetDown = stableMultiplication(toMultDown, this->b_mat_down_inv[timer], Q_up, R_up, P_up, T_up, Q_down, R_down, P_down, T_down);
	for(int i = 1; i < how_many; i++){
		const auto prev = timer;
		timer = l_start + step * i;
		toSetUp = stableMultiplication(this->b_mat_up_inv[prev], this->b_mat_up_inv[timer], Q_up, R_up, P_up, T_up, Q_down, R_down, P_down, T_down);
		toSetUp = stableMultiplication(this->b_mat_down_inv[prev], this->b_mat_down_inv[timer], Q_up, R_up, P_up, T_up, Q_down, R_down, P_down, T_down);
	}
}

//! -------------------------------------------------------- GREENS --------------------------------------------------------

/*
* @brief Compare decomposition created Green's functions with directly calculated
* @param tim at which time shall I compare them?
* @param toll tollerance for them being equal
* @param print_greens shall I print both explicitly?
*/
void hubbard::HubbardQR::compare_green_direct(int tim, double toll, bool print_greens)
{
	this->tempGreen_up.eye();
	this->tempGreen_down.eye();
	// calculate the Green's function directly
	for (int i = 0; i < this->M; i++) {
		this->tempGreen_up = this->b_mat_up[tim] * this->tempGreen_up;
		this->tempGreen_down = this->b_mat_down[tim] * this->tempGreen_down;
		tim = (tim + 1) % this->M;
	}
	this->tempGreen_up = (EYE(this->Ns) + this->tempGreen_up).i();
	this->tempGreen_down = (EYE(this->Ns) + this->tempGreen_down).i();
	if (print_greens)
		this->tempGreen_up.print("temp_green_up\n\n");
	// check equality
	bool up = approx_equal(this->green_up, this->tempGreen_up, "absdiff", toll);
	bool down = approx_equal(this->green_down, this->tempGreen_down, "absdiff", toll);
	stout << " -------------------------------- FOR TIME : " << tim << EL;
	stout << "up Green:\n" << (up ? "THE SAME!" : "BAAAAAAAAAAAAAAAAAAAAAAD!") << EL;
	if (print_greens)
		stout << this->green_up << EL;
	stout << "down Green:\n" << (down ? "THE SAME!" : "BAAAAAAAAAAAAAAAAAAAAAAD!") << EL;
	if (print_greens)
		stout << this->green_down << "\n\n\n";
}

//? -------------------------------------------------------- EQUAL

/*
* Calculate Green with QR decomposition using LOH. Here we calculate the Green matrix at a given time, so we need to take care of the times away from precalculated sectors
* @cite doi:10.1016/j.laa.2010.06.023
* @param which_time The time at which the Green's function is calculated
*/
void hubbard::HubbardQR::cal_green_mat(int which_time) {
	auto tim = which_time;
	int sec = (which_time / this->M_0);							// which sector is used for M_0 multiplication
	int sector_end = (sec + 1) * this->M_0 - 1;
	// multiply those B matrices that are not yet multiplied
	b_mat_mult_left(
		tim + 1, sector_end,
		this->b_mat_up[tim], this->b_mat_down[tim],
		tempGreen_up, tempGreen_down
	);					
	// using tempGreens to store the starting multiplication

	// decomposition
	setUDTDecomp(this->tempGreen_up, Q_up, R_up, P_up, T_up, D_up);
	setUDTDecomp(this->tempGreen_down, Q_down, R_down, P_down, T_down, D_down);

	// multiply by new precalculated sectors
	for (int i = 1; i < this->p - 1; i++)
	{
		sec++;
		if (sec == this->p) sec = 0;
		multiplyMatricesQrFromRight(this->b_up_condensed[sec], Q_up, R_up, P_up, T_up, D_up);
		multiplyMatricesQrFromRight(this->b_down_condensed[sec], Q_down, R_down, P_down, T_down, D_down);
	}
	// we need to handle the last matrices that ale also away from M_0 cycle
	sec++;
	if (sec == this->p) sec = 0;
	sector_end = myModuloEuclidean(which_time - 1, this->M);
	tim = sec * this->M_0;
	b_mat_mult_left(tim + 1, sector_end,
		this->b_mat_up[tim], this->b_mat_down[tim],
		tempGreen_up, tempGreen_down);		
	multiplyMatricesQrFromRight(tempGreen_up, Q_up, R_up, P_up, T_up, D_up);
	multiplyMatricesQrFromRight(tempGreen_down, Q_down, R_down, P_down, T_down, D_down);

	//stout << EL;
	//this->green_up = T_up.i() * (Q_up.t() * T_up.i() + DIAG(R_up)).i()*Q_up.t();
	//this->green_down = T_down.i() * (Q_down.t() * T_down.i() + DIAG(R_down)).i()*Q_down.t();

	// Correction terms
	makeTwoScalesFromUDT(R_up, D_up);
	makeTwoScalesFromUDT(R_down, D_down);
	// calculate equal time Green
	this->green_up = arma::solve(DIAG(D_up) * Q_up.t() + DIAG(R_up) * T_up, DIAG(D_up) * Q_up.t());
	this->green_down = arma::solve(DIAG(D_down) * Q_down.t() + DIAG(R_down) * T_down, DIAG(D_down) * Q_down.t());
}

/*
* Calculate Green with QR decomposition using LOH : doi:10.1016/j.laa.2010.06.023 with premultiplied B matrices. 
* For more look into :
* @copydetails "Advancing Large Scale Many-Body QMC Simulations on GPU Accelerated Multicore Systems". 
* In order to do that the M_0 and p variables will be used to divide the multiplication into smaller chunks of matrices. 
* @param sector Which sector does the Green's function starrts at
*/
void hubbard::HubbardQR::cal_green_mat_cycle(int sector) {
	auto sec = sector;
	setUDTDecomp(this->b_up_condensed[sec], Q_up, R_up, P_up, T_up, D_up);
	setUDTDecomp(this->b_down_condensed[sec], Q_down, R_down, P_down, T_down, D_down);
	for (int i = 1; i < this->p; i++) {
		sec++;
		if (sec == this->p) sec = 0;
		multiplyMatricesQrFromRight(this->b_up_condensed[sec], Q_up, R_up, P_up, T_up, D_up);
		multiplyMatricesQrFromRight(this->b_down_condensed[sec], Q_down, R_down, P_down, T_down, D_down);
	}
	// making two scales for the decomposition following Loh
	makeTwoScalesFromUDT(R_up, D_up);
	makeTwoScalesFromUDT(R_down, D_down);
	this->green_up = arma::solve(DIAG(D_up) * Q_up.t() + DIAG(R_up) * T_up, DIAG(D_up) * Q_up.t());
	this->green_down = arma::solve(DIAG(D_down) * Q_down.t() + DIAG(R_down) * T_down, DIAG(D_down) * Q_down.t());
}

//? -------------------------------------------------------- UNEQUAL

/*
* @brief Calculating unequal time Green's functions given by Bl_1*...*B_{l2+1}*G_{l2+1} \\rightarrow [B_{l2+1}^{-1}...B_l1^{-1} + B_l2...B_1B_{M-1}...B_{l1+1}]^{-1}. 
* Make inverse of function of type (Ql*diag(Rl)*Tl + Qr*diag(Rr)*Tr)^(-1) using:
* @cite SciPost Phys. Core 2, 011 (2020) 
* @param t1 left time t1>t2
* @param t2 right time t2<t1
* @param inv_series_up precalculated inverse matrices multiplication for spin up
* @param inv_series_down precalculated inverse matrices multiplication for spin down
*/
void hubbard::HubbardQR::uneqG_t1gtt2(int t1, int t2, const mat& inv_up, const mat& inv_down, const mat& up, const mat& down)
{
	assert("t1 should be higher than t2" && t1 >= t2);
	const auto row = t1 * this->Ns;
	const auto col = t2 * this->Ns;

	//! ------------------------------------ up ------------------------------------ 
	//? USE DOWN MATRICES AS HELPERS FOR RIGHT SUM TO SAVE PRECIOUS MEMORY!
	//! B(t2 + 1)^(-1)...B(t1)^(-1)
	setUDTDecomp(inv_up, Q_up, R_up, P_up, T_up, D_up);												// decompose the premultiplied inversions to up temporaries

	//! B(M-1)...B(t1 + 1)
	setUDTDecomp(up, Q_down, R_down, P_down, T_down, D_down);										// decompose and use down matrices as temporaries + equal time Green at [0]

	//! SET MATRIX ELEMENT
	setSubmatrixFromMatrix(this->g_up_time,
						inv_left_plus_right_qr(
							Q_up, R_up, P_up, T_up, D_up,
							Q_down, R_down, P_down, T_down, D_down,
							D_tmp
						),
						row, col, this->Ns, this->Ns, false);

	//! ------------------------------------ down ------------------------------------
	//? USE UP MATRICES AS HELPERS FOR RIGHT SUM TO SAVE PRECIOUS MEMORY!
	//! B(t2 + 1)^(-1)...B(t1)^(-1)
	setUDTDecomp(inv_down, Q_up, R_up, P_up, T_up, D_up);											// decompose the premultiplied inversions to up temporaries
	
	//! B(M-1)...B(t1 + 1)
	setUDTDecomp(down, Q_down, R_down, P_down, T_down, D_down);

	//! SET MATRIX ELEMENT
	setSubmatrixFromMatrix(this->g_down_time,
						inv_left_plus_right_qr(
							Q_up, R_up, P_up, T_up, D_up,
							Q_down, R_down, P_down, T_down, D_down,
							D_tmp
						),
						row, col, this->Ns, this->Ns, false);
}

//TODO ----------------------->
/**
* @param t1
* @param t2
*/
void hubbard::HubbardQR::uneqG_t1ltt2(int t1, int t2)
{
	if (t2 <= t1) throw "can't do that m8\n";
	// make inverse of function of type (Ql*diag(Rl)*Tl + Qr*diag(Rr)*Tr)^(-1) using SciPost Phys. Core 2, 011 (2020)
	const auto row = t1 * this->Ns;
	const auto col = t2 * this->Ns;

	// ------------------------------------ up ------------------------------------ USE DOWN MATRICES AS HELPERS FOR RIGHT SUM!
	// B(l2)...B(l1+1)
	//setUDTDecomp(inv_series_up, Q_up, R_up, P_up, T_up, D_up);
	this->tempGreen_up = arma::inv(T_up) * DIAG(D_up) * Q_up.t();
	// B(M-1)...B(t1 + 1)
	//setUDTDecomp(this->g_up_eq[t1], Q_down, R_down, P_down, T_down, D_down);
	setUDTDecomp(this->g_up_eq[0], Q_down, R_down, P_down, T_down, D_down);
	// B(t2)...B(0)
	//multiplyMatricesQrFromRight(this->g_up_tim[t2], Q_down, R_down, P_down, T_down, D_down);
	// SET MATRIX ELEMENT
	setUDTDecomp(inv_left_plus_right_qr(Q_up, R_up, P_up, T_up, D_up, \
		Q_down, R_down, P_down, T_down, D_down, D_tmp), Q_up, R_up, P_up, T_up, D_up);
	setUDTDecomp(this->tempGreen_up, Q_down, R_down, P_down, T_down, D_down);
	setUDTDecomp(DIAG(R_up) * T_up * T_down.i() - Q_up.t() * Q_down * DIAG(R_down), Q_down, R_up, P_up, T_up, D_up);

	setSubmatrixFromMatrix(this->g_up_time, (Q_up * Q_down) * DIAG(R_up) * (T_up * T_down), row, col, this->Ns, this->Ns, false);

	// ------------------ down ------------------
	// B(l2)...B(l1+1)
	//setUDTDecomp(inv_series_down, Q_down, R_down, P_down, T_down, D_down);
	this->tempGreen_down = arma::inv(T_down) * DIAG(D_down) * Q_down.t();
	// B(M-1)...B(t1 + 1)
	//setUDTDecomp(this->g_down_eq[t1], Q_up, R_up, P_up, T_up, D_up);
	setUDTDecomp(this->g_down_eq[0], Q_up, R_up, P_up, T_up, D_up);
	// B(t2)...B(0)
	//multiplyMatricesQrFromRight(this->g_down_tim[t2], Q_up, R_up, P_up, T_up, D_up);
	// SET MATRIX ELEMENT
	setUDTDecomp(inv_left_plus_right_qr(Q_down, R_down, P_down, T_down, D_down, \
		Q_up, R_up, P_up, T_up, D_up, D_tmp), Q_down, R_down, P_down, T_down, D_down);
	setUDTDecomp(this->tempGreen_up, Q_up, R_up, P_up, T_up, D_up);
	setUDTDecomp(DIAG(R_down) * T_down * T_up.i() - Q_down.t() * Q_up * DIAG(R_up), Q_up, R_down, P_down, T_down, D_down);

	setSubmatrixFromMatrix(this->g_down_time, Q_down * Q_up * DIAG(R_down) * T_down * T_up, row, col, this->Ns, this->Ns, false);
}

// TODO 
/*
* @brief Calculate time displaced Greens. NOW ONLY t1>t2
* @TODO make t2>t1
*/
void hubbard::HubbardQR::cal_green_mat_times()
{
	//! inverses precalculated according to the second time
	//? B(t2 + 1)^(-1)...B(t1)^(-1)
	if(this->M > 2){
		// it usually is but keep it tho'
		this->b_up_inv_cond[this->M-2] = this->b_mat_up_inv[this->M-1];
		this->b_down_inv_cond[this->M-2] = this->b_mat_down_inv[this->M-1];
		setUDTDecomp(this->b_up_inv_cond[this->M - 2], Q_up, R_up, P_up, T_up, D_up);
		setUDTDecomp(this->b_down_inv_cond[this->M - 2], Q_down, R_down, P_down, T_down, D_down);
	}
	int counter = 1;
	for(int t2 = this->M-3; t2 >= 0; t2--){
		multiplyMatricesQrFromRight(b_mat_up_inv[t2+1], Q_up, R_up, P_up, T_up, D_up);
		this->b_up_inv_cond[t2] = Q_up * DIAG(R_up) * T_up;
		multiplyMatricesQrFromRight(b_mat_down_inv[t2+1], Q_down, R_down, P_down, T_down, D_down);
		this->b_down_inv_cond[t2] = Q_down * DIAG(R_down) * T_down;
	}
	//! non-inverses -> do the same, use g_eq as it is unused
	//TODO change the name of g_eq probably
	//? B(t2)...B(0)B(M-1)...B(t1+1)
	// we will cover the stuff for the l2's
	this->g_up_eq[0] = this->b_mat_up[0];
	this->g_down_eq[0] = this->b_mat_down[0];
	setUDTDecomp(this->g_up_eq[0], Q_up, R_up, P_up, T_up, D_up);
	setUDTDecomp(this->g_down_eq[0], Q_down, R_down, P_down, T_down, D_down);
	for (int t2 = 1; t2 < this->M - 2; t2++) {
		multiplyMatricesQrFromRight(this->b_mat_up[t2], Q_up, R_up, P_up, T_up, D_up);
		this->g_up_eq[t2] = Q_up * DIAG(R_up) * T_up;
		multiplyMatricesQrFromRight(this->b_mat_down[t2], Q_down, R_down, P_down, T_down, D_down);
		this->g_down_eq[t2] = Q_down * DIAG(R_down) * T_down;
	}
	//! calculate all Greens but only for t1 > t2
	// inverses start from highest on right
	// non-inverses start from 0 basically
	// so we need to multiply by according B(t1) after first loop
	for (int tau1 = this->M - 1; tau1 > 0; tau1--) {
		for (int tau2 = tau1 - 1; tau2 >= 0; tau2--) {
			mat& inv_up = this->b_up_inv_cond[tau2];
			mat& inv_down = this->b_down_inv_cond[tau2];
			mat& up = this->g_up_eq[tau2];
			mat& down = this->g_down_eq[tau2];
			uneqG_t1gtt2(tau1,tau2,inv_up,inv_down,up,down);
			// cause this one won't be used again
			if (tau2 != tau1 - 1) {
				/// up
				setUDTDecomp(this->b_mat_up[tau1], Q_up, R_up, P_up, T_up, D_up);				// use ups for inverses
				Q_down = Q_up; R_down = R_up; P_down = P_up; T_down = T_up; D_down = D_up;		// use downs for noninverses
				multiplyMatricesQrFromRight(inv_up, Q_up, R_up, P_up, T_up, D_up);
				inv_up = Q_up * DIAG(R_up) * T_up;												// use ups for inverses
				multiplyMatricesQrFromRight(up, Q_down, R_down, P_down, T_down, D_down);
				up = Q_down * DIAG(R_down) * T_down;											// use downs for noninverses

				/// down 
				setUDTDecomp(this->b_mat_down[tau1], Q_up, R_up, P_up, T_up, D_up);
				Q_down = Q_up; R_down = R_up; P_down = P_up; T_down = T_up; D_down = D_up;		// use downs for noninverses
				multiplyMatricesQrFromRight(inv_down, Q_up, R_up, P_up, T_up, D_up);
				inv_down = Q_up * DIAG(R_up) * T_up;											// use ups for inverses
				multiplyMatricesQrFromRight(down, Q_down, R_down, P_down, T_down, D_down);
				down = Q_down * DIAG(R_down) * T_down;											// use downs for noninverses


				//inv_up = this->multiplyMatrices(inv_up, this->b_mat_up[tau1], true);
				//inv_down = this->multiplyMatrices(inv_down, this->b_mat_down[tau1], true);
				//up = this->multiplyMatrices(up, this->b_mat_up[tau1], true);
				//down = this->multiplyMatrices(down, this->b_mat_down[tau1], true);
			}
		}
	}
}

/*
* Calculates the unequal times Green's functions using precalculated B matrices
*/
void hubbard::HubbardQR::cal_green_mat_times_cycle()
{
	//! inverses precalculated according to the second time
	//? B(t2 + 1)^(-1)...B(t1)^(-1)
	if (this->M > 2) {
		// it usually is but keep it tho'
		this->b_up_inv_cond[this->M - 2] = this->b_mat_up_inv[this->M - 1];
		this->b_down_inv_cond[this->M - 2] = this->b_mat_down_inv[this->M - 1];
		//svd(Q_up, D_up, R_up, this->b_up_inv_cond[this->M - 2]);
		//svd(Q_down, D_down, R_down, this->b_down_inv_cond[this->M - 2]);
	}
	int counter = 1;
	for (int t2 = this->M - 3; t2 >= 0; t2--) {
		counter++;
		if (counter < this->M_0) {
			this->b_up_inv_cond[t2] = b_mat_up_inv[t2 + 1] * this->b_up_inv_cond[t2 + 1];
			this->b_down_inv_cond[t2] = b_mat_down_inv[t2 + 1] * this->b_down_inv_cond[t2 + 1];
		}
		else if (counter == this->M_0) {
			svd(Q_up, D_up, R_up, this->b_up_inv_cond[t2+1]);
			svd(Q_down, D_down, R_down, this->b_down_inv_cond[t2+1]);
		}
		if (counter >= this->M_0) {
			multiplyMatricesSVDFromRight(b_mat_up_inv[t2 + 1], Q_up, D_up, R_up, T_up);
			this->b_up_inv_cond[t2] = Q_up * DIAG(D_up) * R_up.t();
			multiplyMatricesSVDFromRight(b_mat_down_inv[t2 + 1], Q_down, D_down, R_down, T_down);
			this->b_down_inv_cond[t2] = Q_down * DIAG(D_down) * R_down.t();
		}
	}
	//! non-inverses -> do the same, use g_eq as it is unused
	//TODO change the name of g_eq probably
	//? B(t2)...B(0)B(M-1)...B(t1+1)
	// we will cover the stuff for the l2's
	this->g_up_eq[0] = this->b_mat_up[0];
	this->g_down_eq[0] = this->b_mat_down[0];
	//svd(Q_up, D_up, R_up, this->g_up_eq[0]);
	//svd(Q_down, D_down, R_down, this->g_down_eq[0]);
	counter = 1;
	for (int t2 = 1; t2 < this->M - 2; t2++) {
		counter++;
		if (counter < this->M_0) {
			this->g_up_eq[t2] = this->b_mat_up[t2] * this->g_up_eq[t2 - 1];
			this->g_down_eq[t2] = this->b_mat_down[t2] * this->g_down_eq[t2 - 1];
		}
		else if (counter == this->M_0) {
			svd(Q_up, D_up, R_up, this->g_up_eq[t2-1]);
			svd(Q_down, D_down, R_down, this->g_down_eq[t2-1]);
		}
		if (counter >= this->M_0) {
			multiplyMatricesSVDFromRight(this->b_mat_up[t2], Q_up, D_up, R_up, T_up);
			this->g_up_eq[t2] = Q_up * DIAG(D_up) * R_up.t();
			multiplyMatricesSVDFromRight(this->b_mat_down[t2], Q_down, D_down, R_down, T_down);
			this->g_down_eq[t2] = Q_down * DIAG(D_down) * R_down.t();
		}
	}
	//! calculate all Greens but only for t1 > t2
	// inverses start from highest on right
	// non-inverses start from 0 basically
	// so we need to multiply by according B(t1) after first loop
	for (int tau1 = this->M - 1; tau1 > 0; tau1--) {
		for (int tau2 = tau1 - 1; tau2 >= 0; tau2--) {
			uneqG_t1gtt2(tau1, tau2,
				b_up_inv_cond[tau2],
				b_down_inv_cond[tau2],
				g_up_eq[tau2],
				g_down_eq[tau2]);

			/// up
			svd(Q_up, D_up, R_up, this->b_mat_up[tau1]);									// use ups for inverses
			Q_down = Q_up; R_down = R_up; D_down = D_up;									// use downs for noninverses
			multiplyMatricesSVDFromRight(b_up_inv_cond[tau2], Q_up, D_up, R_up, T_up);
			b_up_inv_cond[tau2] = Q_up * DIAG(D_up) * R_up.t();								// use ups for inverses
			multiplyMatricesSVDFromRight(g_up_eq[tau2], Q_down, D_down, R_down, T_down);
			g_up_eq[tau2] = Q_down * DIAG(D_down) * R_down.t();								// use downs for noninverses

			/// down 
			svd(Q_up, D_up, R_up, this->b_mat_down[tau1]);
			Q_down = Q_up; R_down = R_up; D_down = D_up;									// use downs for noninverses
			multiplyMatricesSVDFromRight(b_down_inv_cond[tau2], Q_up, D_up, R_up, T_up);
			b_down_inv_cond[tau2] = Q_up * DIAG(D_up) * R_up.t();							// use ups for inverses
			multiplyMatricesSVDFromRight(g_down_eq[tau2], Q_down, D_down, R_down, T_down);
			g_down_eq[tau2] = Q_down * DIAG(D_down) * R_down.t();							// use downs for noninverses


			//inv_up = this->multiplyMatrices(inv_up, this->b_mat_up[tau1], true);
			//inv_down = this->multiplyMatrices(inv_down, this->b_mat_down[tau1], true);
			//up = this->multiplyMatrices(up, this->b_mat_up[tau1], true);
			//down = this->multiplyMatrices(down, this->b_mat_down[tau1], true);
		}
	}
}

/*
* Use the space-time formulation for Green's function calculation. 
* * Inversion can be unstable
* @cite Stable Monte Carlo algorit&sn for fermion lattice systems at low temperatures
*/
void hubbard::HubbardQR::cal_green_mat_times_hirsh()
{
	assert("should allow time calculations" && this->cal_times);
	this->g_up_time.eye();
	this->g_down_time.eye();

	setSubmatrixFromMatrix(g_up_time, this->b_mat_up[this->M - 1], 0, (this->M - 1) * this->Ns, this->Ns, this->Ns, false);
	setSubmatrixFromMatrix(g_down_time, this->b_mat_down[this->M - 1], 0, (this->M - 1) * this->Ns, this->Ns, this->Ns, false);
#pragma omp parallel for num_threads(this->inner_threads)
	for (int sec = 0; sec < this->M - 1; sec++) {
		const auto row = (sec + 1) * this->Ns;
		const auto col = (sec)*this->Ns;
		setSubmatrixFromMatrix(g_up_time, this->b_mat_up[sec], row, col, this->Ns, this->Ns, true, true);
		setSubmatrixFromMatrix(g_down_time, this->b_mat_down[sec], row, col, this->Ns, this->Ns, true, true);
	}
	//this->g_up_time = arma::solve(this->g_up_time, DIAG(this->g_up_time));
	//this->g_down_time = arma::solve(this->g_down_time, DIAG(this->g_down_time));
	this->g_up_time = this->g_up_time.i();
	this->g_down_time = this->g_down_time.i();
}

/**
* Use the space-time formulation to calculate only M0*M0 Green's at the same time
*/
void hubbard::HubbardQR::cal_green_mat_times_hirsh_cycle()
{
	assert("should allow time calculations" && this->cal_times);

	this->g_up_time.eye();
	this->g_down_time.eye();

	setSubmatrixFromMatrix(g_up_time, this->b_up_condensed[0], 0, (this->p - 1) * this->Ns, this->Ns, this->Ns, false);
	setSubmatrixFromMatrix(g_down_time, this->b_down_condensed[0], 0, (this->p - 1) * this->Ns, this->Ns, this->Ns, false);
	for (int sec = 0; sec < this->p - 1; sec++) {
		const auto row = (sec + 1) * this->Ns;
		const auto col = (sec)*this->Ns;
		setSubmatrixFromMatrix(g_up_time, this->b_up_condensed[sec + 1], row, col, this->Ns, this->Ns, true, true);
		setSubmatrixFromMatrix(g_down_time, this->b_down_condensed[sec + 1], row, col, this->Ns, this->Ns, true, true);
	}
	this->g_up_time = arma::inv(this->g_up_time);
	this->g_down_time = arma::inv(this->g_down_time);
}

//! -------------------------------------------------------- HELPERS

/*
* @brief A function to sweep all the auxliary Ising fields for a given time configuration in the model
* @param fptr pointer to single update try
* @return sign of the configuration 
*/
int hubbard::HubbardQR::sweep_lat_sites()
{
	int sign = 1;
	for (int j = 0; j < this->Ns; j++){
		sign = heat_bath_single_step(j);
		//if (sign < 0) stout << VEQ(sign) << EL;
	}
	// return sign from the last possible flip
	return sign;
}

/*
* @brief Choose between stable multiplication of matrices according to condition
* @param left left matrix to multiply
* @param right right matrix to multiply
* @param stable should use stable one?
* @return left*right
*/
mat hubbard::HubbardQR::multiplyMatrices(const mat& left, const mat& right, bool stable){
	if(stable)
		return stableMultiplication(left, right,
							Q_up, R_up, P_up, T_up,
							Q_down, R_down, P_down, T_down);
	else
		return left * right;
}


//! -------------------------------------------------------- GREEN UPDATERS --------------------------------------------------------

/*
* @brief After changing one spin we need to update the Green matrices via the Dyson equation
* @param lat_site the site on which HS field has been changed
* @param gamma_over_prob_up changing parameter gamma for up channel over the changing probability for up channel
* @param gamma_over_prob_down changing parameter gamma for down channel over the changing probability for down channel 
*/
void hubbard::HubbardQR::upd_equal_green(int lat_site, double gamma_over_prob_up, double gamma_over_prob_down)
{
	this->D_up = this->green_up.row(lat_site).t();
	this->D_down = this->green_down.row(lat_site).t();
	//#pragma omp parallel for num_threads(this->inner_threads)
	for (int a = 0; a < this->Ns; a++) {
		const auto delta = (a == lat_site) ? 1 : 0;
		const auto g_ai_up = this->green_up(a, lat_site);
		const auto g_ai_down = this->green_down(a, lat_site);
		for (int b = 0; b < this->Ns; b++) {
			// SPIN UP
			this->green_up(a, b) -= (delta - g_ai_up) * gamma_over_prob_up * D_up(b);
			// SPIN DOWN
			this->green_down(a, b) -= (delta - g_ai_down) * gamma_over_prob_down * D_down(b);
		}
	}
}

/*
* @brief Update the Green's matrices after going to next Trotter time, remember, the time is taken to be the previous one
* @param which_time updating to which_time + 1
*/
void hubbard::HubbardQR::upd_next_green(int which_time_green) {
	//this->green_up = (this->b_mat_up[which_time_green] * this->green_up) * this->b_mat_up[which_time_green].i();						// LEFT INCREASE
	//this->green_down = (this->b_mat_down[which_time_green] * this->green_down) * this->b_mat_down[which_time_green].i();
	this->green_up = (this->b_mat_up[which_time_green] * this->green_up) * this->b_mat_up_inv[which_time_green];						// LEFT INCREASE
	this->green_down = (this->b_mat_down[which_time_green] * this->green_down) * this->b_mat_down_inv[which_time_green];				// LEFT INCREASE;
}

/*
* @brief Update the Green's matrices after going to previous Trotter time, remember, the time is taken to be the previous one
* @param which_time updating to which_time - 1
*/
void hubbard::HubbardQR::upd_prev_green(int which_time_green) {
	this->green_up = (this->b_mat_up_inv[which_time_green - 1] * this->green_up) * this->b_mat_up[which_time_green - 1];						// LEFT INCREASE
	this->green_down = (this->b_mat_down_inv[which_time_green - 1] * this->green_down) * this->b_mat_down[which_time_green - 1];				// LEFT INCREASE;
}

// ----------------------------------------------------------------------------------------------------------------

/*
* @brief How to update the Green's function to the next time. 
* If the time step % from_scratch == 0 -> we recalculate from scratch.
* @param im_time_step Current time step that needs to be propagated
* @param forward @todo this
*/
void hubbard::HubbardQR::upd_Green_step(int im_time_step, bool forward) {
	if (im_time_step % this->from_scratch == 0) {
		const auto sector_to_upd = myModuloEuclidean(static_cast<int>(im_time_step / double(this->M_0)) - 1, this->p);		// choose the sector to update
		this->cal_B_mat_cond(sector_to_upd);																				// recalculate the condensed matrices
		this->cal_green_mat_cycle(myModuloEuclidean(static_cast<int>(im_time_step / double(this->M_0)), this->p));			// calculate the greens in cycle
	}
	else
		this->upd_next_green(im_time_step - 1);
}
// -------------------------------------------------------- CALCULATORS

/*
* @brief A single step for calculating averages inside a loop
* @param current_elem_i current Ising spin
* @param sign current sign to multiply the averages
* @param times if we calculate times as well
*/
void hubbard::HubbardQR::av_single_step(int current_elem_i, int sign)
{
	// -------------------------------- ONE SITE PARAMS ----------------------------------------

	// m_z
	this->calOneSiteParam(sign, current_elem_i, cal_mz2, this->avs->av_M2z, this->avs->sd_M2z);
	// m_x
	this->calOneSiteParam(sign, current_elem_i, cal_mx2, this->avs->av_M2x, this->avs->sd_M2x);
	// occupation
	this->calOneSiteParam(sign, current_elem_i, cal_occupation, this->avs->av_occupation, this->avs->sd_occupation);
	// kinetic energy
	auto Ek = this->cal_kinetic_en(sign, current_elem_i, this->green_up, this->green_down);
	this->avs->av_Ek = Ek;
	this->avs->sd_Ek = Ek * Ek;

	// -------------------------------- CORRELATIONS ----------------------------------------
	for (int current_elem_j = 0; current_elem_j < this->Ns; current_elem_j++) {
		// real space coordinates differences
		auto [x, y, z] = this->lattice->getSiteDifference(current_elem_i, current_elem_j);
		auto [xx, yy, zz] = this->lattice->getSymPos(x, y, z);

		auto xi = x + Lx - 1;
		auto yi = y + Ly - 1;
		auto zi = z + Lz - 1;

		//? normal equal - time correlations
		this->avs->av_M2z_corr[xi][yi][zi] += this->cal_mz2_corr(sign, current_elem_i, current_elem_j, this->green_up, this->green_down);
		this->avs->av_occupation_corr[xi][yi][zi] += this->cal_occupation_corr(sign, current_elem_i, current_elem_j, this->green_up, this->green_down);
		this->avs->av_ch2_corr[xi][yi][zi] += this->cal_ch_correlation(sign, current_elem_i, current_elem_j, this->green_up, this->green_down) / (this->Ns * 2.0);

			
		/// time difference different than 0
		//! we handle it
#ifdef CAL_TIMES
		//? handle zero time difference here in greens
		//! we handle it with the calculated current Green's functions
		this->avs->g_up_diffs[0](xx, yy) += this->green_up(current_elem_i, current_elem_j);
		this->avs->g_down_diffs[0](xx, yy) += this->green_down(current_elem_i, current_elem_j);
		this->avs->sd_g_up_diffs[0](xx, yy) += this->green_up(current_elem_i, current_elem_j) * this->green_up(current_elem_i, current_elem_j);
		this->avs->sd_g_down_diffs[0](xx, yy) += this->green_down(current_elem_i, current_elem_j) * this->green_down(current_elem_i, current_elem_j);
		
		for (int time2 = 0; time2 < this->M; time2++) {
			auto tim = (this->current_time - time2);

			// check if we include time symmetry
			if(tim == 0 || (tim < 0 && !this->all_times)) continue;
			int xi = 1;
			// handle antiperiodicity
			if (tim < 0) {
				xi = -1;
				tim += this->M;
			}
			const auto col = time2 * this->Ns;
			const auto row = this->current_time * this->Ns;
			const auto up_elem = xi * this->g_up_time(row + current_elem_i, col + current_elem_j);
			const auto down_elem = xi * this->g_down_time(row + current_elem_i, col + current_elem_j);
			// save only the positive first half
			this->avs->g_up_diffs[tim](xx, yy) += up_elem;
			this->avs->g_down_diffs[tim](xx, yy) += down_elem;
			this->avs->sd_g_up_diffs[tim](xx, yy) += up_elem * up_elem;
			this->avs->sd_g_down_diffs[tim](xx, yy) += down_elem * down_elem;
		}
#endif
	}
}
// ---------------------------------------------------------------------------------------------------------------- HEAT BATH ----------------------------------------------------------------------------------------------------------------

/*
* @brief sweep space-time from time 0 to time M
* @param ptfptr function for lattice sweep - can be heat-bath based etc
*/
void hubbard::HubbardQR::sweep_0_M()
{
	// update configuration sign to check the changes
	this->config_sign = 1;
	for (int time_im = 0; time_im < this->M; time_im++) {
		this->current_time = time_im;
		this->upd_Green_step(this->current_time, true);

		//this->compare_green_direct(this->current_time, 1e-4, true);
		this->config_sign = (this->sweep_lat_sites() > 0) ? +this->config_sign : -this->config_sign;
	}
}

/*
* @brief sweep space-time from time M to time 0
* @param ptfptr function for lattice sweep - can be heat-bath based etc
*/
void hubbard::HubbardQR::sweep_M_0()
{
	int sign = this->config_sign;
	for (int time_im = this->M - 1; time_im >= 0; time_im--) {
		// imaginary Trotter times
		this->current_time = time_im;//tim[time_im];
		sign = sweep_lat_sites() > 0 ? +sign : -sign;
		this->upd_Green_step(time_im, false);
	}
	this->config_sign = sign;// (sign == 1) ? +this->config_sign : -this->config_sign;
}

/*
* @brief heat - bath based algorithm for the propositon of HS field spin flip
* @param lat_site site at which we try
* @return sign of the probility
*/
int hubbard::HubbardQR::heat_bath_single_step(int lat_site)
{
	auto [gamma_up, gamma_down] = this->cal_gamma(lat_site);										// first up then down
	auto [proba_up, proba_down] = this->cal_proba(lat_site, gamma_up, gamma_down);					// take the probabilities

	this->probability = (proba_up * proba_down);													// Metropolis probability
	if (this->U < 0) {
		this->probability *= (this->gammaExp0.second + 1.0);										// add phase factor for U<0
	}
	
	this->probability = this->probability / (1.0 + this->probability);								// heat-bath probability

	const int sign = (this->probability >= 0) ? 1 : -1;												// check sign
	//if (sign < 0) stout << VEQ(proba_up) << "," << VEQ(proba_down) << EL;
	if (this->ran.randomReal_uni() <= sign * this->probability) {
		const auto delta_up = gamma_up + 1;
		const auto delta_down = gamma_down + 1;
		this->hsFields(this->current_time, lat_site) *= -1;											// flip the field!
		//this->upd_int_exp(lat_site, delta_up, delta_down);
		this->upd_B_mat(lat_site, delta_up, delta_down);											// update the B matrices
		this->upd_equal_green(lat_site, gamma_up / proba_up, gamma_down / proba_down);				// update Greens via Dyson
	}
	return sign;
}

/*
* @brief Drive the system to equilibrium with heat bath
* @param mcSteps number of Monte Carlo steps
* @param conf save configurations?
* @param quiet quiet?
*/
void hubbard::HubbardQR::heat_bath_eq(int mcSteps, bool conf, bool quiet, bool save_greens)
{
	if (!quiet && mcSteps != 1) {
#pragma omp critical
		stout << "\t\t----> STARTING RELAXING FOR : " + this->info << EL;
		this->neg_num = 0;																		// counter of negative signs
		this->pos_num = 0;																		// counter of positive signs
	}
#ifdef SAVE_CONF
	stout << "\t\t\t----> Saving configurations of Hubbard Stratonovich fields" << EL;
	mat confMine;
	if(conf) confMine.zeros(this->Ns,this->Ns);
#endif
	
	// progress bar
	this->pbar.reset(new pBar(20, mcSteps));
	
	// sweep all
	for (int step = 0; step < mcSteps; step++) {
		// Monte Carlo steps
#ifdef SAVE_CONF
		confMine = this->hsFields;											// save how it was before
#endif
		this->sweep_0_M();													// sweep forward
#ifdef SAVE_CONF								
		this->print_hs_fields("\t", confMine);								// print HS fields
#endif
		if (!quiet) {
			this->config_sign > 0 ? this->pos_num++ : this->neg_num++;		// increase sign

			if (step % pbar->percentageSteps == 0) 
				pbar->printWithTime(" -> RELAXATION PROGRESS for " + this->info);
		}
	}
}

/*
* @brief Average the system in equilibrium with heat bath
* @param corr_time time after which the new measurement is uncorrelated
* @param avNum number of averagings
* @param quiet quiet?
*/
void hubbard::HubbardQR::heat_bath_av(int corr_time, int avNum, bool quiet)
{
#pragma omp critical
	stout << "\t\t----> STARTING AVERAGING FOR : " + this->info << EL;

	this->neg_num = 0;																				// counter of negative signs
	this->pos_num = 0;																				// counter of positive signs
	this->avs->av_sign = 0;
	this->equalibrate = false;

	const uint bucket_num = 1;
	// Progress bar
	this->pbar.reset(new pBar(34, avNum));

	// check if this saved already
	for (int step = 0; step < avNum; step++) {
		// Monte Carlo steps
#ifdef USE_HIRSH
		// calculate time-displaced Green's functions USING HIRSH
		this->cal_green_mat_times_hirsh();
#endif
		// stout << "CALCULATING: " << VEQ(step) << EL;

		for (auto time_im = 0; time_im < this->M; time_im++) {
			// imaginary Trotter times
			this->current_time = time_im;
			//stout << "\t->TIME: " << VEQ(current_time) << EL;
#if !defined CAL_TIMES || defined CAL_TIMES && !defined USE_HIRSH
			this->upd_Green_step(this->current_time);
#endif
#ifdef CAL_TIMES
			// because we save the 0'th on the fly :3
			const auto elem = this->current_time * this->Ns;
#ifdef USE_HIRSH
			//? using Hirsh we know that we can set the whole matrix so we can set the eq times Green's function too
			setMatrixFromSubmatrix(green_up, g_up_time, elem, elem, Ns, Ns, false);
			setMatrixFromSubmatrix(green_down, g_down_time, elem, elem, Ns, Ns, false);
#else
			//? otherwise we do differently, we set it from the standardly calculated one
			setSubmatrixFromMatrix(g_up_time, green_up, elem, elem, Ns, Ns, false);
			setSubmatrixFromMatrix(g_down_time, green_down, elem, elem, Ns, Ns, false);
#endif
#endif
			// collect all averages
			for (int i = 0; i < this->Ns; i++) {
				//stout << "\t\t->SITE: " << VEQ(i) << EL;
				this->av_single_step(i, this->config_sign);
			}
			//! increase sign
			this->config_sign > 0 ? this->pos_num++ : this->neg_num++;

#ifdef CAL_TIMES
			//? Average the Green's over the buckets
			if (step % bucket_num == 0) {
				if (step != 0) this->save_unequal_greens(step / bucket_num, bucket_num);
				this->avs->resetGreens();
		}
#endif
	}
		//! kill correlations
		for (int ii = 0; ii < corr_time; ii++) {
			this->sweep_0_M();
		}

		//! printer
		if (!quiet && step % pbar->percentageSteps == 0)
			pbar->printWithTime(" -> AVERAGES PROGRESS for " + this->info);
	}
	//! Normalise after
	this->av_normalise(avNum, this->M);
}

// ---------------------------------------------------------------------------------------------------------------- PUBLIC CALCULATORS ----------------------------------------------------------------------------------------------------------------