#include "include/hubbard_dqmc_qr.h"
using namespace arma;

void hubbard::HubbardQR::initializeMemory()
{	
	/// hopping exponent
	this->hopping_exp.zeros(this->Ns, this->Ns);

	/// interaction for all times
	this->int_exp_down.ones(this->Ns, this->M);
	this->int_exp_up.ones(this->Ns, this->M);

	/// all times exponents multiplication
	this->b_mat_up = std::vector<mat>(this->M, arma::zeros(this->Ns, this->Ns));
	this->b_mat_down = std::vector<mat>(this->M, arma::zeros(this->Ns, this->Ns));
	this->b_mat_up_inv = std::vector<mat>(this->M, arma::zeros(this->Ns, this->Ns));
	this->b_mat_down_inv = std::vector<mat>(this->M, arma::zeros(this->Ns, this->Ns));

	this->b_up_condensed = std::vector<mat>(this->p, arma::zeros(this->Ns, this->Ns));
	this->b_down_condensed = std::vector<mat>(this->p, arma::zeros(this->Ns, this->Ns));

	/// all times hs fields for real spin up and down
	this->hsFields.ones(this->M, this->Ns);
	//this->hsFields_img = v_1d<v_1d<std::string>>(this->M, v_1d<std::string>(this->Ns," "));
	/// Green's function matrix
	this->green_up.zeros(this->Ns, this->Ns);
	this->green_down.zeros(this->Ns, this->Ns);
	this->tempGreen_up.zeros(this->Ns, this->Ns);
	this->tempGreen_down.zeros(this->Ns, this->Ns);

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
	this->g_down_eq = v_1d<mat>(this->M, arma::eye(this->Ns, this->Ns));
	this->g_up_eq = v_1d<mat>(this->M, arma::eye(this->Ns, this->Ns));
}

//? -------------------------------------------------------- CONSTRUCTORS
hubbard::HubbardQR::HubbardQR(const v_1d<double>& t, double dtau, int M_0, double U, double mu, double beta, std::shared_ptr<Lattice> lattice, int threads, bool ct)
{
	this->lattice = lattice;
	this->inner_threads = threads;
	this->dir = std::shared_ptr<hubbard::directories>(new directories());
	this->cal_times = ct;
	this->Lx = this->lattice->get_Lx();
	this->Ly = this->lattice->get_Ly();
	this->Lz = this->lattice->get_Lz();
	this->t = t;
	this->config_sign = 1;

	/// Params
	this->U = U;
	this->mu = mu;
	this->beta = beta;
	this->T = 1.0 / this->beta;
	this->Ns = this->lattice->get_Ns();
	this->ran = randomGen();																		// random number generator initialization

	/// Trotter
	this->dtau = dtau;
	this->M = static_cast<int>(this->beta / this->dtau);											// number of Trotter times
	this->M_0 = M_0;																				
	this->p = (this->M / this->M_0);																// number of QR decompositions (sectors)

	this->avs = std::make_shared<averages_par>(Lx, Ly, Lz, M, this->cal_times);
	/// Calculate alghorithm parameters
	this->lambda = std::acosh(exp((abs(this->U) * this->dtau) * 0.5));

	/// Calculate changing exponents before, not to calculate exp all the time
	this->gammaExp = { std::expm1(-2.0 * this->lambda), std::expm1(2.0 * this->lambda) };			// 0 -> sigma * hsfield = 1, 1 -> sigma * hsfield = -1

	/// Helping params
	this->from_scratch = this->M_0;
	this->pos_num = 0;
	this->neg_num = 0;

	/// Say hi to the world
#pragma omp critical
	stout << "CREATING THE HUBBARD MODEL WITH QR DECOMPOSITION WITH PARAMETERS:" << el;
#pragma omp critical
	this->say_hi();

	/// Initialize memory
	this->initializeMemory();
	/// Set HS fields												
	this->set_hs();																					

	/// Calculate something
	this->cal_hopping_exp();
	this->cal_int_exp();
	this->cal_B_mat();
	/// Precalculate the multipliers of B matrices for convinience
	for (int i = 0; i < this->p; i++) {
		this->cal_B_mat_cond(i);
	}
}

//! -------------------------------------------------------- B MATS --------------------------------------------------------

/**
* Precalculate the multiplications of B matrices
* @param which_sector
*/
void hubbard::HubbardQR::cal_B_mat_cond(int which_sector)
{
	int tim = which_sector * this->M_0;
	this->b_down_condensed[which_sector] = this->b_mat_down[tim];
	this->b_up_condensed[which_sector] = this->b_mat_up[tim];
	//#pragma omp parallel for num_threads(this->inner_threads)
	for (int i = 1; i < this->M_0; i++) {
		tim++;
		this->b_down_condensed[which_sector] = this->b_mat_down[tim] * this->b_down_condensed[which_sector];
		this->b_up_condensed[which_sector] = this->b_mat_up[tim] * this->b_up_condensed[which_sector];
	}
}

//TODO ----------------------->
/**
*
* @param l_start
* @param l_end
* @param tmp_up
* @param tmp_down
*/
void hubbard::HubbardQR::b_mat_multiplier_left(int l_start, int l_end, mat& tmp_up, mat& tmp_down)
{
	int timer = l_start;
	const int how_many = abs(l_end - l_start);
	tmp_up = this->b_mat_up[timer];
	tmp_down = this->b_mat_down[timer];

	if (how_many > this->M_0) {
		for (int j = 1; j < this->M_0; j++) {
			timer++;
			if (timer == this->M) timer = 0;
			tmp_down = this->b_mat_down[timer] * tmp_down;
			tmp_up = this->b_mat_up[timer] * tmp_up;
		}
		setUDTDecomp(tmp_up, Q_up, R_up, P_up, T_up, D_up);
		setUDTDecomp(tmp_down, Q_down, R_down, P_down, T_down, D_down);
		for (int j = 0; j <= how_many - this->M_0; j++) {
			timer++;
			if (timer == this->M) timer = 0;
			multiplyMatricesQrFromRight(this->b_mat_up[timer], Q_up, R_up, P_up, T_up, D_up);
			multiplyMatricesQrFromRight(this->b_mat_down[timer], Q_down, R_down, P_down, T_down, D_down);
		}
		tmp_up = Q_up * arma::diagmat(R_up) * T_up;
		tmp_down = Q_down * arma::diagmat(R_down) * T_down;
	}
	else {
		for (int j = 1; j <= how_many; j++) {
			timer++;
			if (timer == this->M) timer = 0;
			tmp_down = this->b_mat_down[timer] * tmp_down;
			tmp_up = this->b_mat_up[timer] * tmp_up;
		}
	}
}

//TODO ----------------------->
/**
 * @brief 
 * 
 * @param l_start 
 * @param l_end 
 * @param tmp_up 
 * @param tmp_down 
 */
void hubbard::HubbardQR::b_mat_multiplier_right(int l_start, int l_end, mat& tmp_up, mat& tmp_down)
{
	int timer = l_start;
	const int how_many = abs(l_end - l_start);
	tmp_up = this->b_mat_up[timer];
	tmp_down = this->b_mat_down[timer];
	for (int j = 1; j <= how_many; j++) {
		timer++;
		if (timer == this->M) timer = 0;
		tmp_down = tmp_down * this->b_mat_down[timer];
		tmp_up = tmp_up * this->b_mat_up[timer];
	}
}

//TODO ----------------------->
/**
 * @brief 
 * 
 * @param l_start 
 * @param l_end 
 * @param tmp_up 
 * @param tmp_down 
 */
void hubbard::HubbardQR::b_mat_multiplier_left_inv(int l_start, int l_end, mat& tmp_up, mat& tmp_down)
{
	auto timer = l_start;
	const int how_many = abs(l_end - l_start);
	tmp_up = this->b_mat_up_inv[timer];
	tmp_down = this->b_mat_down_inv[timer];
	for (int j = 1; j <= how_many; j++) {
		timer++;
		if (timer == this->M) timer = 0;
		tmp_up = this->b_mat_up_inv[timer] * tmp_up;
		tmp_down = this->b_mat_down_inv[timer] * tmp_down;
	}
}

//TODO ----------------------->
/**
 * @brief 
 * 
 * @param l_start 
 * @param l_end 
 * @param tmp_up 
 * @param tmp_down 
 */
void hubbard::HubbardQR::b_mat_multiplier_right_inv(int l_start, int l_end, mat& tmp_up, mat& tmp_down)
{
	int timer = l_start;
	const int how_many = abs(l_end - l_start);
	tmp_up = this->b_mat_up_inv[timer];
	tmp_down = this->b_mat_down_inv[timer];
	for (int j = 1; j <= how_many; j++) {
		timer++;
		if (timer == this->M) timer = 0;
		tmp_down = tmp_down * this->b_mat_down_inv[timer];
		tmp_up = tmp_up * this->b_mat_up_inv[timer];
	}
}

//! -------------------------------------------------------- GREENS --------------------------------------------------------

/**
* Compare decomposition created Green's functions with directly calculated
* @param tim at which time shall I compare them?
* @param toll tollerance for them being equal
* @param print_greens shall I print both explicitly?
*/
void hubbard::HubbardQR::compare_green_direct(int tim, double toll, bool print_greens)
{
	mat tmp_up(this->Ns, this->Ns, arma::fill::eye);
	mat tmp_down(this->Ns, this->Ns, arma::fill::eye);
	for (int i = 0; i < this->M; i++) {
		tmp_up = this->b_mat_up[tim] * tmp_up;
		tmp_down = this->b_mat_down[tim] * tmp_down;
		tim = (tim + 1) % this->M;
	}
	tmp_up = (arma::eye(this->Ns, this->Ns) + tmp_up).i();
	tmp_down = (arma::eye(this->Ns, this->Ns) + tmp_down).i();
	bool up = approx_equal(this->green_up, tmp_up, "absdiff", toll);
	bool down = approx_equal(this->green_down, tmp_down, "absdiff", toll);
	stout << " -------------------------------- FOR TIME : " << tim << el;
	stout << "up Green:\n" << (up ? "THE SAME!" : "BAAAAAAAAAAAAAAAAAAAAAAD!") << el;
	if (print_greens)
		stout << this->green_up - tmp_up << el;
	stout << "down Green:\n" << (down ? "THE SAME!" : "BAAAAAAAAAAAAAAAAAAAAAAD!") << el;
	if (print_greens)
		stout << this->green_down - tmp_down << "\n\n\n";
}

//? -------------------------------------------------------- EQUAL

/**
* Calculate Green with QR decomposition using LOH. Here we calculate the Green matrix at a given time, so we need to take care of the times away from precalculated sectors
* @cite doi:10.1016/j.laa.2010.06.023
* @param which_time The time at which the Green's function is calculated
*/
void hubbard::HubbardQR::cal_green_mat(int which_time) {
	auto tim = which_time;
	auto sec = static_cast<int>(which_time / this->M_0);										// which sector is used for M_0 multiplication
	auto sector_end = (sec + 1) * this->M_0 - 1;
	// multiply those B matrices that are not yet multiplied
	b_mat_multiplier_left(tim, sector_end, tempGreen_up, tempGreen_down);						// using tempGreens to store the starting multiplication

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
	b_mat_multiplier_left(tim, sector_end, tempGreen_up, tempGreen_down);
	multiplyMatricesQrFromRight(tempGreen_up, Q_up, R_up, P_up, T_up, D_up);
	multiplyMatricesQrFromRight(tempGreen_down, Q_down, R_down, P_down, T_down, D_down);

	//stout << el;
	//this->green_up = T_up.i() * (Q_up.t() * T_up.i() + arma::diagmat(R_up)).i()*Q_up.t();
	//this->green_down = T_down.i() * (Q_down.t() * T_down.i() + arma::diagmat(R_down)).i()*Q_down.t();

	// Correction terms
	makeTwoScalesFromUDT(R_up, D_up);
	makeTwoScalesFromUDT(R_down, D_down);
	// calculate equal time Green
	this->green_up = arma::solve(arma::diagmat(D_up) * Q_up.t() + arma::diagmat(R_up) * T_up, arma::diagmat(D_up) * Q_up.t());
	this->green_down = arma::solve(arma::diagmat(D_down) * Q_down.t() + arma::diagmat(R_down) * T_down, arma::diagmat(D_down) * Q_down.t());
}

/**
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
	this->green_up = arma::solve(arma::diagmat(D_up) * Q_up.t() + arma::diagmat(R_up) * T_up, arma::diagmat(D_up) * Q_up.t());
	this->green_down = arma::solve(arma::diagmat(D_down) * Q_down.t() + arma::diagmat(R_down) * T_down, arma::diagmat(D_down) * Q_down.t());
}

//? -------------------------------------------------------- UNEQUAL

/**
* Calculating unequal time Green's functions given by Bl_1*...*B_{l2+1}*G_{l2+1} \\rightarrow [B_{l2+1}^{-1}...B_l1^{-1} + B_l2...B_1B_{M-1}...B_{l1+1}]^{-1}. 
* Make inverse of function of type (Ql*diag(Rl)*Tl + Qr*diag(Rr)*Tr)^(-1) using:
* @cite SciPost Phys. Core 2, 011 (2020) 
* @param t1 left time t1>t2
* @param t2 right time t2<t1
* @param inv_series_up precalculated inverse matrices multiplication for spin up
* @param inv_series_down precalculated inverse matrices multiplication for spin down
*/
void hubbard::HubbardQR::unequalG_greaterFirst(int t1, int t2, const mat& inv_series_up, const mat& inv_series_down)
{
	assert("t2 should be higher than t1", t2 >= t1);
	const auto row = t1 * this->Ns;
	const auto col = t2 * this->Ns;

	//! ------------------------------------ up ------------------------------------ 
	//? USE DOWN MATRICES AS HELPERS FOR RIGHT SUM TO SAVE PRECIOUS MEMORY!
	//! B(t2 + 1)^(-1)...B(t1)^(-1)
	setUDTDecomp(inv_series_up, Q_up, R_up, P_up, T_up, D_up);												// decompose the premultiplied inversions to up temporaries
	// setUDTDecomp(arma::inv(T_up) * arma::diagmat(D_up) * Q_up.t(), Q_up, R_up, P_up, T_up, D_up);		// if the inv_series_up was composed from non_inverses

	//! B(M-1)...B(t1 + 1)
	setUDTDecomp(this->g_up_eq[0], Q_down, R_down, P_down, T_down, D_down);									// decompose and use down matrices as temporaries + equal time Green at [0]

	//! SET MATRIX ELEMENT
	setSubmatrixFromMatrix(this->g_up_time,
						inv_left_plus_right_qr(
							Q_up, R_up, P_up, T_up, D_up,
							Q_down, R_down, P_down, T_down, D_down, D_tmp
						),
						row, col, this->Ns, this->Ns, false);

	//! ------------------------------------ down ------------------------------------
	//? USE UP MATRICES AS HELPERS FOR RIGHT SUM TO SAVE PRECIOUS MEMORY!
	//! B(t2 + 1)^(-1)...B(t1)^(-1)
	setUDTDecomp(inv_series_down, Q_down, R_down, P_down, T_down, D_down);									// decompose the premultiplied inversions to up temporaries
	//setUDTDecomp(arma::inv(T_down) * arma::diagmat(D_down) * Q_down.t(), Q_down, R_down, P_down, T_down, D_down);
	
	//! B(M-1)...B(t1 + 1)
	setUDTDecomp(this->g_down_eq[0], Q_up, R_up, P_up, T_up, D_up);

	//! SET MATRIX ELEMENT
	setSubmatrixFromMatrix(this->g_down_time,
						inv_left_plus_right_qr(
							Q_down, R_down, P_down, T_down, D_down,
							Q_up, R_up, P_up, T_up, D_up, D_tmp
						),
						row, col, this->Ns, this->Ns, false);
}

/**
/// @param t1
/// @param t2
/// @param D_tmp
///
*/
void hubbard::HubbardQR::unequalG_greaterLast(int t1, int t2, const mat& inv_series_up, const mat& inv_series_down)
{
	if (t2 <= t1) throw "can't do that m8\n";
	// make inverse of function of type (Ql*diag(Rl)*Tl + Qr*diag(Rr)*Tr)^(-1) using SciPost Phys. Core 2, 011 (2020)
	const auto row = t1 * this->Ns;
	const auto col = t2 * this->Ns;

	// ------------------------------------ up ------------------------------------ USE DOWN MATRICES AS HELPERS FOR RIGHT SUM!
	// B(l2)...B(l1+1)
	setUDTDecomp(inv_series_up, Q_up, R_up, P_up, T_up, D_up);
	this->tempGreen_up = arma::inv(T_up) * arma::diagmat(D_up) * Q_up.t();
	// B(M-1)...B(t1 + 1)
	//setUDTDecomp(this->g_up_eq[t1], Q_down, R_down, P_down, T_down, D_down);
	setUDTDecomp(this->g_up_eq[0], Q_down, R_down, P_down, T_down, D_down);
	// B(t2)...B(0)
	//multiplyMatricesQrFromRight(this->g_up_tim[t2], Q_down, R_down, P_down, T_down, D_down);
	// SET MATRIX ELEMENT
	setUDTDecomp(inv_left_plus_right_qr(Q_up, R_up, P_up, T_up, D_up, \
		Q_down, R_down, P_down, T_down, D_down, D_tmp), Q_up, R_up, P_up, T_up, D_up);
	setUDTDecomp(this->tempGreen_up, Q_down, R_down, P_down, T_down, D_down);
	setUDTDecomp(arma::diagmat(R_up) * T_up * T_down.i() - Q_up.t() * Q_down * arma::diagmat(R_down), Q_down, R_up, P_up, T_up, D_up);

	setSubmatrixFromMatrix(this->g_up_time, Q_up * Q_down * arma::diagmat(R_up) * T_up * T_down, row, col, this->Ns, this->Ns, false);

	// ------------------ down ------------------
	// B(l2)...B(l1+1)
	setUDTDecomp(inv_series_down, Q_down, R_down, P_down, T_down, D_down);
	this->tempGreen_down = arma::inv(T_down) * arma::diagmat(D_down) * Q_down.t();
	// B(M-1)...B(t1 + 1)
	//setUDTDecomp(this->g_down_eq[t1], Q_up, R_up, P_up, T_up, D_up);
	setUDTDecomp(this->g_down_eq[0], Q_up, R_up, P_up, T_up, D_up);
	// B(t2)...B(0)
	//multiplyMatricesQrFromRight(this->g_down_tim[t2], Q_up, R_up, P_up, T_up, D_up);
	// SET MATRIX ELEMENT
	setUDTDecomp(inv_left_plus_right_qr(Q_down, R_down, P_down, T_down, D_down, \
		Q_up, R_up, P_up, T_up, D_up, D_tmp), Q_down, R_down, P_down, T_down, D_down);
	setUDTDecomp(this->tempGreen_up, Q_up, R_up, P_up, T_up, D_up);
	setUDTDecomp(arma::diagmat(R_down) * T_down * T_up.i() - Q_down.t() * Q_up * arma::diagmat(R_up), Q_up, R_down, P_down, T_down, D_down);

	setSubmatrixFromMatrix(this->g_down_time, Q_down * Q_up * arma::diagmat(R_down) * T_down * T_up, row, col, this->Ns, this->Ns, false);
}

/**
///

/// @param tau1
/// @param tau2
*/
void hubbard::HubbardQR::cal_green_mat_times()
{
	const uint tau1 = this->M - 1;
}

/**
///

/// @param tau1
/// @param tau2
*/
void hubbard::HubbardQR::cal_green_mat_times_cycle()
{
	// inverses
	//this->g_down_eq[this->M-1].eye();
	//this->g_up_eq[this->M-1].eye();
	this->g_down_eq[this->M - 2] = this->b_mat_down[this->M - 1];
	this->g_up_eq[this->M - 2] = this->b_mat_up[this->M - 1];
	for (int i = this->M - 2; i > 0; i--) {
		// give the new matrices a save
		setUDTDecomp(this->b_mat_down[i], Q_down, R_down, P_down, T_down, D_down);
		setUDTDecomp(this->b_mat_up[i], Q_up, R_up, P_up, T_up, D_up);

		multiplyMatricesQrFromRight(this->g_up_eq[i], Q_up, R_up, P_up, T_up, D_up);
		multiplyMatricesQrFromRight(this->g_down_eq[i], Q_down, R_down, P_down, T_down, D_down);
		this->g_up_eq[i - 1] = Q_up * arma::diagmat(R_up) * T_up;
		this->g_down_eq[i - 1] = Q_down * arma::diagmat(R_down) * T_down;
	}

	// calculate all Greens
	for (int tau1 = 1; tau1 < this->M; tau1++) {
		//this->g_up_eq[0] = this->g_up_eq[tau1];
		//this->g_down_eq[0] = this->g_down_eq[tau1];
		for (int tau2 = 0; tau2 < 1; tau2++) {
			setUDTDecomp(this->g_up_eq[tau1], Q_up, R_up, P_up, T_up, D_up);
			setUDTDecomp(this->g_down_eq[tau1], Q_down, R_down, P_down, T_down, D_down);
			multiplyMatricesQrFromRight(this->b_mat_up[tau2], Q_up, R_up, P_up, T_up, D_up);
			multiplyMatricesQrFromRight(this->b_mat_down[tau2], Q_down, R_down, P_down, T_down, D_down);
			this->g_up_eq[0] = Q_up * arma::diagmat(R_up) * T_up;
			this->g_down_eq[0] = Q_down * arma::diagmat(R_down) * T_down;

			//if (tau2 > tau1) {
			//	b_mat_multiplier_left(tau1 + 1, tau2, this->tempGreen_up, this->tempGreen_down);
			//	unequalG_greaterLast(tau1, tau2, this->tempGreen_up, this->tempGreen_down);
			//}
			//else{
			b_mat_multiplier_left(tau2 + 1, tau1, this->tempGreen_up, this->tempGreen_down);
			unequalG_greaterFirst(tau1, tau2, this->tempGreen_up, this->tempGreen_down);
			//}
		}
	}
}

/**
///
*/
void hubbard::HubbardQR::cal_green_mat_times_hirsh()
{
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
	this->g_up_time = arma::solve(this->g_up_time, arma::arma::diagmat(this->g_up_time));
	this->g_down_time = arma::solve(this->g_down_time, arma::arma::diagmat(this->g_down_time));
	//this->g_up_time = this->g_up_time.i();
	//this->g_down_time = this->g_down_time.i();
}

/**
///
*/
void hubbard::HubbardQR::cal_green_mat_times_hirsh_cycle()
{
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

// -------------------------------------------------------- HELPERS

/**
///

/// @param fptr
/// <returns></returns>
*/
int hubbard::HubbardQR::sweep_lat_sites(std::function<int(int)> fptr)
{
	int sign = 1;
	for (int j = 0; j < this->Ns; j++) {
		//const auto lat_site = ran.randomInt_uni(0, this->Ns - 1);
		const auto lat_site = j;
		sign = (fptr)(lat_site);
	}
	// return sign from the last possible flip
	return sign;
}

// -------------------------------------------------------- GREEN UPDATERS --------------------------------------------------------

/**
/// After changing one spin we need to update the Green matrices via the Dyson equation

/// @param lat_site the site on which HS field has been changed
/// @param prob_up the changing probability for up channel
/// @param prob_down the changing probability for down channel
/// @param gamma_up changing parameter gamma for up channel
/// @param gamma_down changing probability for down channel
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

/**
/// Update the Green's matrices after going to next Trotter time, remember, the time is taken to be the previous one
/// @param which_time updating to which_time + 1
*/
void hubbard::HubbardQR::upd_next_green(int which_time_green) {
	this->green_up = (this->b_mat_up[which_time_green] * this->green_up) * this->b_mat_up_inv[which_time_green];						// LEFT INCREASE
	this->green_down = (this->b_mat_down[which_time_green] * this->green_down) * this->b_mat_down_inv[which_time_green];				// LEFT INCREASE;
}

/**

/// @param which_time
///
*/
void hubbard::HubbardQR::upd_prev_green(int which_time_green) {
	this->green_up = (this->b_mat_up_inv[which_time_green - 1] * this->green_up) * this->b_mat_up[which_time_green - 1];						// LEFT INCREASE
	this->green_down = (this->b_mat_down_inv[which_time_green - 1] * this->green_down) * this->b_mat_down[which_time_green - 1];				// LEFT INCREASE;
}

// ----------------------------------------------------------------------------------------------------------------

/**
///
*/
/// @param im_time_step
void hubbard::HubbardQR::upd_Green_step(int im_time_step, bool forward) {
	if (im_time_step % this->from_scratch == 0) {
		const auto sector_to_upd = myModuloEuclidean(static_cast<int>(im_time_step / double(this->M_0)) - 1, this->p);
		this->cal_B_mat_cond(sector_to_upd);
		this->cal_green_mat_cycle(myModuloEuclidean(static_cast<int>(im_time_step / double(this->M_0)), this->p));
	}
	else
		this->upd_next_green(im_time_step - 1);
}
// -------------------------------------------------------- CALCULATORS

/**
/// A single step for calculating averages inside a loop

/// @param current_elem_i  Current Green matrix element in averages
*/
void hubbard::HubbardQR::av_single_step(int current_elem_i, int sign, bool times)
{
	//this->avs->av_sign += sign;
	// m_z
	const double m_z2 = this->cal_mz2(sign, current_elem_i, this->green_up, this->green_down);
	this->avs->av_M2z += m_z2;
	this->avs->sd_M2z += m_z2 * m_z2;
	// m_x
	const double m_x2 = this->cal_mx2(sign, current_elem_i, this->green_up, this->green_down);
	this->avs->av_M2x += m_x2;
	this->avs->sd_M2x += m_x2 * m_x2;
	// occupation
	const double occ = this->cal_occupation(sign, current_elem_i, this->green_up, this->green_down);
	this->avs->av_occupation += occ;
	this->avs->sd_occupation += occ * occ;
	// kinetic energy
	const double Ek = this->cal_kinetic_en(sign, current_elem_i, this->green_up, this->green_down);
	this->avs->av_Ek += Ek;
	this->avs->av_Ek2 += Ek * Ek;
	// Correlations
	for (int current_elem_j = 0; current_elem_j < this->Ns; current_elem_j++) {
		auto [x, y, z] = this->lattice->getSiteDifference(current_elem_i, current_elem_j);
		const int xx = x - (Lx - 1); const int yy = y - (Ly - 1); const int zz = z - (Lz - 1);
		//const int xx = abs(x) % (Lx/2 + 1); const int yy = abs(y) % (Ly/2 + 1); const int zz = abs(z) % (Lz/2 + 1);
		// normal equal - time correlations
		this->avs->av_M2z_corr[x][y][z] += this->cal_mz2_corr(sign, current_elem_i, current_elem_j, this->green_up, this->green_down);
		this->avs->av_occupation_corr[x][y][z] += this->cal_occupation_corr(sign, current_elem_i, current_elem_j, this->green_up, this->green_down);
		this->avs->av_ch2_corr[x][y][z] += this->cal_ch_correlation(sign, current_elem_i, current_elem_j, this->green_up, this->green_down) / (this->Ns * 2.0);

		// handle zero time difference here in greens
		if ((xx <= this->Lx / 2 && xx >= 0) && (yy <= this->Ly / 2 && yy >= 0) && (zz <= this->Lz / 2 && zz >= 0)) {
			this->avs->g_up_diffs[0](xx, yy) += this->green_up(current_elem_i, current_elem_j);
			this->avs->g_down_diffs[0](xx, yy) += this->green_down(current_elem_i, current_elem_j);
			this->avs->sd_g_up_diffs[0](xx, yy) += this->green_up(current_elem_i, current_elem_j) * this->green_up(current_elem_i, current_elem_j);
			this->avs->sd_g_down_diffs[0](xx, yy) += this->green_down(current_elem_i, current_elem_j) * this->green_down(current_elem_i, current_elem_j);
		}


		if (times) {
			if ((xx <= this->Lx / 2 && xx >= 0) && (yy <= this->Ly / 2 && yy >= 0) && (zz <= this->Lz / 2 && zz >= 0)) {
				for (int time2 = 0; time2 < this->M; time2++) {
					auto tim = (this->current_time - time2);
					if(tim == 0) continue;
					int xi = 1;
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
			}
		}
	}
}
// ---------------------------------------------------------------------------------------------------------------- HEAT BATH ----------------------------------------------------------------------------------------------------------------

/**
///
*/
void hubbard::HubbardQR::sweep_0_M(std::function<int(int)> ptfptr)
{
	this->config_sign = 1;
	for (int time_im = 0; time_im < this->M; time_im++) {
		this->current_time = time_im;
		this->upd_Green_step(this->current_time, true);
		this->config_sign = (this->sweep_lat_sites(ptfptr) > 0) ? +this->config_sign : -this->config_sign;
	}
}

/**
///
*/
/// @param function
void hubbard::HubbardQR::sweep_M_0(std::function<int(int)> ptfptr)
{
	int sign = this->config_sign;
	for (int time_im = this->M - 1; time_im >= 0; time_im--) {
		// imaginary Trotter times
		this->current_time = time_im;//tim[time_im];
		sign = sweep_lat_sites(ptfptr) > 0 ? +sign : -sign;
		this->upd_Green_step(time_im, false);
	}
	this->config_sign = sign;// (sign == 1) ? +this->config_sign : -this->config_sign;
}

/**
/// Single step for the candidate to flip the HS field
*/
/// @param lat_site the candidate lattice site
/// <returns>sign of probability</returns>
int hubbard::HubbardQR::heat_bath_single_step(int lat_site)
{
	auto [gamma_up, gamma_down] = this->cal_gamma(lat_site);										// first up then down
	auto [proba_up, proba_down] = this->cal_proba(lat_site, gamma_up, gamma_down);					// take the probabilities

	this->probability = (proba_up * proba_down);													// Metropolis probability
	if (this->U < 0) this->probability *= (this->gammaExp[1] + 1);									// add phase factor for
	this->probability = this->probability / (1.0 + this->probability);								// heat-bath probability
	//proba = std::min(proba, 1.0);																		// metropolis
	const int sign = (this->probability >= 0) ? 1 : -1;												// check sign
	if (this->ran.randomReal_uni() <= sign * this->probability) {
		const auto delta_up = gamma_up + 1;
		const auto delta_down = gamma_down + 1;
		this->hsFields(this->current_time, lat_site) *= -1;
		//this->upd_int_exp(lat_site, delta_up, delta_down);
		this->upd_B_mat(lat_site, delta_up, delta_down);											// update the B matrices
		this->upd_equal_green(lat_site, gamma_up / proba_up, gamma_down / proba_down);				// update Greens via Dyson
	}
	return sign;
}

/**
/// Drive the system to equilibrium with heat bath
*/
/// @param mcSteps Number of Monte Carlo steps
/// @param conf If or if not to save configurations
/// @param quiet If should be quiet
void hubbard::HubbardQR::heat_bath_eq(int mcSteps, bool conf, bool quiet, bool save_greens)
{
	if (!quiet && mcSteps != 1) {
#pragma omp critical
		stout << "\t\t----> STARTING RELAXING FOR : " + this->info << el;
		this->neg_num = 0;																				// counter of negative signs
		this->pos_num = 0;																				// counter of positive signs
	}
	if (conf) stout << "\t\t\t----> Saving configurations of Hubbard Stratonovich fields\n";
	// Progress bar
	auto progress = pBar();
	const double percentage = 34;
	const int percentage_steps = static_cast<int>(percentage * mcSteps / 100.0);

	// function
	std::function<int(int)> fptr = std::bind(&HubbardQR::heat_bath_single_step, this, std::placeholders::_1);	// pointer to non-saving configs;
	mat confMine(this->Ns, this->Ns);
	// sweep all
	for (int step = 0; step < mcSteps; step++) {
		// Monte Carlo steps
		
		if (conf) confMine = this->hsFields;
		// save how it was before
		this->sweep_0_M(fptr);
		if (!quiet) {
			this->config_sign > 0 ? this->pos_num++ : this->neg_num++;								// increase sign
			if(conf) this->print_hs_fields("\t", confMine);
			if (step % percentage_steps == 0) progress.printWithTime(" -> RELAXATION PROGRESS for " + this->info, percentage);
		}
	}
}

/**
///
*/
/// @param corr_time
/// @param avNum
/// @param quiet
/// @param times
void hubbard::HubbardQR::heat_bath_av(int corr_time, int avNum, bool quiet, bool times)
{
#pragma omp critical
	stout << "\t\t----> STARTING AVERAGING FOR : " + this->info << el;
	this->neg_num = 0;																				// counter of negative signs
	this->pos_num = 0;																				// counter of positive signs
	this->avs->av_sign = 0;

	std::function<int(int)> fptr = std::bind(&HubbardQR::heat_bath_single_step, this, std::placeholders::_1);

	const uint bucket_num = 1;
	// Progress bar
	auto progress = pBar();
	const double percentage = 34;
	const auto percentage_steps = static_cast<int>(percentage * avNum / 100.0);

	v_1d<mat> avs_up, avs_down;
	if (times) {
		this->g_up_time.eye(Ns * M, Ns * M);
		this->g_down_time.eye(Ns * M, Ns * M);
	}
	const bool useHirsh = true;
	// check if this saved already
	for (int step = 0; step < avNum; step++) {
		// Monte Carlo steps
		if (times) useHirsh ? this->cal_green_mat_times_hirsh() : this->cal_green_mat_times_cycle();
		for (int time_im = 0; time_im < this->M; time_im++) {
			// imaginary Trotter times
			this->current_time = time_im;
			if (!times || (!useHirsh)) this->upd_Green_step(this->current_time);
			if (times) {
				// because we save the 0'th on the fly :3
				if (useHirsh) {
					setMatrixFromSubmatrix(green_up, g_up_time, time_im * Ns, time_im * Ns, Ns, Ns, false);
					setMatrixFromSubmatrix(green_down, g_down_time, time_im *Ns, time_im * Ns, Ns, Ns, false);
				}
				else {
					setSubmatrixFromMatrix(g_up_time, green_up, time_im * Ns, time_im * Ns, Ns, Ns, false);
					setSubmatrixFromMatrix(g_down_time, green_down, time_im * Ns, time_im * Ns, Ns, Ns, false);
				}
			}
			for (int i = 0; i < this->Ns; i++) this->av_single_step(i, this->config_sign, times);			// collect all averages
		}
		// increase sign
		this->config_sign > 0 ? this->pos_num++ : this->neg_num++;											

		if (times && step % bucket_num == 0) {
			this->save_unequal_greens(step / bucket_num, bucket_num);
			this->avs->resetGreens();
		}
		// kill correlations
		for (int ii = 0; ii < corr_time; ii++) this->sweep_0_M(fptr);										
		// printer
		if (!quiet && step % percentage_steps == 0) progress.printWithTime(" -> AVERAGES PROGRESS for " + this->info, percentage);
	}
	// After

	this->av_normalise(avNum, this->M, times);
}

// ---------------------------------------------------------------------------------------------------------------- PUBLIC CALCULATORS ----------------------------------------------------------------------------------------------------------------