#include "include/hubbard_dqmc_qr.h"


// ---------------------------------------------------------------------------------------------------------------- HUBBARD MODEL WITH QR DECOMPOSITION ----------------------------------------------------------------------------------------------------------------

// -------------------------------------------------------- CONSTRUCTORS

hubbard::HubbardQR::HubbardQR(const std::vector<double>& t, double dtau, int M_0, double U, double mu, double beta, std::shared_ptr<Lattice> lattice, int threads)
{
	this->lattice = lattice;
	this->inner_threads = threads;
	int Lx = this->lattice->get_Lx();
	int Ly = this->lattice->get_Ly();
	int Lz = this->lattice->get_Lz();
	this->avs = std::make_shared<averages_par>(Lx, Ly, Lz);
	this->t = t;
	this->config_sign = 1;

	// Params 
	this->U = U;
	this->mu = mu;
	this->beta = beta;
	this->T = 1.0 / this->beta;
	this->Ns = this->lattice->get_Ns();
	this->ran = randomGen();																// random number generator initialization

	// Trotter 
	this->dtau = dtau;
	this->M = static_cast<int>(this->beta / this->dtau);									// number of Trotter times
	this->M_0 = M_0;
	this->p = (this->M / this->M_0);														// number of QR decompositions

	// Calculate alghorithm parameters 
	//this->lambda = 2 * std::atan(tanh((abs(this->U) * this->dtau) / 4.0));
	this->lambda = std::acosh(exp((abs(this->U) * this->dtau) * 0.5));

	// Calculate changing exponents before, not to calculate exp all the time 
	this->gammaExp = { std::expm1(-2.0 * this->lambda), std::expm1(2.0 * this->lambda) };			// 0 -> sigma * hsfield = 1, 1 -> sigma * hsfield = -1

	// Helping params 
	this->from_scratch = this->M_0;
	this->pos_num = 0;
	this->neg_num = 0;
	this->neg_dir = std::string();
	this->pos_dir = std::string();
	this->neg_log = std::string();
	this->pos_dir = std::string();

	// Say hi to the world 
#pragma omp critical
	std::cout << "CREATING THE HUBBARD MODEL WITH QR DECOMPOSITION WITH PARAMETERS:" << std::endl \
		// decomposition
		<< "->M = " << this->M << std::endl \
		<< "->M_0 = " << this->M_0 << std::endl \
		<< "->p = " << this->p << std::endl \
		// physical
		<< "->beta = " << this->beta << std::endl \
		<< "->U = " << this->U << std::endl \
		<< "->dtau = " << this->dtau << std::endl \
		<< "->mu = " << this->mu << std::endl \
		<< "->t = " << this->t << std::endl \
		// lattice
		<< "->dimension = " << this->getDim() << std::endl \
		<< "->Lx = " << this->lattice->get_Lx() << std::endl \
		<< "->Ly = " << this->lattice->get_Ly() << std::endl \
		<< "->Lz = " << this->lattice->get_Lz() << std::endl \
		<< "->lambda = " << this->lambda << std::endl << std::endl;
	/* Setting info about the model for files */
	this->info = "M=" + std::to_string(this->M) + ",M_0=" + std::to_string(this->M_0) + \
		",dtau=" + to_string_prec(this->dtau) + ",Lx=" + std::to_string(this->lattice->get_Lx()) + \
		",Ly=" + std::to_string(this->lattice->get_Ly()) + ",Lz=" + std::to_string(this->lattice->get_Lz()) + \
		",beta=" + to_string_prec(this->beta) + ",U=" + to_string_prec(this->U) + \
		",mu=" + to_string_prec(this->mu);

	// Initialize memory 
	this->hopping_exp.zeros(this->Ns, this->Ns);

	// interaction for all times
	this->int_exp_down.ones(this->Ns, this->M);
	this->int_exp_up.ones(this->Ns, this->M);

	// all times exponents multiplication
	this->b_mat_up = std::vector<arma::mat>(this->M, arma::zeros(this->Ns, this->Ns));
	this->b_mat_down = std::vector<arma::mat>(this->M, arma::zeros(this->Ns, this->Ns));
	this->b_mat_up_inv = std::vector<arma::mat>(this->M, arma::zeros(this->Ns, this->Ns));
	this->b_mat_down_inv = std::vector<arma::mat>(this->M, arma::zeros(this->Ns, this->Ns));

	this->b_up_condensed = std::vector<arma::mat>(this->p, arma::zeros(this->Ns, this->Ns));
	this->b_down_condensed = std::vector<arma::mat>(this->p, arma::zeros(this->Ns, this->Ns));

	// all times hs fields for real spin up and down
	this->hsFields.ones(this->M, this->Ns);
	//this->hsFields_img = v_1d<v_1d<std::string>>(this->M, v_1d<std::string>(this->Ns," "));
	// Green's function matrix
	this->green_up.zeros(this->Ns, this->Ns);
	this->green_down.zeros(this->Ns, this->Ns);
	this->tempGreen_up.zeros(this->Ns, this->Ns);
	this->tempGreen_down.zeros(this->Ns, this->Ns);
	this->Q_up.zeros(this->Ns, this->Ns);
	this->Q_down.zeros(this->Ns, this->Ns);
	this->P_up.zeros(this->Ns, this->Ns);
	this->P_down.zeros(this->Ns, this->Ns);
	this->R_up.zeros(this->Ns, this->Ns);
	this->R_down.zeros(this->Ns, this->Ns);
	this->D_down.zeros(this->Ns);
	this->D_up.zeros(this->Ns);
	this->T_down.zeros(this->Ns, this->Ns);
	this->T_up.zeros(this->Ns, this->Ns);

	// set equal time greens
	this->g_down_eq = v_1d<arma::mat>(this->M, arma::zeros(this->Ns, this->Ns));
	this->g_up_eq = v_1d<arma::mat>(this->M, arma::zeros(this->Ns, this->Ns));
	//this->g_up_time = v_2d<arma::mat>(this->M,v_1d<arma::mat>(this->M, arma::zeros(this->Ns, this->Ns)));
	//this->g_down_time = v_2d<arma::mat>(this->M,v_1d<arma::mat>(this->M, arma::zeros(this->Ns, this->Ns)));

	// Set HS fields
	this->set_hs();

	// Calculate something
	this->cal_hopping_exp();
	this->cal_int_exp();
	this->cal_B_mat();
	// Precalculate the multipliers of B matrices for convinience
	for (int i = 0; i < this->p; i++) {
		this->cal_B_mat_cond(i);
	}
}

// -------------------------------------------------------- B MATS --------------------------------------------------------

/// <summary>
/// Precalculate the multiplications of B matrices
/// </summary>
/// <param name="which_sector"></param>
void hubbard::HubbardQR::cal_B_mat_cond(int which_sector)
{
	int tim = which_sector*this->M_0;
	this->b_down_condensed[which_sector] = this->b_mat_down[tim];
	this->b_up_condensed[which_sector] = this->b_mat_up[tim];
#pragma omp parallel for num_threads(this->inner_threads)
	for (int i = 1; i < this->M_0; i++) {
		this->b_down_condensed[which_sector] = this->b_mat_down[tim + i] * this->b_down_condensed[which_sector];
		this->b_up_condensed[which_sector] = this->b_mat_up[tim + i] * this->b_up_condensed[which_sector];
		//this->b_down_condensed[which_sector] = this->b_down_condensed[which_sector] * this->b_mat_down[tim + i];
		//this->b_up_condensed[which_sector] =  this->b_up_condensed[which_sector] * this->b_mat_up[tim + i];
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="l_start"></param>
/// <param name="l_end"></param>
/// <param name="tmp_up"></param>
/// <param name="tmp_down"></param>
void hubbard::HubbardQR::b_mat_multiplier_left(int l_start, int l_end, arma::mat& tmp_up, arma::mat& tmp_down)
{
	int timer = l_start;
	const int how_many = abs(l_end - l_start);
	tmp_up = this->b_mat_up[timer];
	tmp_down = this->b_mat_down[timer];
	for (int j = 1; j <= how_many; j++) {
		timer++;
		if (timer == this->M) timer = 0;
		tmp_down = this->b_mat_down[timer] * tmp_down;
		tmp_up = this->b_mat_up[timer] * tmp_up;
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="l_start"></param>
/// <param name="l_end"></param>
/// <param name="tmp_up"></param>
/// <param name="tmp_down"></param>
void hubbard::HubbardQR::b_mat_multiplier_right(int l_start, int l_end, arma::mat& tmp_up, arma::mat& tmp_down)
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

/// <summary>
/// 
/// </summary>
/// <param name="l_start"></param>
/// <param name="l_end"></param>
/// <param name="tmp_up"></param>
/// <param name="tmp_down"></param>
void hubbard::HubbardQR::b_mat_multiplier_left_inv(int l_start, int l_end, arma::mat& tmp_up, arma::mat& tmp_down)
{
	int timer = l_start;
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

/// <summary>
/// 
/// </summary>
/// <param name="l_start"></param>
/// <param name="l_end"></param>
/// <param name="tmp_up"></param>
/// <param name="tmp_down"></param>
void hubbard::HubbardQR::b_mat_multiplier_right_inv(int l_start, int l_end, arma::mat& tmp_up, arma::mat& tmp_down)
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
// -------------------------------------------------------- GREENS --------------------------------------------------------

/// <summary>
/// 
/// </summary>
/// <param name="tim"></param>
/// <param name="toll"></param>
/// <param name="print_greens"></param>
void hubbard::HubbardQR::compare_green_direct(int tim, double toll, bool print_greens)
{
	arma::mat tmp_up(this->Ns, this->Ns, arma::fill::eye);
	arma::mat tmp_down(this->Ns, this->Ns, arma::fill::eye);
	for (int i = 0; i < this->M; i++) {
		tmp_up = this->b_mat_up[tim] * tmp_up;
		tmp_down = this->b_mat_down[tim] * tmp_down;
		tim = (tim + 1) % this->M;
	}
	tmp_up = (arma::eye(this->Ns, this->Ns) + tmp_up).i();
	tmp_down = (arma::eye(this->Ns, this->Ns) + tmp_down).i();
	bool up = approx_equal(this->green_up, tmp_up, "absdiff", toll) ;
	bool down = approx_equal(this->green_down, tmp_down, "absdiff", toll);
	stout << " -------------------------------- FOR TIME : " << tim << std::endl;
	stout << "up Green:\n" << (up ? "THE SAME!" : "BAAAAAAAAAAAAAAAAAAAAAAD!") << std::endl;
	if(print_greens)
		stout << this->green_up - tmp_up << std::endl;
	stout << "down Green:\n" << (down ? "THE SAME!" : "BAAAAAAAAAAAAAAAAAAAAAAD!") << std::endl;
	if(print_greens)
		stout << this->green_down - tmp_down << "\n\n\n";
}

// -------------------------------------------------------- EQUAL 

/// <summary>
/// Calculate Green with QR decomposition using LOH : doi:10.1016/j.laa.2010.06.023
/// For more look into :
/// "Advancing Large Scale Many-Body QMC Simulations on GPU Accelerated Multicore Systems"
/// In order to do that the M_0 and p variables will be used to divide the multiplication into smaller chunks of matrices
/// </summary>
/// <param name="which_time"></param>
void hubbard::HubbardQR::cal_green_mat(int which_time) {

	int tim = (which_time);
	int sec = static_cast<int>(which_time / this->M_0);
	int sector_end = (sec + 1) * this->M_0 - 1;
	// multiply those B matrices that are not yet multiplied
	b_mat_multiplier_left(tim, sector_end, tempGreen_up, tempGreen_down);

	if (!arma::qr(Q_up, R_up, P_up, this->tempGreen_up, "matrix")) throw "decomposition failed\n";
	if (!arma::qr(Q_down, R_down, P_down, this->tempGreen_down, "matrix")) throw "decomposition failed\n";

	T_down = (diagmat(R_down).i()) * R_down * (P_down.t());
	T_up = (diagmat(R_up).i()) * R_up * (P_up.t());

	for (int i = 1; i < this->p; i++)
	{
		// starting the multiplication

		sec++;
		if (sec == this->p) sec = 0;
		multiplyMatricesQrFromRight(this->b_up_condensed[sec], this->tempGreen_up, Q_up, R_up, P_up, T_up);
		multiplyMatricesQrFromRight(this->b_down_condensed[sec], this->tempGreen_down, Q_down, R_down, P_down, T_down);
		//tim = (which_time + i * this->M_0) % this->M;
		//b_mat_multiplier_left(tim, tim + this->M_0, tempGreen_up, tempGreen_down);
		//multiplyMatricesQr(tempGreen_up, tempGreen_up, Q_up, R_up, P_up, T_up);
		//multiplyMatricesQr(tempGreen_down, tempGreen_down, Q_down, R_down, P_down, T_down);
	}
	// we need to handle the last matrices that ale also away from M_0 cycle
	sec++;
	if (sec == this->p) sec = 0;
	sector_end = which_time - 1;
	tim = sec * this->M_0;
	b_mat_multiplier_left(tim, sector_end, tempGreen_up, tempGreen_down);
	multiplyMatricesQrFromRight(tempGreen_up, tempGreen_up, Q_up, R_up, P_up, T_up);
	multiplyMatricesQrFromRight(tempGreen_down, tempGreen_down, Q_down, R_down, P_down, T_down);

	//stout << std::endl;
	//this->green_up = T_up.i() * (Q_up.t() * T_up.i() + diagmat(R_up)).i()*Q_up.t();
	//this->green_down = T_down.i() * (Q_down.t() * T_down.i() + diagmat(R_down)).i()*Q_down.t();
	// Correction terms
	D_up = diagvec(R_up);
	D_down = diagvec(R_down);

	arma::vec Db_up(this->Ns, arma::fill::ones);
	arma::vec Db_down(this->Ns, arma::fill::ones);
	
	for (int i = 0; i < this->Ns; i++)
	{
		//if (abs(D_up(i)) > 1) {
		if (abs(R_up(i,i)) > 1) {
			Db_up(i) = R_up(i,i);
			//Db_up(i) = D_up(i);
			//Ds_up(i) = 1;
			D_up(i) = 1;
		}
	
		//if (abs(D_down(i)) > 1) {
		if (abs(R_down(i,i)) > 1) {
			
			//Db_down(i) = D_down(i);
			Db_down(i) = R_down(i,i);
			//Ds_down(i) = 1;
			D_down(i) = 1;
		}
	}
	//this->green_up = arma::solve((diagmat(Db_up).i()) * Q_up.t() + diagmat(Ds_up) * T_up, (diagmat(Db_up).i())*Q_up.t());
	//this->green_down = arma::solve((diagmat(Db_down).i()) * Q_down.t() + diagmat(Ds_down) * T_down, (diagmat(Db_down).i())*Q_down.t());
	this->green_up = arma::solve((diagmat(Db_up).i()) * Q_up.t() + diagmat(D_up) * T_up, (diagmat(Db_up).i())*Q_up.t());
	this->green_down = arma::solve((diagmat(Db_down).i()) * Q_down.t() + diagmat(D_down) * T_down, (diagmat(Db_down).i())*Q_down.t());
}



// <summary>
/// Calculate Green with QR decomposition using LOH : doi:10.1016/j.laa.2010.06.023 with premultiplied B matrices
/// For more look into :
/// "Advancing Large Scale Many-Body QMC Simulations on GPU Accelerated Multicore Systems"
/// In order to do that the M_0 and p variables will be used to divide the multiplication into smaller chunks of matrices
/// </summary>
/// <param name="which_time"></param>
void hubbard::HubbardQR::cal_green_mat_cycle(int sector) {
	//stout << "STARTING CALCULATING GREEN FOR : " << which_time << std::endl;
	//bool loh = true;
	int sec = sector;

	if (!arma::qr(Q_up, R_up, P_up, this->b_up_condensed[sector], "matrix")) throw "decomposition failed\n";
	if (!arma::qr(Q_down, R_down, P_down, this->b_down_condensed[sector], "matrix")) throw "decomposition failed\n";

	T_up = ((diagmat(R_up).i()) * R_up) * (P_up.t());
	T_down = ((diagmat(R_down).i()) * R_down) * (P_down.t());


	for (int i = 1; i < this->p; i++) {
		sec++;
		if (sec == this->p) sec = 0;
		multiplyMatricesQrFromRight(this->b_up_condensed[sec], this->tempGreen_up, Q_up, R_up, P_up, T_up);
		multiplyMatricesQrFromRight(this->b_down_condensed[sec], this->tempGreen_down, Q_down, R_down, P_down, T_down);
	}
	
	// Correction terms
	//arma::vec Ds_up = D_up;
	//arma::vec Ds_down = D_down;

	D_up = diagvec(R_up);
	D_down = diagvec(R_down);

	arma::vec Db_up(this->Ns, arma::fill::ones);
	arma::vec Db_down(this->Ns, arma::fill::ones);
	
	for (int i = 0; i < this->Ns; i++)
	{
		//if (abs(D_up(i)) > 1) {
		if (abs(R_up(i,i)) > 1) {
			Db_up(i) = R_up(i,i);
			//Db_up(i) = D_up(i);
			//Ds_up(i) = 1;
			D_up(i) = 1;
		}
	
		//if (abs(D_down(i)) > 1) {
		if (abs(R_down(i,i)) > 1) {
			
			//Db_down(i) = D_down(i);
			Db_down(i) = R_down(i,i);
			//Ds_down(i) = 1;
			D_down(i) = 1;
		}
	}
	//this->green_up = arma::solve((diagmat(Db_up).i()) * Q_up.t() + diagmat(Ds_up) * T_up, (diagmat(Db_up).i())*Q_up.t());
	//this->green_down = arma::solve((diagmat(Db_down).i()) * Q_down.t() + diagmat(Ds_down) * T_down, (diagmat(Db_down).i())*Q_down.t());
	this->green_up = arma::solve((diagmat(Db_up).i()) * Q_up.t() + diagmat(D_up) * T_up, (diagmat(Db_up).i())*Q_up.t());
	this->green_down = arma::solve((diagmat(Db_down).i()) * Q_down.t() + diagmat(D_down) * T_down, (diagmat(Db_down).i())*Q_down.t());

		//this->green_up = T_up.i() * (Q_up.t() * T_up.i() + diagmat(D_up)).i()*Q_up.t();
		//this->green_down = T_down.i() * (Q_down.t() * T_down.i() + diagmat(D_down)).i()*Q_down.t();
	//}	
}

// -------------------------------------------------------- TIMES 

/// <summary>
/// 
/// </summary>
/// <param name="tau1"></param>
/// <param name="tau2"></param>
void hubbard::HubbardQR::cal_green_mat_times()
{
	arma::mat Q2_d(this->Ns, this->Ns);
	arma::mat Q2_u(this->Ns, this->Ns);
	arma::mat R2_d(this->Ns, this->Ns);																			// right triangular matrix down
	arma::mat R2_u(this->Ns, this->Ns);																			// right triangular matrix up
	arma::mat T2_d(this->Ns, this->Ns);
	arma::mat T2_u(this->Ns, this->Ns);
	arma::umat P2_d(this->Ns, this->Ns);
	arma::umat P2_u(this->Ns, this->Ns);
	
	// save the equal times
	//this->g_up_time[tau1][tau1] = this->g_up_eq[tau1];
	//this->g_down_time[tau1][tau1] = this->g_down_eq[tau1];

	// precompute B(M-1)...B(0)


	for (int slice = 0; slice < this->p; slice++) {
		// M_0 multiplications is "safe"

		int time = slice * this->M_0;
		this->tempGreen_up = this->b_mat_up[time];
		this->tempGreen_down = this->b_mat_down[time];
		// go through all time slices
		for (int time_in_slice = 1; time_in_slice < this->M_0 - 1; time_in_slice++) {
			time++;
			this->tempGreen_up = this->b_mat_up[time] * this->tempGreen_up;
			this->tempGreen_down = this->b_mat_down[time] * this->tempGreen_down;
			// save the first "safe" multiplications for later
			if (slice == 0) {
				//this->g_up_time[time][time] = this->tempGreen_up;
				//this->g_down_time[time][time] = this->tempGreen_down;
			}
			// otherwise safely multiply with LOH
			else {
					// start with saving the first one

			}
		}
		// save the first one as they are given ad hoc
		if (slice == 0) {
			if (!arma::qr(Q_up, R_up, P_up, this->b_up_condensed[slice], "matrix")) throw "decomposition failed\n";
			if (!arma::qr(Q_down, R_down, P_down, this->b_down_condensed[slice], "matrix")) throw "decomposition failed\n";
			T_up = ((diagmat(R_up).i()) * R_up) * (P_up.t());
			T_down = ((diagmat(R_down).i()) * R_down) * (P_down.t());
		}
		else {
			// multiply QR 
			multiplyMatricesQrFromRight(this->b_up_condensed[slice], this->tempGreen_up, Q_up, R_up, P_up, T_up);
			multiplyMatricesQrFromRight(this->b_down_condensed[slice], this->tempGreen_down, Q_down, R_down, P_down, T_down);
		}
	}
	
}

/// <summary>
/// 
/// </summary>
/// <param name="tau1"></param>
/// <param name="tau2"></param>
void hubbard::HubbardQR::cal_green_mat_times_cycle()
{
}


void hubbard::HubbardQR::cal_green_mat_times_hirsh()
{
	this->g_up_time.eye();
	this->g_down_time.eye();

	int col_begin = (this->p-1)*this->Ns;
	int col_end = col_begin + this->Ns - 1;
	int row_begin = 0;
	int row_end = row_begin + this->Ns - 1;
	arma::subview up = g_up_time.submat(row_begin, col_begin, row_end, col_end);
	arma::subview down = g_down_time.submat(row_begin, col_begin, row_end , col_end);
	up = this->b_mat_up[0];
	down = this->b_mat_down[0];
	//up *= -1;
	//down *= -1;
	
	for (int sec = 0; sec < this->M - 1; sec++) {
		row_begin = (sec + 1) * this->Ns;
		row_end = row_begin + this->Ns - 1;
		col_begin = (sec) * this->Ns;
		col_end = col_begin + this->Ns - 1;
		// stout << "\trow sector: " << int(row_begin /this->Ns) << ", col sector" << int(col_begin/this->Ns) << std::endl;
		arma::subview up = tempGreen_up.submat(row_begin, col_begin, row_end, col_end);
		arma::subview down = tempGreen_down.submat(row_begin, col_begin, row_end, col_end);
		up = -b_mat_up[sec + 1];
		down = -b_mat_down[sec + 1];
	}
	this->green_up = this->tempGreen_up.i();
	this->green_down = this->tempGreen_down.i();

}



// -------------------------------------------------------- HELPERS

/// <summary>
/// 
/// </summary>
/// <param name="fptr"></param>
/// <returns></returns>
int hubbard::HubbardQR::sweep_lat_sites(std::function<int(int)> fptr)
{
	int sign = 1;
	for (int j = 0; j < this->Ns; j++) {
		const int lat_site = ran.randomInt_uni(0, this->Ns - 1);
		sign = (fptr)(lat_site);
	}
	return sign;
}

// -------------------------------------------------------- GREEN UPDATERS --------------------------------------------------------

/// <summary>
/// After changing one spin we need to update the Green matrices via the Dyson equation
/// </summary>
/// <param name="lat_site">the site on which HS field has been changed</param>
/// <param name="prob_up">the changing probability for up channel</param>
/// <param name="prob_down">the changing probability for down channel</param>
/// <param name="gamma_up">changing parameter gamma for up channel</param>
/// <param name="gamma_down">changing probability for down channel</param>
void hubbard::HubbardQR::upd_equal_green(int lat_site, double gamma_over_prob_up, double gamma_over_prob_down)
{
	// create temporaries as the elements cannot change inplace

	//arma::Row tmp_eye = arma::zeros(this->Ns).t();
	//tmp_eye(lat_site) = 1;

	//this->green_up -= gamma_over_prob_up * this->green_up.col(lat_site) * (tmp_eye - this->green_up.row(lat_site));
	//this->green_down -= gamma_over_prob_down * this->green_down.col(lat_site) * (tmp_eye - this->green_down.row(lat_site));
	//return;

	//this->tempGreen_up = this->green_up;
	//this->tempGreen_down = this->green_down;
	this->D_up = this->green_up.row(lat_site).t();
	this->D_down = this->green_down.row(lat_site).t();
//#pragma omp parallel for num_threads(this->inner_threads)
	for (int a = 0; a < this->Ns; a++) {
	//for (int b = 0; b < this->Ns; b++) {
		const double delta = (a == lat_site) ? 1 : 0;
		//const double delta = (b == lat_site) ? 1 : 0;
		// those elements change so we can skip saving whole matrix and only remember them
		const double g_ai_up = this->green_up(a,lat_site);
		const double g_ai_down = this->green_down(a, lat_site);
		for (int b = 0; b < this->Ns; b++) {
		//for (int a = 0; a < this->Ns; a++) {
			// SPIN UP
			this->green_up(a,b) -= (delta - g_ai_up) * gamma_over_prob_up * D_up(b);
			//this->green_up(a,b) -= (delta - this->tempGreen_up(a, lat_site)) * gamma_over_prob_up * this->tempGreen_up(lat_site, b);
			// SPIN DOWN
			this->green_down(a,b) -= (delta - g_ai_down) * gamma_over_prob_down * D_down(b);
			//this->green_down(a,b) -= (delta - this->tempGreen_down(a, lat_site)) * gamma_over_prob_down * this->tempGreen_down(lat_site, b);
		}
	}
	
}

/// <summary>
/// Update the Green's matrices after going to next Trotter time, remember, the time is taken to be the previous one
/// <param name="which_time">updating to which_time + 1</param>
/// </summary>
void hubbard::HubbardQR::upd_next_green(int which_time_green) {
	//this->green_up = (this->b_mat_up[which_time_green] * this->green_up) * this->b_mat_up[which_time_green].i();						// LEFT INCREASE
	//this->green_down = (this->b_mat_down[which_time_green] * this->green_down) * this->b_mat_down[which_time_green].i();				// LEFT INCREASE;
	this->green_up = (this->b_mat_up[which_time_green] * this->green_up) * this->b_mat_up_inv[which_time_green];						// LEFT INCREASE
	this->green_down = (this->b_mat_down[which_time_green] * this->green_down) * this->b_mat_down_inv[which_time_green];				// LEFT INCREASE;
}

/// <summary>
/// 
/// </summary>
/// <param name="which_time"></param>
void hubbard::HubbardQR::upd_prev_green(int which_time_green) {
	//this->green_up = (this->b_mat_up[which_time_green - 1].i() * this->green_up) * this->b_mat_up[which_time_green - 1];						// LEFT INCREASE
	//this->green_down = (this->b_mat_down[which_time_green - 1].i() * this->green_down) * this->b_mat_down[which_time_green - 1];				// LEFT INCREASE;
	this->green_up = (this->b_mat_up_inv[which_time_green - 1] * this->green_up) * this->b_mat_up[which_time_green - 1];						// LEFT INCREASE
	this->green_down = (this->b_mat_down_inv[which_time_green - 1] * this->green_down) * this->b_mat_down[which_time_green - 1];				// LEFT INCREASE;
}

// ---------------------------------------------------------------------------------------------------------------- 


/// <summary>
/// 
/// </summary>
/// <param name="im_time_step"></param>
void hubbard::HubbardQR::upd_Green_step(int im_time_step, bool forward) {
	if (forward) {
		if (im_time_step % this->from_scratch == 0) {
			// the B matrices that have changed are before so we substract 1
			//if (this->from_scratch == this->M_0) {
			const int sector_to_upd = myModuloEuclidean(static_cast<int>(im_time_step / double(this->M_0)) - 1, this->p);
			this->cal_B_mat_cond(sector_to_upd);
			this->cal_green_mat_cycle(myModuloEuclidean(static_cast<int>(im_time_step / double(this->M_0)), this->p));
			//}
			//else {
			//	this->cal_green_mat(im_time_step);
			//}
			//stout << "Calculating Green. I am in sector : " << myModuloEuclidean(static_cast<int>(this->current_time / double(this->M_0)), this->p);
			//stout << " , so I need to recalculate sector : " << sector_to_upd << std::endl;
			//compare_green_direct(this->current_time, 1e-8,false);
		}
		else {
			this->upd_next_green(im_time_step - 1);
			//compare_green_direct(this->current_time, 1e-8,false);
			//stout << "updating from time : " << times[im_time_step - 1] << " to time : " << this->current_time << std::endl;
		}
	}
	else {
		if (im_time_step % this->from_scratch == 0) {
			if (this->from_scratch == this->M_0) {
				const int sector_to_upd = myModuloEuclidean(static_cast<int>(this->current_time / double(this->M_0)), this->p);
				this->cal_B_mat_cond(sector_to_upd);
				this->cal_green_mat_cycle(sector_to_upd);
			}
			else {
				this->cal_green_mat(im_time_step);
			}
			
			if(im_time_step != 0) // if zero we don't need to go lower ;o
				this->upd_prev_green(im_time_step);	// we must go back by one
		}
		else // if zero we don't need to go lower ;o
		{
			if(im_time_step != 0) // if zero we don't need to go lower ;o
				this->upd_prev_green(im_time_step);	// we must go back by one
		}
	}
}
// -------------------------------------------------------- CALCULATORS

/// <summary>
/// A single step for calculating averages inside a loop
/// </summary>
/// <param name="current_elem_i"> Current Green matrix element in averages</param>
void hubbard::HubbardQR::av_single_step(int current_elem_i, int sign)
{
	const arma::mat& g_up = this->g_up_eq[this->current_time];
	const arma::mat& g_down = this->g_down_eq[this->current_time];
	//this->avs->av_sign += sign;
	// m_z
	const double m_z2 = this->cal_mz2(sign, current_elem_i, g_up, g_down);
	this->avs->av_M2z += m_z2;
	this->avs->sd_M2z += m_z2 * m_z2;
	// m_x
	const double m_x2 = this->cal_mx2(sign, current_elem_i, g_up, g_down);
	this->avs->av_M2x += m_x2;
	this->avs->sd_M2x += m_x2 * m_x2;
	// occupation
	const double occ = this->cal_occupation(sign, current_elem_i, g_up, g_down);
	this->avs->av_occupation += occ;
	this->avs->sd_occupation += occ * occ;
	// kinetic energy
	const double Ek = this->cal_kinetic_en(sign, current_elem_i, g_up, g_down);
	this->avs->av_Ek += Ek;
	this->avs->av_Ek2 += Ek * Ek;
	// Correlations
	for (int j = 0; j < this->Ns; j++) {
		const int current_elem_j = j;
		const int j_minus_i_z = this->lattice->get_coordinates(current_elem_j, 2) - this->lattice->get_coordinates(current_elem_i, 2);
		const int j_minus_i_y = this->lattice->get_coordinates(current_elem_j, 1) - this->lattice->get_coordinates(current_elem_i, 1);
		const int j_minus_i_x = this->lattice->get_coordinates(current_elem_j, 0) - this->lattice->get_coordinates(current_elem_i, 0);
		const int z = j_minus_i_z + this->lattice->get_Lz() - 1;
		const int y = j_minus_i_y + this->lattice->get_Ly() - 1;
		const int x = j_minus_i_x + this->lattice->get_Lx() - 1;
		// normal equal - time correlations
		this->avs->av_M2z_corr[x][y][z] += this->cal_mz2_corr(sign, current_elem_i, current_elem_j, g_up, g_down);
		this->avs->av_occupation_corr[x][y][z] += this->cal_occupation_corr(sign, current_elem_i, current_elem_j, g_up, g_down);
		this->avs->av_ch2_corr[x][y][z] += this->cal_ch_correlation(sign, current_elem_i, current_elem_j, g_up, g_down) / (this->Ns * 2.0);
	}
}
// ---------------------------------------------------------------------------------------------------------------- HEAT BATH ----------------------------------------------------------------------------------------------------------------



/// <summary>
/// 
/// </summary>
void hubbard::HubbardQR::sweep_0_M(std::function<int(int)> ptfptr, bool save_greens)
{
	//double sign_up = 1;
	//double sign_down = 1;
	//double val;
	//this->cal_green_mat_cycle(0);
	//arma::log_det(val, sign_up, this->green_up);
	//arma::log_det(val, sign_down, this->green_down);

	this->config_sign = 1;//static_cast<int>(sign_up * sign_down);

	for (int time_im = 0; time_im < this->M; time_im++) {
		// imaginary Trotter times
		this->current_time = time_im;//tim[time_im];
		this->upd_Green_step(time_im, true);
		//stout << "Current time is : " << this->current_time << std::endl;
		this->config_sign = (this->sweep_lat_sites(ptfptr) > 0) ? +this->config_sign : -this->config_sign;
		if (save_greens) {
			this->g_up_eq[time_im] = this->green_up;
			this->g_down_eq[time_im] = this->green_down;
		}
		//else
		//	sign > 0 ? this->pos_num++ : this->neg_num++;
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="function"></param>
void hubbard::HubbardQR::sweep_M_0(std::function<int(int)> ptfptr, bool save_greens)
{
	int sign = this->config_sign;
	for (int time_im = this->M-1; time_im >= 0; time_im--) {
		// imaginary Trotter times
		this->current_time = time_im;//tim[time_im];
		//stout << "Current time is : " << this->current_time << std::endl;
		sign = sweep_lat_sites(ptfptr) > 0 ? +sign : -sign;
		if (save_greens) {
			this->g_up_eq[time_im] = this->green_up;
			this->g_down_eq[time_im] = this->green_down;
		}
		//else
		//	sign > 0 ? this->pos_num++ : this->neg_num++;
		this->upd_Green_step(time_im, false);
	}
	this->config_sign = sign;// (sign == 1) ? +this->config_sign : -this->config_sign;
}



/// <summary>
/// Single step for the candidate to flip the HS field
/// </summary>
/// <param name="lat_site">the candidate lattice site</param>
/// <returns>sign of probability</returns>
int hubbard::HubbardQR::heat_bath_single_step(int lat_site)
{
	//this->hsFields[this->current_time][lat_site] = -this->hsFields[this->current_time][lat_site];			// try to flip before, why not
	auto [gamma_up, gamma_down] = this->cal_gamma(lat_site);										// first up then down
	auto [proba_up, proba_down] = this->cal_proba(lat_site, gamma_up, gamma_down);					// take the probabilities

	double proba = (proba_up * proba_down);															// Metropolis probability
	if(this->U < 0) proba *= (this->gammaExp[1] + 1);												// add phase factor for 
	proba = proba / (1.0 + proba);																	// heat-bath probability
	//proba = std::min(proba, 1.0);																		// metropolis
	const int sign = (proba > 0) ? 1 : -1;
	if(this->ran.randomReal_uni() <= sign * proba){
		const double delta_up = gamma_up + 1;
		const double delta_down = gamma_down + 1;
		this->hsFields(this->current_time, lat_site) *= -1;
		//this->upd_int_exp(lat_site, delta_up, delta_down);
		//this->cal_B_mat(this->current_time);
		this->upd_B_mat(lat_site, delta_up, delta_down);											// update the B matrices
		this->upd_equal_green(lat_site, gamma_up/proba_up, gamma_down/proba_down);					// update Greens via Dyson
	}
	return sign;
}

/// <summary>
/// Single step for the candidate to flip the HS field
/// </summary>
/// <param name="lat_site">the candidate lattice site</param>
/// <returns>sign of probability</returns>
int hubbard::HubbardQR::heat_bath_single_step_no_upd(int lat_site)
{
	//this->hsFields[this->current_time][lat_site] = -this->hsFields[this->current_time][lat_site];	// try to flip before, why not
	const auto [gamma_up, gamma_down] = this->cal_gamma(lat_site);									// first up then down
	const auto [proba_up, proba_down] = this->cal_proba(lat_site, gamma_up, gamma_down);			// take the probabilities
	double proba = proba_up * proba_down;															// Metropolis probability
	//double multiplier = exp(2*hsFields[current_time][lat_site]*lambda);									// https://iopscience.iop.org/article/10.1088/1742-6596/1483/1/012002/pdf
	proba = proba / (1.0 + proba);																	// heat-bath probability
	return sgn(proba);
}

/// <summary>
/// Single step for the candidate to flip the HS field with saving configurations
/// </summary>
/// <param name="lat_site">the candidate lattice site</param>
/// <returns>sign of probability</returns>
int hubbard::HubbardQR::heat_bath_single_step_conf(int lat_site)
{
	std::ofstream file_conf, file_log;														// savefiles
	std::string name_conf, name_log;														// filenames to save
	const auto [gamma_up, gamma_down] = this->cal_gamma(lat_site);							// first up then down
	const auto [proba_up, proba_down] = this->cal_proba(lat_site, gamma_up, gamma_down);	// take the probabilities
	double proba = proba_up * proba_down;													// Metropolis probability
	proba = proba / (1.0 + proba);															// heat-bath probability
	int sign = 1;
	if (proba < 0) {
		sign = -1;
		proba = -proba;																		// set abs(proba)
		if (this->neg_num <= this->pos_num) {												// to maitain the same number of both signs
			name_conf = this->neg_dir + "negative_" + this->info + \
				",n=" + std::to_string(this->neg_num) + ".dat";
			name_log = this->neg_log;
		}
	}
	else {
		if (this->neg_num <= this->pos_num) {												// to maitain the same number of both signs
			name_conf = this->pos_dir + "positive_" + this->info + \
				",n=" + std::to_string(this->pos_num) + ".dat";
			name_log = this->pos_log;
		}
	}
	// open files
	file_log.open(name_log);
	file_conf.open(name_conf);
	if (!file_conf.is_open() || !file_log.is_open()) {
		std::cout << "Couldn't open either: " + name_log + " , or " + name_conf + "\n";
		throw - 1;
		exit(-1);
	}
	else {
		this->print_hs_fields(file_conf, this->current_time, lat_site, this->hsFields(this->current_time,lat_site));
		file_conf.close();
		file_log << name_conf << "\t" << proba << "\t" << sign << std::endl;
		file_log.close();
	}
	// continue with a standard approach
	if (this->ran.randomReal_uni() < abs(proba)) {
		this->hsFields(this->current_time,lat_site) *= -1;
		this->upd_B_mat(lat_site, gamma_up + 1, gamma_down + 1);								// update the B matrices
		this->upd_equal_green(lat_site, gamma_up/proba_up, gamma_down/proba_down);		// update Greens via Dyson
	}
	return sgn(proba);
}


/// <summary>
/// Drive the system to equilibrium with heat bath
/// </summary>
/// <param name="mcSteps">Number of Monte Carlo steps</param>
/// <param name="conf">If or if not to save configurations</param>
/// <param name="quiet">If should be quiet</param>
void hubbard::HubbardQR::heat_bath_eq(int mcSteps, bool conf, bool quiet)
{
	auto start = std::chrono::high_resolution_clock::now();
	if (!quiet && mcSteps != 1) {
#pragma omp critical
		stout << "\t\t----> STARTING RELAXING FOR : " + this->info << std::endl;
		this->neg_num = 0;																				// counter of negative signs
		this->pos_num = 0;																				// counter of positive signs
	}
	// Progress bar
	auto progress = pBar();
	const double percentage = 33.33;
	const int percentage_steps = static_cast<int>(percentage * mcSteps / 100.0);

	// function
	std::function<int(int)> fptr = std::bind(& HubbardQR::heat_bath_single_step, this, std::placeholders::_1);	// pointer to non-saving configs;
	// choose the correct function
	if (conf) {
		stout << "Saving configurations of Hubbard Stratonovich fields\n";
		fptr = std::bind(&HubbardQR::heat_bath_single_step_conf, this, std::placeholders::_1);			// pointer to saving configs
	}

	// sweep all
	bool save_greens = false;
	for (int step = 0; step < mcSteps; step++) {
		// Monte Carlo steps

		// save current equal time Greens after relaxation
		if(step == mcSteps - 1) save_greens = true;
		this->sweep_0_M(fptr, save_greens);
		//if(step == mcSteps - 1) save_greens = true;
		//this->sweep_M_0(fptr, save_greens);
		if (!quiet) {
			this->config_sign > 0 ? this->pos_num++ : this->neg_num++;								// increase sign
			if (!quiet && step % percentage_steps == 0) {
#pragma omp critical
				{
					stout << "\t\t\t\t-> time: " << tim_s(start) << " -> RELAXATION PROGRESS for " << this->info << " : \n";
					progress.print();
					stout << std::endl;
					progress.update(percentage);
				}
			}
		}
	}
}

/// <summary>
///
/// </summary>
/// <param name="corr_time"></param>
/// <param name="avNum"></param>
/// <param name="quiet"></param>
/// <param name="times"></param>
void hubbard::HubbardQR::heat_bath_av(int corr_time, int avNum, bool quiet, bool times)
{
	auto start = std::chrono::high_resolution_clock::now();
#pragma omp critical
	stout << "\t\t----> STARTING AVERAGING FOR : " + this->info << std::endl;
	this->neg_num = 0;																				// counter of negative signs
	this->pos_num = 0;	
	this->avs->av_sign = 0;				// counter of positive signs

	// allocate memory for time displaced functions
	if (times) {
		try {
			this->g_up_time.eye(this->Ns*this->M,this->Ns*this->M);
			this->g_down_time.eye(this->Ns*this->M,this->Ns*this->M);
		}
		catch (...) {
			throw "Can't handle so much memory";
		}
	}


	// Progress bar
	auto progress = pBar();
	const double percentage = 34;
	const auto percentage_steps = static_cast<int>(percentage * avNum / 100.0);
	
	// times
	const int tim_size = this->M;
	for (int step = 0; step < avNum; step++) {
		// Monte Carlo steps
		for (int time_im = 0; time_im < tim_size; time_im++) {
			// imaginary Trotter times
			this->current_time = time_im;
			for (int i = 0; i < this->Ns; i++)
				// go through the lattice
				this->av_single_step(i, this->config_sign);										// collect all averages
			if (times) {
				// after sweep this sector shall be recalculated
				//this->cal_B_mat_cond(this->p-1);
				this->cal_green_mat_times_hirsh();
			}
			//this->avs->av_gr_down += this->g_downs_eq[this->current_time];
			//this->avs->av_gr_up += this->g_ups_eq[this->current_time];
		}
		this->config_sign > 0 ? this->pos_num++ : this->neg_num++;								// increase sign
		//this->avs->av_sign += this->config_sign;
		
		// kill correlations
		this->heat_bath_eq(corr_time, false, true);
		
		if (step % percentage_steps == 0) {
#pragma omp critical
			{
				stout << "\t\t\t\t-> time: " << tim_s(start) << " -> AVERAGES PROGRESS for " << this->info << " : \n";
				progress.print();
				stout << std::endl;
			}
			progress.update(percentage);
		}
	}
	// After

	this->av_normalise(avNum, tim_size, times);
}

// ---------------------------------------------------------------------------------------------------------------- PUBLIC CALCULATORS ----------------------------------------------------------------------------------------------------------------

