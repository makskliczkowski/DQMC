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
	this->lambda = std::acosh(std::exp((abs(this->U) * this->dtau) / 2.0));

	// Calculate changing exponents before, not to calculate exp all the time 
	this->gammaExp = { std::exp(2.0 * this->lambda), std::exp(-2.0 * this->lambda) };			// 0 -> sigma * hsfield = 1, 1 -> sigma * hsfield = -1

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
	this->int_exp_down.zeros(this->Ns, this->M);
	this->int_exp_up.zeros(this->Ns, this->M);

	// all times exponents multiplication
	this->b_mat_up = v_1d<arma::cx_mat>(this->M, arma::cx_mat(this->Ns, this->Ns, arma::fill::zeros));
	this->b_mat_down = v_1d<arma::cx_mat>(this->M, arma::cx_mat(this->Ns, this->Ns, arma::fill::zeros));
	this->b_up_condensed = std::vector<arma::cx_mat>(this->p, arma::cx_mat(this->Ns, this->Ns, arma::fill::zeros));
	this->b_down_condensed = std::vector<arma::cx_mat>(this->p, arma::cx_mat(this->Ns, this->Ns, arma::fill::zeros));

	// all times hs fields for real spin up and down
	this->hsFields.ones(this->M, this->Ns);
	//this->hsFields_img = v_1d<v_1d<std::string>>(this->M, v_1d<std::string>(this->Ns));
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
	this->g_ups_eq = v_1d<arma::cx_mat>(this->M,arma::zeros(this->Ns, this->Ns));
	this->g_downs_eq = v_1d<arma::cx_mat>(this->M,arma::zeros(this->Ns, this->Ns));

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

// -------------------------------------------------------- GREENS --------------------------------------------------------

/// <summary>
/// 
/// </summary>
/// <param name="tim"></param>
/// <param name="toll"></param>
/// <param name="print_greens"></param>
void hubbard::HubbardQR::compare_green_direct(int tim, double toll, bool print_greens)
{
	arma::cx_mat tmp_up(this->Ns, this->Ns, arma::fill::eye);
	arma::cx_mat tmp_down(this->Ns, this->Ns, arma::fill::eye);
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

/// <summary>
/// Calculate Green with QR decomposition using LOH : doi:10.1016/j.laa.2010.06.023
/// For more look into :
/// "Advancing Large Scale Many-Body QMC Simulations on GPU Accelerated Multicore Systems"
/// In order to do that the M_0 and p variables will be used to divide the multiplication into smaller chunks of matrices
/// </summary>
/// <param name="which_time"></param>
void hubbard::HubbardQR::cal_green_mat(int which_time) {
	auto multiplier = [&](auto& tim) mutable {
		this->tempGreen_up = this->b_mat_up[tim];
		this->tempGreen_down = this->b_mat_down[tim];
		//stout << tim << std::endl;
		for (int j = 1; j < this->M_0; j++) {
			tim++;
			if (tim == this->M) tim = 0;
			//stout << tim << std::endl;
			this->tempGreen_up = this->b_mat_up[tim] * this->tempGreen_up;
			this->tempGreen_down = this->b_mat_down[tim] * this->tempGreen_down;
			//this->green_up = this->green_up * this->b_mat_up[tim];
			//this->green_down = this->green_down * this->b_mat_down[tim];
		}
		//stout << std::endl;
	};
	int tim = (which_time);
	multiplier(tim);

	if (!arma::qr(Q_up, R_up, P_up, this->tempGreen_up, "matrix")) throw "decomposition failed\n";
	if (!arma::qr(Q_down, R_down, P_down, this->tempGreen_down, "matrix")) throw "decomposition failed\n";

	D_down = diagvec(R_down);
	D_up = diagvec(R_up);
	T_down = (diagmat(D_down).i()) * R_down * (P_down.t());
	T_up = (diagmat(D_up).i()) * R_up * (P_up.t());

	for (int i = 1; i < this->p; i++)
	{
		// starting the multiplication
		tim = (which_time + i * this->M_0) % this->M;
		multiplier(tim);
		this->tempGreen_up = (this->tempGreen_up * Q_up) * diagmat(D_up);				// multiply by the former ones
		this->tempGreen_down = (this->tempGreen_down * Q_down) * diagmat(D_down);		// multiply by the former ones
		//this->green_down.print();

		if (!arma::qr(Q_up, R_up, P_up, this->tempGreen_up)) throw "decomposition failed\n";
		if (!arma::qr(Q_down, R_down, P_down, this->tempGreen_down)) throw "decomposition failed\n";

		D_up = diagvec(R_up);
		D_down = diagvec(R_down);
		//D_up.print();

		T_up = ((diagmat(D_up).i()) * R_up) * P_up.t() * T_up;
		T_down = ((diagmat(D_down).i()) * R_down) * P_down.t() * T_down;
	}
	//stout << std::endl;
	this->green_up = T_up.i() * (Q_up.t() * T_up.i() + diagmat(D_up)).i()*Q_up.t();
	this->green_down = T_down.i() * (Q_down.t() * T_down.i() + diagmat(D_down)).i()*Q_down.t();
	// Correction terms
	/*
	arma::vec Ds_up = D_up;
	arma::vec Ds_down = D_down;

	arma::vec Db_up(this->Ns, arma::fill::ones);
	arma::vec Db_down(this->Ns, arma::fill::ones);
	
	for (int i = 0; i < this->Ns; i++)
	{
		if (abs(D_up(i)) > 1) {
			Db_up(i) = D_up(i);
			Ds_up(i) = 1;
		}

		if (abs(D_down(i)) > 1) {
			Db_down(i) = D_down(i);
			Ds_down(i) = 1;
		}
	}
	this->green_up = arma::solve((diagmat(Db_up).i()) * Q_up.t() + diagmat(Ds_up) * T_up, (diagmat(Db_up).i())*Q_up.t());
	this->green_down = arma::solve((diagmat(Db_down).i()) * Q_down.t() + diagmat(Ds_down) * T_down, (diagmat(Db_down).i())*Q_down.t());

	//this->green_down = T_down.i() * (Q_down.st() * T_down.i() + diagmat(D_down)).i() * Q_down.st();
	//this->green_up = T_up.i() * (Q_up.st() * T_up.i() + diagmat(D_up)).i() * Q_up.st();
	//
	//this->green_down = arma::solve((T_down.t().i() * Q_down.t() * diagmat(Db_down) + diagmat(Ds_down)).t(), diagmat(Db_down) * Q_down.t());
	//this->green_up = arma::solve((T_up.t().i() * Q_up.t() * diagmat(Db_up) + diagmat(Ds_up)).t(), diagmat(Db_up) * Q_up.t());
	*/
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
	bool loh = false;
	int sec = sector;
	if (!loh) {
		if (!arma::qr(Q_up, R_up, this->b_up_condensed[sector])) throw "decomposition failed\n";
		if (!arma::qr(Q_down, R_down, this->b_down_condensed[sector])) throw "decomposition failed\n";
		D_down = diagvec(R_down);
		D_up = diagvec(R_up);
		T_down = (diagmat(D_down).i()) * R_down;
		T_up = (diagmat(D_up).i()) * R_up;

		// new  SciPost Phys. Core 2, 011 (2020)
		arma::mat Q_up_tmp(this->Ns, this->Ns, arma::fill::zeros);
		arma::mat Q_down_tmp(this->Ns, this->Ns, arma::fill::zeros);
		arma::mat R_down_tmp(this->Ns, this->Ns, arma::fill::zeros);																								// right triangular matrix down
		arma::mat R_up_tmp(this->Ns, this->Ns, arma::fill::zeros);																									// right triangular matrix up
		arma::vec D_down_tmp(this->Ns, arma::fill::zeros);
		arma::vec D_up_tmp(this->Ns, arma::fill::zeros);
		arma::mat T_down_tmp(this->Ns, this->Ns, arma::fill::zeros);
		arma::mat T_up_tmp(this->Ns, this->Ns, arma::fill::zeros);

		for (int i = 1; i < this->p; i++) {
			sec++;
			if (sec == this->p) sec = 0;
			//stout << sector << std::endl;

			// decompose the second sector
			if (!arma::qr(Q_up_tmp, R_up_tmp, this->b_up_condensed[sec])) throw "decomposition failed\n";
			if (!arma::qr(Q_down_tmp, R_down_tmp, this->b_down_condensed[sec])) throw "decomposition failed\n";

			D_up_tmp = diagvec(R_up_tmp);
			D_down_tmp = diagvec(R_down_tmp);
			T_down_tmp = (diagmat(D_down_tmp).i()) * R_down_tmp;
			T_up_tmp = (diagmat(D_up_tmp).i()) * R_up_tmp;

			this->tempGreen_up = diagmat(D_up_tmp) * ((T_up_tmp * Q_up) * diagmat(D_up));
			this->tempGreen_down = diagmat(D_down_tmp) * ((T_down_tmp * Q_down) * diagmat(D_down));

			if (!arma::qr(Q_up, R_up, this->tempGreen_up)) throw "decomposition failed\n";
			if (!arma::qr(Q_down, R_down, this->tempGreen_down)) throw "decomposition failed\n";

			Q_up = Q_up_tmp * Q_up;
			Q_down = Q_down_tmp * Q_down;
			D_up = diagvec(R_up);
			D_down = diagvec(R_down);
			T_up = ((diagmat(D_up).i()) * R_up) * T_up;
			T_down = ((diagmat(D_down).i()) * R_down) * T_down;
		}
		this->tempGreen_up = Q_up.t() * T_up.i() + diagmat(D_up);
		this->tempGreen_down = Q_down.t() * T_down.i() + diagmat(D_down);

		if (!arma::qr(Q_up_tmp, R_up_tmp, this->tempGreen_up)) throw "decomposition failed\n";
		if (!arma::qr(Q_down_tmp, R_down_tmp, this->tempGreen_down)) throw "decomposition failed\n";

		D_up_tmp = diagvec(R_up_tmp);
		D_down_tmp = diagvec(R_down_tmp);
		T_down_tmp = (diagmat(D_down_tmp).i()) * R_down_tmp;
		T_up_tmp = (diagmat(D_up_tmp).i()) * R_up_tmp;

		this->green_up = (T_up_tmp * T_up).i() * diagmat(D_up_tmp).i() * (Q_up * Q_up_tmp).i();
		this->green_down = (T_down_tmp * T_down).i() * diagmat(D_down_tmp).i() * (Q_down * Q_down_tmp).i();
	}
	else {
		if (!arma::qr(Q_up, R_up, P_up, this->b_up_condensed[sector], "matrix")) throw "decomposition failed\n";
		if (!arma::qr(Q_down, R_down, P_down, this->b_down_condensed[sector], "matrix")) throw "decomposition failed\n";

		D_down = diagvec(R_down);
		D_up = diagvec(R_up);
		T_down = (diagmat(D_down).i()) * R_down * (P_down.t());
		T_up = (diagmat(D_up).i()) * R_up * (P_up.t());
		
		for (int i = 1; i < this->p; i++) {
			sec++;
			if (sec == this->p) sec = 0;
			//stout << sector << std::endl;

			this->tempGreen_up = (this->b_up_condensed[sec] * Q_up) * diagmat(D_up);				// multiply by the former ones
			this->tempGreen_down = (this->b_down_condensed[sec] * Q_down) * diagmat(D_down);		// multiply by the former ones

			if (!arma::qr(Q_up, R_up, P_up, this->tempGreen_up)) throw "decomposition failed\n";
			if (!arma::qr(Q_down, R_down, P_down, this->tempGreen_down)) throw "decomposition failed\n";

			D_up = diagvec(R_up);
			D_down = diagvec(R_down);

			T_up = ((diagmat(D_up).i()) * R_up) * P_up.t() * T_up;
			T_down = ((diagmat(D_down).i()) * R_down) * P_down.t() * T_down;
		}
		// Correction terms
		arma::vec Ds_up = D_up;
		arma::vec Ds_down = D_down;

		arma::vec Db_up(this->Ns, arma::fill::ones);
		arma::vec Db_down(this->Ns, arma::fill::ones);
		
		for (int i = 0; i < this->Ns; i++)
		{
			if (abs(D_up(i)) > 1) {
				Db_up(i) = D_up(i);
				Ds_up(i) = 1;
			}

			if (abs(D_down(i)) > 1) {
				Db_down(i) = D_down(i);
				Ds_down(i) = 1;
			}
		}
		this->green_up = arma::solve((diagmat(Db_up).i()) * Q_up.t() + diagmat(Ds_up) * T_up, (diagmat(Db_up).i())*Q_up.t());
		this->green_down = arma::solve((diagmat(Db_down).i()) * Q_down.t() + diagmat(Ds_down) * T_down, (diagmat(Db_down).i())*Q_down.t());
		//this->green_up = T_up.i() * (Q_up.t() * T_up.i() + diagmat(D_up)).i()*Q_up.t();
		//this->green_down = T_down.i() * (Q_down.t() * T_down.i() + diagmat(D_down)).i()*Q_down.t();
	}	
}

// -------------------------------------------------------- HELPERS

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
	this->green_up = (this->b_mat_up[which_time_green] * this->green_up) * this->b_mat_up[which_time_green].i();						// LEFT INCREASE
	this->green_down = (this->b_mat_down[which_time_green] * this->green_down) * this->b_mat_down[which_time_green].i();				// LEFT INCREASE;
}

/// <summary>
/// 
/// </summary>
/// <param name="which_time"></param>
void hubbard::HubbardQR::upd_prev_green(int which_time_green) {
	this->green_up = (this->b_mat_up[which_time_green - 1].i() * this->green_up) * this->b_mat_up[which_time_green - 1];						// LEFT INCREASE
	this->green_down = (this->b_mat_down[which_time_green - 1].i() * this->green_down) * this->b_mat_down[which_time_green - 1];				// LEFT INCREASE;
}

// ---------------------------------------------------------------------------------------------------------------- 

/// <summary>
/// 
/// </summary>
/// <param name="im_time_step"></param>
/// <param name="times"></param>
void hubbard::HubbardQR::upd_Green_step(int im_time_step, const v_1d<int>& times)
{
	if (im_time_step < this->M) {
		// we have updated the sector of B matrices right befor the new one starting form current time
		if (this->current_time % this->from_scratch == 0) {
			// the B matrices that have changed are before so we substract 1
			//const int sector_to_upd = myModuloEuclidean(static_cast<int>(this->current_time / double(this->M_0)) - 1, this->p);
			//this->cal_B_mat_cond(sector_to_upd);
			//this->cal_green_mat_cycle(myModuloEuclidean(static_cast<int>(this->current_time / double(this->M_0)), this->p));
			this->cal_green_mat(this->current_time);
			//stout << "Calculating Green. I am in sector : " << myModuloEuclidean(static_cast<int>(this->current_time / double(this->M_0)), this->p);
			//stout << " , so I need to recalculate sector : " << sector_to_upd << std::endl;
		}
		else {
			this->upd_next_green(this->current_time - 1);
			//stout << "updating from time : " << times[im_time_step - 1] << " to time : " << this->current_time << std::endl;
		}
	}
	else if (im_time_step == this->M) {
		// calculate without precalculated multipliers on 
		this->cal_green_mat(this->current_time);
		//stout << "Calculating Green on jump : " << std::endl;
	}
	else {
		// the B matrices that have changed are after so update that sector
		if (this->current_time % this->from_scratch == 0) {
			const int sector_to_upd = myModuloEuclidean(static_cast<int>(this->current_time / double(this->M_0)), this->p);
			this->cal_B_mat_cond(sector_to_upd);
			if(sector_to_upd + 1 < this->p)
				this->cal_B_mat_cond(sector_to_upd + 1);
			
			this->cal_green_mat_cycle(sector_to_upd);
			//stout << "Calculating Green. I am in sector : " << sector_to_upd;
			//stout << " , so I need to recalculate sector : " << sector_to_upd << std::endl;
		}
		else
		{
			this->upd_prev_green(this->current_time + 1);
			//stout << "updating from time : " << times[im_time_step - 1] << " to time : " << this->current_time << std::endl;
		}
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="im_time_step"></param>
void hubbard::HubbardQR::upd_Green_step(int im_time_step) {
	if ((im_time_step % this->from_scratch == 0)) {
		// the B matrices that have changed are before so we substract 1
		if (this->from_scratch = this->M_0) {
			const int sector_to_upd = myModuloEuclidean(static_cast<int>(this->current_time / double(this->M_0)) - 1, this->p);
			this->cal_B_mat_cond(sector_to_upd);
			this->cal_green_mat_cycle(myModuloEuclidean(static_cast<int>(this->current_time / double(this->M_0)), this->p));
		}
		else {
			this->cal_green_mat(this->current_time);
		}
		//stout << "Calculating Green. I am in sector : " << myModuloEuclidean(static_cast<int>(this->current_time / double(this->M_0)), this->p);
		//stout << " , so I need to recalculate sector : " << sector_to_upd << std::endl;
		//compare_green_direct(this->current_time, 1e-8,false);
	}
	else {
		this->upd_next_green(this->current_time - 1);
		//compare_green_direct(this->current_time, 1e-8,false);
		//stout << "updating from time : " << times[im_time_step - 1] << " to time : " << this->current_time << std::endl;
	}
}
// -------------------------------------------------------- CALCULATORS

/// <summary>
/// A single step for calculating averages inside a loop
/// </summary>
/// <param name="current_elem_i"> Current Green matrix element in averages</param>
void hubbard::HubbardQR::av_single_step(int current_elem_i, int sign)
{
	const mat& g_up = this->g_ups_eq[this->current_time];
	const mat& g_down = this->g_downs_eq[this->current_time];
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
/// Single step for the candidate to flip the HS field
/// </summary>
/// <param name="lat_site">the candidate lattice site</param>
/// <returns>sign of probability</returns>
int hubbard::HubbardQR::heat_bath_single_step(int lat_site)
{
	//this->hsFields[this->current_time][lat_site] = -this->hsFields[this->current_time][lat_site];			// try to flip before, why not
	const auto [gamma_up, gamma_down] = this->cal_gamma(lat_site);									// first up then down
	const auto [proba_up, proba_down] = this->cal_proba(lat_site, gamma_up, gamma_down);			// take the probabilities


	//this->tempGreen_up = this->green_up;
	//this->tempGreen_down = this->green_down;

	//this->upd_int_exp(lat_site, delta_up, delta_down);
	//this->upd_B_mat(lat_site, delta_up, delta_down);											// update the B matrices
	//this->cal_green_mat(this->current_time);

	//cx_double p_up = (log_det(this->green_up) - log_det(this->tempGreen_up));
	//cx_double p_down = (log_det(this->green_down) - log_det(this->tempGreen_down));
	//double proba = p_up.real() * p_down.real();


	double proba = (proba_up * proba_down);															// Metropolis probability
	//double multiplier = exp(2*hsFields[current_time][lat_site]*lambda);									// https://iopscience.iop.org/article/10.1088/1742-6596/1483/1/012002/pdf
	proba = proba / (1.0 + proba);																	// heat-bath probability
	//proba = std::min(proba, 1.0);																	// metropolis
	//const int sign = (((abs(p_up.imag()) - PI) > 1e-6) ? 1 : -1) * (((abs(p_down.imag()) - PI) > 1e-6) ? 1 : -1);
	const int sign = (proba >= 0) ? 1 : -1;
	/*
	if (sign < 0) {
		std::vector<int> ivec(this->Ns);
		std::iota(ivec.begin(), ivec.end(), 0);
		stout << "\n\nLat_site: " << lat_site << std::endl;
		stout << "up gamma: " << gamma_up << ", down gamma: " << gamma_down << std::endl;
		stout << "up proba: " << proba_up << ", down proba: " << proba_down << std::endl;
		stout << ivec << std::endl;
		this->green_up.diag().t().print("diag of up green");
		this->green_down.diag().t().print("diag of down green");
	}
	*/
	//auto r = this->ran.randomReal_uni(0,1);
	//stout << "random-> " << r << ((r <= proba) ? " <= " : " > " ) << proba << "<-proba\n";
	//if (this->ran.bernoulli(abs(proba))) {
	if(this->ran.randomReal_uni(0,1) <= abs(proba)){
		//cx_double p_u = arma::log_det(this->green_up);
		//cx_double p_d = arma::log_det(this->green_down);
		//p_u = abs(p_u.imag() - PI) < 1e-6 ? -p_u.real() : p_u.real();
		//p_d = abs(p_d.imag() - PI) < 1e-6  ? -p_d.real() : p_d.real();
		const double delta_up = gamma_up + 1;
		const double delta_down = gamma_down + 1;
		//stout << "\tI am in, updating\n";
		this->upd_int_exp(lat_site, delta_up, delta_down);
		//this->cal_B_mat(this->current_time);
		this->upd_B_mat(lat_site, delta_up, delta_down);											// update the B matrices
		//const int sector_to_upd = static_cast<int>(this->current_time / double(this->M_0));
		//this->cal_B_mat_cond(sector_to_upd);
		this->upd_equal_green(lat_site, gamma_up/proba_up, gamma_down/proba_down);					// update Greens via Dyson
		//this->cal_green_mat(this->current_time);
		this->hsFields(this->current_time, lat_site) = -this->hsFields(this->current_time, lat_site);
		//this->hsFields_img[this->current_time][ lat_site] = this->hsFields(this->current_time, lat_site) > 0 ? " " : "|";
		//cx_double p_u2 = arma::log_det(green_up);
		//cx_double p_d2 = arma::log_det(green_down);
		//p_u2 = abs(p_u2.imag() - PI) < 1e-7 ? -p_u2.real() : p_u2.real();
		//p_d2 = abs(p_d2.imag() - PI) < 1e-7 ? -p_d2.real() : p_d2.real();
		//p_u = -p_u2 + p_u;
		//p_d = -p_d2 + p_d;
		//double diff_up = proba_up - exp(p_u).real();
		//double diff_down = proba_down - exp(p_d).real();
		//if (sign < 0) {
		//	stout << "i = " << lat_site << ", time = " << this->current_time << std::endl;
		//	stout << "up from Morrison: " << proba_up << ", from dets: " << exp(p_u).real() << ", difference = " << diff_up  << std::endl;
		//	stout << "down from Morrison: " << proba_down << ", from dets: " << exp(p_d).real() << ", difference = " << diff_down << std::endl << std::endl << std::endl;
		//	stout << "gotem\n";
		//}
		//auto tmp_up = this->b_mat_up[this->current_time];
		//auto tmp_down = this->b_mat_down[this->current_time];
		//this->upd_int_exp(lat_site, gamma_up + 1, gamma_down + 1);
		//this->cal_B_mat(this->current_time);
		//bool up = approx_equal(this->b_mat_up[this->current_time], tmp_up, "absdiff", 1e-6) ;
		//bool down = approx_equal(this->b_mat_down[this->current_time], tmp_down, "absdiff", 1e-6);
		//stout << "up difference: " << (up ? "THE SAME!" : "BAAAAAAAAAAAAAAAAAAAAAAD!") << std::endl;
		//stout << "down difference: " << (down ? "THE SAME!" : "BAAAAAAAAAAAAAAAAAAAAAAD!") << std::endl;
		//sign > 0 ? this->pos_num++ : this->neg_num++;
	}
	//else this->hsFields[this->current_time][lat_site] = -this->hsFields[this->current_time][lat_site];
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
	if (mcSteps != 1) {
#pragma omp critical
		stout << "\t\t----> STARTING RELAXING FOR : " + this->info << std::endl;
		this->neg_num = 0;																				// counter of negative signs
		this->pos_num = 0;																				// counter of positive signs
	}
	// Progress bar
	auto progress = pBar();
	const double percentage = 20;
	const int percentage_steps = static_cast<int>(percentage * mcSteps / 100.0);

	// function
	int (HubbardQR:: * ptfptr)(int);																	// pointer to a single step function depending on whether we do configs or not

	// times
	const int tim_size = this->M;

	if (conf) {
		stout << "Saving configurations of Hubbard Stratonovich fields\n";
		ptfptr = &HubbardQR::heat_bath_single_step_conf;												// pointer to saving configs
	}
	else
		ptfptr = &HubbardQR::heat_bath_single_step;														// pointer to non-saving configs

	for (int step = 0; step < mcSteps; step++) {
		// Monte Carlo steps
		//stout << "Starting sweep number : " << step << std::endl;
		for (int time_im = 0; time_im < tim_size; time_im++) {
			// imaginary Trotter times
			this->current_time = time_im;//tim[time_im];
			//stout << "Current time is : " << this->current_time << std::endl;
			this->upd_Green_step(time_im);
			for (int j = 0; j < this->Ns; j++) {
				this->config_sign = ((this->*ptfptr)(j) >= 0);//) ? +this->config_sign : -this->config_sign;// get current sign and make single step
				if (mcSteps != 1)
					this->config_sign > 0 ? this->pos_num++ : this->neg_num++;								// increase sign
				/*for (int a = 0; a < this->M; a++) {
					for (int b = 0; b < this->Ns; b++) {
						stout << this->hsFields_img[a][b] << "\t";
					}
					stout << std::endl;
				}*/
			}
			// save current equal time Greens after relaxation
			if (step == (mcSteps - 1)) {
				this->g_ups_eq[time_im] = this->green_up;
				this->g_downs_eq[time_im] = this->green_down;
			}
		}

		if (mcSteps != 1 && step % percentage_steps == 0) {
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

	// Progress bar
	auto progress = pBar();
	const double percentage = 25;
	const auto percentage_steps = static_cast<int>(percentage * avNum / 100.0);
	
	// times
	const int tim_size = this->M;
	for (int step = 0; step < avNum; step++) {
		// Monte Carlo steps
		for (int time_im = 0; time_im < tim_size; time_im++) {
			// imaginary Trotter times
			this->current_time = time_im;//tim[time_im];
			for (int i = 0; i < this->Ns; i++) {
				// go through the lattice
				// this->config_sign = this->heat_bath_single_step(i);
				// stout << "sign: " << sign << ", pos: " << this->pos_num << ", neg: " << this->neg_num << std::endl;
				this->av_single_step(i, this->config_sign);										// collect all averages
			}
			this->avs->av_gr_down += this->g_downs_eq[this->current_time];
			this->avs->av_gr_up += this->g_ups_eq[this->current_time];
		}
		this->config_sign > 0 ? this->pos_num++ : this->neg_num++;								// increase sign
		this->avs->av_sign += this->config_sign;
		
		for (int corr = 0; corr < corr_time; corr++)
			this->heat_bath_eq(1, false, true);
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

