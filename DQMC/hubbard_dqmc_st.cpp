#include "include/hubbard_dqmc_st.h"
// ---------------------------------------------------------------------------------------------------------------- HUBBARD MODEL WITH SPACE TINE FORMULATION ----------------------------------------------------------------------------------------------------------------

// -------------------------------------------------------- CONSTRUCTORS

hubbard::HubbardST::HubbardST(const std::vector<double>& t, double dtau, int M_0, double U, double mu, double beta, std::shared_ptr<Lattice> lattice, int threads)
{
	this->lattice = lattice;
	this->inner_threads = threads;
	int Lx = this->lattice->get_Lx();
	int Ly = this->lattice->get_Ly();
	int Lz = this->lattice->get_Lz();
	this->avs = std::make_shared<averages_par>(Lx, Ly, Lz);
	this->t = t;

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
	this->green_size = this->Ns * this->p;

	// Calculate alghorithm parameters
	//this->lambda = 2 * std::atan(tanh((abs(this->U) * this->dtau) / 4.0));
	this->lambda = std::acosh(std::exp((abs(this->U) * this->dtau) / 2.0));

	// Calculate changing exponents before, not to calculate exp all the time
	this->gammaExp = { std::exp(2.0 * this->lambda), std::exp(-2.0 * this->lambda) };			// 0 -> sigma * hsfield = -1, 1 -> sigma * hsfield = 1

	// Helping params
	this->from_scratch = this->M_0;
	this->pos_num = 0;
	this->neg_num = 0;

	// Say hi to the world
#pragma omp critical
	std::cout << "CREATING THE HUBBARD MODEL WITH SPACE-TIME FORMULATION WITH PARAMETERS:" << std::endl \
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
	this->info = "M=" + STR(this->M) + ",M_0=" + STR(this->M_0) + \
		",dtau=" + STR(this->dtau) + ",Lx=" + STR(this->lattice->get_Lx()) + \
		",Ly=" + STR(this->lattice->get_Ly()) + ",Lz=" + STR(this->lattice->get_Lz()) + \
		",beta=" + STR(this->beta) + ",U=" + STR(this->U) + \
		",mu=" + STR(this->mu);

	// Initialize memory
	this->hopping_exp.zeros(this->Ns, this->Ns);

	// interaction for all times
	this->int_exp_down.zeros(this->Ns, this->M);
	this->int_exp_up.zeros(this->Ns, this->M);

	// all times exponents multiplication
	this->b_mat_up = std::vector<arma::mat>(this->M, arma::mat(this->Ns, this->Ns, arma::fill::zeros));
	this->b_mat_down = std::vector<arma::mat>(this->M, arma::mat(this->Ns, this->Ns, arma::fill::zeros));

	// all times hs fields for real spin up and down
	this->hsFields.ones(this->M, this->Ns);

	// Green's function matrix
	this->green_up.zeros(green_size, green_size);
	this->green_down.zeros(green_size, green_size);
	this->tempGreen_up.zeros(green_size, green_size);
	this->tempGreen_down.zeros(green_size, green_size);

	// Set HS fields
	this->set_hs();

	// Calculate something
	this->cal_hopping_exp();
	this->cal_int_exp();
	this->cal_B_mat();
}

// -------------------------------------------------------- B MATS --------------------------------------------------------

// -------------------------------------------------------- GREENS --------------------------------------------------------

/// <summary>
///
/// </summary>
/// <param name="tim"></param>
/// <param name="toll"></param>
/// <param name="print_greens"></param>
void hubbard::HubbardST::compare_green_direct(int tim, double toll, bool print_greens)
{
	/*
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
		*/
	return;
}

/// <summary>
/// </summary>
/// <param name="which_time"></param>
void hubbard::HubbardST::cal_green_mat(int which_time) {
	// stout << "\n\nCALCULATING GREEN in t = " << which_time <<std::endl;
	auto multiplier = [&](auto tim_sector, auto& tim_in_sector, auto& up, auto& down) mutable {
		int tim = tim_in_sector;
		int total_tim = total_time(tim, tim_sector);
		// stout << "starting from B matrix at t = " << total_tim << std::endl;
		up = -this->b_mat_up[total_tim];
		down = -this->b_mat_down[total_tim];
		//stout << tim << std::endl;
		for (int j = 1; j < this->M_0; j++) {
			tim++;
			if (tim == this->M_0) {
				tim = 0;
			}
			total_tim = total_time(tim, tim_sector);
			// stout << "multiplying from left by t = " << total_tim << std::endl;
			up = this->b_mat_up[total_tim] * up;
			down = this->b_mat_down[total_tim] * down;
			//this->green_up = this->green_up * this->b_mat_up[tim];
			//this->green_down = this->green_down * this->b_mat_down[tim];
		}
		//stout << std::endl;
	};

	this->tempGreen_up.eye();
	this->tempGreen_down.eye();

	int col_begin = (this->p - 1) * this->Ns;
	int col_end = col_begin + this->Ns - 1;
	int row_begin = 0;
	int row_end = row_begin + this->Ns - 1;
	// submatrices starting from the right edge
	arma::subview up = tempGreen_up.submat(row_begin, col_begin, row_end, col_end);
	arma::subview down = tempGreen_down.submat(row_begin, col_begin, row_end, col_end);
	// stout << "\trow sector: " << int(row_begin /this->Ns) << ", col sector t = " << int(col_begin/this->Ns) << std::endl;
	multiplier(this->p - 1, which_time, up, down);
	up *= -1;
	down *= -1;

	for (int sec = 0; sec < this->p - 1; sec++) {
		row_begin = (sec + 1) * this->Ns;
		row_end = row_begin + this->Ns - 1;
		col_begin = (sec)*this->Ns;
		col_end = col_begin + this->Ns - 1;
		// stout << "\trow sector: " << int(row_begin /this->Ns) << ", col sector" << int(col_begin/this->Ns) << std::endl;
		arma::subview up = tempGreen_up.submat(row_begin, col_begin, row_end, col_end);
		arma::subview down = tempGreen_down.submat(row_begin, col_begin, row_end, col_end);

		multiplier(sec, which_time, up, down);
		//this->tempGreen_up.print("after multiplication up");
	}
	this->green_up = this->tempGreen_up.i();
	this->green_down = this->tempGreen_down.i();
}

// -------------------------------------------------------- HELPERS

/// <summary>
///
/// </summary>
/// <param name="im_time_step"></param>
void hubbard::HubbardST::upd_Green_step(int im_time_step, bool forward) {
}

// -------------------------------------------------------- GREEN UPDATERS --------------------------------------------------------

/// <summary>
///
///
/// </summary>
void hubbard::HubbardST::upd_equal_green(int lat_site, double gamma_over_prob_up, double gamma_over_prob_down)
{
	const int lat_site_changed = (this->current_time_slice * this->Ns) + lat_site;
	const int begin = this->current_time_slice * this->Ns;
	const int end = begin + this->Ns - 1;
	arma::mat up = this->green_up.submat(begin, begin, end, end);
	arma::mat down = this->green_down.submat(begin, begin, end, end);

	/*arma::Row row_up = up.row(lat_site);
	arma::Row row_down = down.row(lat_site);
	row_up(lat_site) -= 1;
	row_down(lat_site) -= 1;

	up += up.col(lat_site) * (row_up * gamma_over_prob_up);
	down += down.col(lat_site) * (row_down * gamma_over_prob_down);
	*/

	/*
	this->tempGreen_up = this->green_up;
	this->tempGreen_down = this->green_down;
#pragma omp parallel for num_threads(this->inner_threads)
	for (int a = 0; a < green_size; a++) {
		const double delta = (a == lat_site_changed) ? 1 : 0;
		//const int delta = (b == lat_site) ? 1 : 0;
		for (int b = 0; b < green_size; b++) {
			// SPIN UP
			this->green_up(a,b) -= (delta - tempGreen_up(a,lat_site_changed)) *gamma_over_prob_up * tempGreen_up(lat_site_changed,b);
			// SPIN DOWN
			//gamma_over_prob_down = gamma_down / (1+(1-tempGreen_down(lat_site, lat_site))*gamma_down);
			this->green_down(a,b) -= (delta - tempGreen_down(a,lat_site_changed))*gamma_over_prob_down * tempGreen_down(lat_site_changed,b);
		}
	}
	*/
#pragma omp parallel for num_threads(this->inner_threads)
	for (int a = 0; a < this->Ns; a++) {
		const double delta = (a == lat_site) ? 1 : 0;
		const int i = a + begin;
		//const int delta = (b == lat_site) ? 1 : 0;
		for (int b = 0; b < this->Ns; b++) {
			// SPIN UP
			const int j = a + begin;
			this->green_up(i, j) += (delta - up(a, lat_site)) * gamma_over_prob_up * up(lat_site, b);
			// SPIN DOWN
			//gamma_over_prob_down = gamma_down / (1+(1-tempGreen_down(lat_site, lat_site))*gamma_down);
			this->green_down(i, j) -= (delta - down(a, lat_site)) * gamma_over_prob_down * down(lat_site, b);
		}
	}
}

/// <summary>
/// Update the Green's matrices after going to next Trotter time, remember, the time is taken to be the previous one
/// <param name="which_time">updating to which_time + 1</param>
/// </summary>
void hubbard::HubbardST::upd_next_green(int which_time_green) {
	for (int row = 0; row < this->p; row++) {
		const int row_begin = row * this->Ns;
		const int row_end = row_begin + this->Ns - 1;
		const int time_row = row * this->M_0 + which_time_green;
		for (int col = 0; col < this->p; col++) {
			const int col_begin = col * this->Ns;;
			const int col_end = col_begin + this->Ns - 1;
			const int time_col = col * this->M_0 + which_time_green;

			arma::subview up = this->green_up.submat(row_begin, col_begin, row_end, col_end);
			arma::subview down = this->green_down.submat(row_begin, col_begin, row_end, col_end);

			up = (this->b_mat_up[time_row] * up) * this->b_mat_up[time_col].i();
			down = (this->b_mat_down[time_row] * down) * this->b_mat_down[time_col].i();
		}
	}
}

/// <summary>
///
/// </summary>
/// <param name="which_time"></param>
void hubbard::HubbardST::upd_prev_green(int which_time_green) {
	return;
}

// -------------------------------------------------------- CALCULATORS

/// <summary>
/// A single step for calculating averages inside a loop
/// </summary>
/// <param name="current_elem_i"> Current Green matrix element in averages</param>
void hubbard::HubbardST::av_single_step(int current_elem_i, int sign, bool times)
{
	const mat& g_up = this->green_up;
	const mat& g_down = this->green_down;
	const int elem_i = current_elem_i;
	current_elem_i += this->current_time_slice * this->Ns;

	this->avs->av_sign += sign;
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
	//const double Ek = this->cal_kinetic_en(sign, elem_i);
	//this->avs->av_Ek += Ek;
	//this->avs->av_Ek2 += Ek * Ek;
	// Correlations

	for (int j = 0; j < this->Ns; j++) {
		const int elem_j = j;
		const int j_minus_i_z = this->lattice->get_coordinates(elem_j, 2) - this->lattice->get_coordinates(elem_i, 2);
		const int j_minus_i_y = this->lattice->get_coordinates(elem_j, 1) - this->lattice->get_coordinates(elem_i, 1);
		const int j_minus_i_x = this->lattice->get_coordinates(elem_j, 0) - this->lattice->get_coordinates(elem_i, 0);
		const int z = j_minus_i_z + this->lattice->get_Lz() - 1;
		const int y = j_minus_i_y + this->lattice->get_Ly() - 1;
		const int x = j_minus_i_x + this->lattice->get_Lx() - 1;
		// normal equal - time correlations
		const int current_elem_j = elem_j + this->Ns * this->current_time_slice;
		this->avs->av_M2z_corr[x][y][z] += this->cal_mz2_corr(sign, current_elem_i, current_elem_j, g_up, g_down);
		this->avs->av_occupation_corr[x][y][z] += this->cal_occupation_corr(sign, current_elem_i, current_elem_j, g_up, g_down);
		this->avs->av_ch2_corr[x][y][z] += this->cal_ch_correlation(sign, current_elem_i, current_elem_j, g_up, g_down) / (this->Ns * 2.0);
	}
}

// ---------------------------------------------------------------------------------------------------------------- HEAT BATH ----------------------------------------------------------------------------------------------------------------

/// <summary>
/// </summary>
/// <param name="lat_site">the candidate lattice site</param>
/// <returns>sign of probability</returns>
int hubbard::HubbardST::heat_bath_single_step(int lat_site)
{
	const int green_elem = this->current_time_slice * this->Ns + lat_site;							// we need to take the element from the different site of Green
	const auto [gamma_up, gamma_down] = this->cal_gamma(lat_site);									// first up then down
	const auto [proba_up, proba_down] = this->cal_proba(green_elem, gamma_up, gamma_down);			// take the probabilities
	double proba = proba_up * proba_down;															// Metropolis probability
	//double multiplier = exp(2*hsFields[current_time][lat_site]*lambda);									// https://iopscience.iop.org/article/10.1088/1742-6596/1483/1/012002/pdf
	proba = proba / (1.0 + proba);																	// heat-bath probability
	const int sign = proba >= 0 ? 1 : -1;
	//auto r = this->ran.randomReal_uni(0,1);
	//stout << "random-> " << r << ((r <= proba) ? " <= " : " > " ) << proba << "<-proba\n";
	//if (this->ran.bernoulli(proba)) {
	if (this->ran.randomReal_uni(0, 1) <= sign * proba) {
		const double delta_up = gamma_up + 1;
		const double delta_down = gamma_down + 1;
		//stout << "\tI am in, updating\n";
		this->upd_int_exp(lat_site, delta_up, delta_down);
		//this->cal_B_mat(this->current_time);
		this->upd_B_mat(lat_site, delta_up, delta_down);											// update the B matrices
		this->upd_equal_green(lat_site, gamma_up / proba_up, gamma_down / proba_down);				// update Greens via Dyson
		this->hsFields(this->current_time, lat_site) *= -1;
	}
	return sign;
}

/// <summary>
/// Drive the system to equilibrium with heat bath
/// </summary>
/// <param name="mcSteps">Number of Monte Carlo steps</param>
/// <param name="conf">If or if not to save configurations</param>
/// <param name="quiet">If should be quiet</param>
void hubbard::HubbardST::heat_bath_eq(int mcSteps, bool conf, bool quiet, bool save_greens)
{
	auto start = std::chrono::high_resolution_clock::now();
	if (mcSteps != 1) {
		stout << "\t\t----> STARTING RELAXING FOR : " + this->info << std::endl;
		this->neg_num = 0;																				// counter of negative signs
		this->pos_num = 0;																				// counter of positive signs
	}
	// Progress bar
	auto progress = pBar();
	const double percentage = 20;
	const int percentage_steps = static_cast<int>(percentage * mcSteps / 100.0);

	// function
	int (HubbardST:: * ptfptr)(int);																	// pointer to a single step function depending on whether we do configs or not
	ptfptr = &HubbardST::heat_bath_single_step;															// pointer to non-saving configs

	for (int step = 0; step < mcSteps; step++) {
		// Monte Carlo steps
		//stout << "Starting sweep number : " << step << std::endl;
		this->cal_green_mat(0);
		//green_up.print("upgreen");
		for (int time_in_slice = 0; time_in_slice < this->M_0; time_in_slice++) {
			this->current_time_in_silce = time_in_slice;
			if (this->current_time_in_silce != 0) this->upd_next_green(this->current_time_in_silce - 1);
			for (int time_slice = 0; time_slice < this->p; time_slice++) {
				this->current_time_slice = time_slice;
				this->current_time = this->current_time_slice * this->M_0 + this->current_time_in_silce;

				for (int j = 0; j < this->Ns; j++) {
					const int sign = (this->*ptfptr)(j);													// get current sign and make single step
					sign > 0 ? this->pos_num++ : this->neg_num++;											// increase sign
					// compare Green's functions
					// this->compare_green_direct(this->current_time, 1e-6, false);
				}
			}
		}
		if (mcSteps != 1 && step % percentage_steps == 0) {
			stout << "\t\t\t\t-> time: " << tim_s(start) << " -> RELAXATION PROGRESS for " << this->info << " : \n";
			progress.print();
			stout << std::endl;
			progress.update(percentage);
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
void hubbard::HubbardST::heat_bath_av(int corr_time, int avNum, bool quiet, bool times)
{
	auto start = std::chrono::high_resolution_clock::now();
#pragma omp critical
	stout << "\t\t----> STARTING AVERAGING FOR : " + this->info << std::endl;
	this->neg_num = 0L;																				// counter of negative signs
	this->pos_num = 0L;																				// counter of positive signs

	// Progress bar
	auto progress = pBar();
	const double percentage = 25;
	const int percentage_steps = static_cast<int>(percentage * avNum / 100.0);

	for (int step = 0; step < avNum; step++) {
		// Monte Carlo steps
		//stout << "Starting sweep number : " << step << std::endl;
		this->cal_green_mat(0);

		for (int time_in_slice = 0; time_in_slice < this->M_0; time_in_slice++) {
			this->current_time_in_silce = time_in_slice;
			for (int time_slice = 0; time_slice < this->p; time_slice++) {
				this->current_time_slice = time_slice;
				this->current_time = this->current_time_slice * this->M_0 + this->current_time_in_silce;

				for (int j = 0; j < this->Ns; j++) {
					const int sign = this->heat_bath_single_step(j);									// get current sign and make single step
					sign > 0 ? this->pos_num++ : this->neg_num++;										// increase sign
					// compare Green's functions
					// this->compare_green_direct(this->current_time, 1e-6, false);
					this->av_single_step(j, sign, false);					// collect all averages
				}
			}
			this->upd_next_green(time_in_slice);
		}
		for (int corr = 1; corr < corr_time; corr++)
			this->heat_bath_eq(1, false, true);

		if (step % percentage_steps == 0) {
			stout << "\t\t\t\t-> time: " << tim_s(start) << " -> RELAXATION PROGRESS for " << this->info << " : \n";
			progress.print();
			stout << std::endl;
			progress.update(percentage);
		}
	}
	// After
	this->av_normalise(avNum, this->M, times);
}

// ---------------------------------------------------------------------------------------------------------------- PUBLIC CALCULATORS ----------------------------------------------------------------------------------------------------------------