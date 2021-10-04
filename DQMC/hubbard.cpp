#include "include/hubbard.h"
// -------------------------------------------------------- HUBBARD MODEL -------------------------------------------------------- */
// -------------------------------------------------------- HELPERS


/// <summary>
/// Normalise all the averages taken during simulation
/// </summary>
/// <param name="avNum">Number of avs taken</param>
/// <param name="times">If the non-equal time properties were calculated</param>
void hubbard::HubbardModel::av_normalise(int avNum, int timesNum, bool times)
{
	const double normalization = static_cast<double>(avNum * timesNum * this->Ns);						// average points taken
	this->avs->av_sign /= normalization;																		// average sign is needed
	this->avs->sd_sign = sqrt((1.0 - (this->avs->av_sign * this->avs->av_sign)) / normalization);
	const double normalisation_sign = normalization * this->avs->av_sign;										// we divide by average sign actually
	// with minus
	this->avs->av_gr_down /= normalisation_sign / this->Ns;
	this->avs->av_gr_up /= normalisation_sign / this->Ns;

	this->avs->av_occupation /= normalisation_sign;
	this->avs->sd_occupation = sqrt((this->avs->sd_occupation / normalisation_sign - this->avs->av_occupation * this->avs->av_occupation) / normalisation_sign);

	this->avs->av_M2z /= normalisation_sign;
	this->avs->sd_M2z = sqrt((this->avs->sd_M2z / normalisation_sign - this->avs->av_M2z * this->avs->av_M2z) / normalisation_sign);
	this->avs->av_M2x /= normalisation_sign;
	this->avs->sd_M2x = sqrt((this->avs->sd_M2x / normalisation_sign - this->avs->av_M2x * this->avs->av_M2x) / normalisation_sign);
	// Ek
	this->avs->av_Ek /= normalisation_sign;
	this->avs->sd_Ek = sqrt((this->avs->av_Ek2 / normalisation_sign - this->avs->av_Ek * this->avs->av_Ek) / normalisation_sign);
	// correlations
	for (int x = -this->lattice->get_Lx() + 1; x < this->lattice->get_Lx(); x++) {
		for (int y = -this->lattice->get_Ly() + 1; y < this->lattice->get_Ly(); y++) {
			for (int z = -this->lattice->get_Lz() + 1; z < this->lattice->get_Lz(); z++) {
				int x_pos = x + this->lattice->get_Lx() - 1;
				int y_pos = y + this->lattice->get_Ly() - 1;
				int z_pos = z + this->lattice->get_Lz() - 1;
				this->avs->av_M2z_corr[x_pos][y_pos][z_pos] /= normalisation_sign;
				this->avs->av_ch2_corr[x_pos][y_pos][z_pos] /= normalisation_sign;
				this->avs->av_occupation_corr[x_pos][y_pos][z_pos] = this->Ns * this->avs->av_occupation_corr[x_pos][y_pos][z_pos] / normalisation_sign;
				//if (times) {
					//for (int l = 0; l < this->M; l++) {
						//this->avs->av_green_down[x_pos][y_pos][z_pos][l] /= normalisation_sign/this->M;
						//this->avs->av_green_up[x_pos][y_pos][z_pos][l] /= normalisation_sign / this->M;
						//this->avs->av_M2z_corr_uneqTime[x][y][z][l] /= normalisation_sign / this->M_0;
						//this->avs->av_Charge2_corr_uneqTime[x][y][z][l] /= normalisation_sign / this->M_0;
				//	}
			}
		}
	}
}

// -------------------------------------------------------- SETTERS
/// <summary>
/// Setting the Hubbard - Stratonovich fields
/// </summary>
void hubbard::HubbardModel::set_hs()
{
	for (int l = 0; l < this->M; l++) {
		for (int i = 0; i < this->Ns; i++) {
			this->hsFields[l][i] = this->ran.bernoulli(0.5) ? -1 : 1;		// set the hs fields to uniform -1 or 1
		}
	}
}

/// <summary>
/// Sets the directories for saving configurations of Hubbard - Stratonovich fields. It adds /negative/ and /positive/ to dir
/// </summary>
/// <param name="dir">directory to be used for configurations</param>
void hubbard::HubbardModel::setConfDir(std::string dir)
{
	this->neg_dir = dir + std::string(kPSep) + "negative";
	this->pos_dir = dir + std::string(kPSep) + "positive";
	// create directories
	std::filesystem::create_directories(this->neg_dir);
	std::filesystem::create_directories(this->pos_dir);
	// add a separator
	this->neg_dir += std::string(kPSep);
	this->pos_dir += std::string(kPSep);
	// for .log files
	std::ofstream fileN, fileP;																	// files for saving the configurations
	this->neg_log = this->neg_dir.substr(0, \
		this->neg_dir.length() - 9) + "neg_log," + info + ".dat";								// for storing the labels of negative files in csv for ML
	this->pos_log = this->pos_dir.substr(0, \
		this->pos_dir.length() - 9) + "pos_log," + info + ".dat";								// for storing the labels of positive files in csv for ML
	fileN.open(this->neg_log);
	fileP.open(this->pos_log);
	fileN.close();																				// close just to create file neg
	fileP.close();																				// close just to create file pos
}


// -------------------------------------------------------- HELPERS --------------------------------------------------------

/// <summary>
/// Function to calculate the change in the potential exponential
/// Attractive case needs to be done
/// </summary>
/// <param name="lat_site">lattice site on which the change has been made</param>
/// <returns>A tuple for gammas for two spin channels, 0 is spin up, 1 is spin down</returns>
std::tuple<double, double> hubbard::HubbardModel::cal_gamma(int lat_site) const
{
	std::tuple< double, double> tmp(0, 0);								// here we will save the gammas
	if (this->U > 0) {
		// Repulsive case
		if (this->hsFields[this->current_time][lat_site] > 0)
			tmp = std::make_tuple(this->gammaExp[1] - 1.0, this->gammaExp[0] - 1.0);
		else
			tmp = std::make_tuple(this->gammaExp[0] - 1.0, this->gammaExp[1] - 1.0);
	}
	else {
		/* Attractive case */
	}
	return tmp;
}

/// <summary>
/// Return probabilities of spin flip for both spin channels
/// </summary>
/// <param name="lat_stie">flipping candidate site</param>
/// <param name="gamma_up">the changing parameter for spin up</param>
/// <param name="gamma_down">the changing parameter for spin down</param>
/// <returns>A tuple for probabilities on both spin channels, remember, 0 is spin up, 1 is spin down</returns>
std::tuple<double, double> hubbard::HubbardModel::cal_proba(int lat_site, double gamma_up, double gamma_down) const
{
	//double tmp_up = exp(-2*this->lambda * this->hsFields[this->current_time][lat_site]) - 1.0;
	//double tmp_down = exp(2*this->lambda * this->hsFields[this->current_time][lat_site]) - 1.0;
	//stout << "up: " << tmp_up << "\tdown: " << tmp_down << std::endl;
	//stout << "gammaUp: " << gamma_up << "\tgamma_down: " << gamma_down << std::endl << std::endl << std::endl;
	// SPIN UP
	const double p_up = 1.0 + (gamma_up) * (1.0 - this->green_up(lat_site, lat_site));
	// SPIN DOWN
	const double p_down = 1.0 + (gamma_down) * (1.0 - this->green_down(lat_site, lat_site));

	return std::make_tuple(p_up, p_down);
}

// -------------------------------------------------------- UPDATERS --------------------------------------------------------

/// <summary>
/// Update the interaction matrix for current spin whenever the given lattice site HS field is changed
/// Only for testing purpose
/// </summary>
/// <param name="lat_site">the site of changed HS field</param>
/// <param name="delta_sigma">difference between changed and not</param>
/// <param name="sigma">spin channel</param>
void hubbard::HubbardModel::upd_int_exp(int lat_site, double delta_up, double delta_down)
{
	this->int_exp_up(lat_site, this->current_time) *= delta_up;
	this->int_exp_down(lat_site, this->current_time) *= delta_down;
}

/// <summary>
/// After accepting spin change update the B matrix by multiplying it by diagonal element ( the delta )
/// </summary>
/// <param name="lat_site"></param>
/// <param name="delta_up"></param>
/// <param name="delta_down"></param>
void hubbard::HubbardModel::upd_B_mat(int lat_site, double delta_up, double delta_down) {
		this->b_mat_up[this->current_time].col(lat_site) *= delta_up;				// spin up
		this->b_mat_down[this->current_time].col(lat_site) *= delta_down;			// spin down
		//this->b_mat_up[this->current_time](j, lat_site) *= delta_up;						// spin up
		//this->b_mat_down[this->current_time](j, lat_site) *= delta_down;					// spin down
}

// -------------------------------------------------------- GETTERS

// -------------------------------------------------------- CALCULATORS

/// <summary>
/// Function to calculate the hopping matrix exponential (with nn for now)
/// </summary>
void hubbard::HubbardModel::cal_hopping_exp()
{
	bool checkerboard = true;
	const int Lx = this->lattice->get_Lx();
	const int Ly = this->lattice->get_Ly();

	// USE CHECKERBOARD
	const int dim = this->lattice->get_Dim();
	if (checkerboard) {
		arma::mat Kx_a, Kx_b, Ky_a, Ky_b, Kz_a, Kz_b;
		Kx_a.zeros(this->Ns, this->Ns);
		Kx_b = Kx_a;
		if (dim >= 2) {
			// 2D
			Ky_a = Kx_a;
			Ky_b = Ky_a;
			if (dim == 3) {
				// 3D
				Kz_a = Kx_a;
				Kz_b = Kz_a;
			}
		}
		// set elements
		for (int i = 0; i < this->Ns; i++) {
			const int n_of_neigh = this->lattice->get_nn_number(i);												// take number of nn at given site
			for (int j = 0; j < n_of_neigh; j++) {
				const int where_neighbor = this->lattice->get_nn(i, j);											// get given nn
				const int y = i / Lx;
				const int x = i - y * Lx;
				const int y_nei = where_neighbor / Ly;
				const int x_nei = where_neighbor - y_nei * Lx;
				if (y_nei == y) {
					// even rows
					if (i % 2 == 0) {
						if (x_nei == (x + 1)% Lx) {
							Kx_a(i, where_neighbor) = 1;
						}
						else {
							Kx_b(i, where_neighbor) = 1;
						}
					}
					// odd rows
					else {
						if (x_nei == (x + 1) % Lx) {
							Kx_b(i, where_neighbor) = 1;
						}
						else {
							Kx_a(i, where_neighbor) = 1;
						}
					}
				}
				else {
					// ky
					if (where_neighbor % 2 == 0) {
						Ky_a(i, where_neighbor) = 1;
						Ky_a(where_neighbor, i) = 1;
					}
					else {
						Ky_b(i, where_neighbor) = 1;
						Ky_b(where_neighbor, i) = 1;
					}
				}
			}
		}
		//Kx_a.print("Kx a:");
		//Kx_b.print("Kx b:");
		//Ky_a.print("Ky a:");
		//Ky_b.print("Ky b:");
		//this->hopping_exp = Kx_a + Kx_b + Ky_a + Ky_b;
		//this->hopping_exp.print("HOPPING MATRIX:");

		//arma::mat tmp_exp = arma::expmat(this->hopping_exp);
		//tmp_exp.print("NORMALLY CALCULATED EXPONENT");

		arma::mat one = arma::eye(this->Ns, this->Ns); 
		one *= cosh(this->dtau * t[0]);
		const double sinus = sinh(this->dtau * this->t[0]);

		Kx_a = (Kx_a  * sinus + one);
		Kx_b = (Kx_b * sinus + one);
		Ky_a = (Ky_a  * sinus + one);
		Ky_b = (Ky_b * sinus + one);
		this->hopping_exp = Ky_a * Kx_a * Ky_b * Kx_b;
		//this->hopping_exp.print("BETTER CALCULATED EXP");
		return;
	}
	/*else {
#pragma omp parallel for num_threads(this->inner_threads)
	for (int i = 0; i < this->Ns; i++) {
		//this->hopping_exp(i, i) = this->dtau * this->mu;													// diagonal elements
		const int n_of_neigh = this->lattice->get_nn_number(i);												// take number of nn at given site
		for (int j = 0; j < n_of_neigh; j++) {
			const int where_neighbor = this->lattice->get_nn(i, j);											// get given nn
			this->hopping_exp(i, where_neighbor) = this->dtau * this->t[i];									// assign non-diagonal elements
		}
	}
	//this->hopping_exp.print("hopping before exponentiation");
	//arma::vec eigval;
	//arma::mat eigvec;
	//arma::eig_sym(eigval, eigvec, this->hopping_exp);
	//stout << "eigenvalues:\n" << eigval.t() << std::endl;
	//arma::mat jordan = eigvec.i() * this->hopping_exp * eigvec;
	//jordan = arma::expmat_sym(jordan);
	//this->hopping_exp = eigvec * this->hopping_exp * eigvec.i();

#pragma omp critical
	stout << "condition number of hopping matrix is : " << arma::cond(this->hopping_exp) << std::endl;
	this->hopping_exp = arma::expmat_sym(this->hopping_exp);												// take the exponential
	//this->hopping_exp.print("hopping after exponentiation");

	}
*/
}

/// <summary>
/// Function to calculate the interaction exponential at all times, each column represents the given Trotter time
/// </summary>
void hubbard::HubbardModel::cal_int_exp() {
	if (this->U > 0) {
		// Repulsive case 
		const double exp_plus = exp(this->lambda + this->dtau * (this->mu));				// plus exponent for faster computation
		const double exp_minus = exp(-this->lambda + this->dtau * (this->mu));			// minus exponent for faster computation
//#pragma omp parallel for collapse(2) num_threads(this->inner_threads)
		for (int l = 0; l < this->M; l++) {
			// Trotter times 
			for (int i = 0; i < this->Ns; i++) {
				// Lattice sites 
				if (hsFields[l][i] > 0) {
					this->int_exp_up(i, l) = exp_plus;			// diagonal up spin channel
					this->int_exp_down(i, l) = exp_minus;		// diagonal down spin channel
				}
				else {
					this->int_exp_up(i, l) = exp_minus;			// diagonal up spin channel
					this->int_exp_down(i, l) = exp_plus;		// diagonal down spin channel
				}
			}
		}
	}
	else if (U < 0) {
		const double exp_plus = exp(this->lambda + this->dtau * this->mu);				// plus exponent for faster computation
		const double exp_minus = exp(-this->lambda + this->dtau * this->mu);			// minus exponent for faster computation
		// Attractive case
		for (int l = 0; l < M; l++) {
			// Trotter times 
			for (int i = 0; i < Ns; i++) {
				// Lattice sites 
				this->int_exp_down(i, l) = (this->hsFields[l][i] > 0) ? exp_plus : exp_minus;
				this->int_exp_up(i, l) = this->int_exp_down(i, l);
			}
		}
	}
	else {
		this->int_exp_down = arma::eye(this->Ns, this->Ns);
		this->int_exp_up = arma::eye(this->Ns, this->Ns);
	}
}

/// <summary>
/// Function to calculate all B exponents for a given model. Those are used for the Gibbs weights
/// </summary>
void hubbard::HubbardModel::cal_B_mat() {
//#pragma omp parallel for num_threads(this->inner_threads)
	for (int l = 0; l < this->M; l++) {
		// Trotter times 
		this->b_mat_down[l] = arma::diagmat(this->int_exp_down.col(l)) * this->hopping_exp;
		this->b_mat_up[l] = arma::diagmat(this->int_exp_up.col(l)) * this->hopping_exp;
		//this->b_mat_down[l] = this->hopping_exp * arma::diagmat(this->int_exp_down.col(l));
		//this->b_mat_up[l] = this->hopping_exp * arma::diagmat(this->int_exp_up.col(l));
	}
	//b_mat_down[0].print("B_mat_down in t = 0");
}

/// <summary>
/// Function to calculate all B exponents for a given model at a given time. Those are used for the Gibbs weights
/// </summary>
void hubbard::HubbardModel::cal_B_mat(int which_time)
{
	this->b_mat_down[which_time] = this->hopping_exp * arma::diagmat(this->int_exp_down.col(which_time));
	this->b_mat_up[which_time] = this->hopping_exp * arma::diagmat(this->int_exp_up.col(which_time));
}



// -------------------------------------------------------- PRINTERS

/// <summary>
///
/// </summary>
/// <param name="output"></param>
/// <param name="which_time_caused"></param>
/// <param name="which_site_caused"></param>
/// <param name="this_site_spin"></param>
/// <param name="separator"></param>
void hubbard::HubbardModel::print_hs_fields(std::ostream& output, int which_time_caused, int which_site_caused, short this_site_spin, std::string separator) const
{
	for (int i = 0; i < this->M; i++) {
		for (int j = 0; j < this->Ns; j++) {
			if (j == which_site_caused && i == which_time_caused)
			{
				output << (this_site_spin > 0 ? 1 : 0) << separator;
			}
			else
			{
				output << ((this->hsFields[i][j] == 1) ? 0.75 : 0.25) << separator;
			}
		}
		output << "\n";
	}
}

// -------------------------------------------------------- EQUAL TIME AVERAGES

double hubbard::HubbardModel::cal_kinetic_en(int sign, int current_elem_i) const
{
	const int nei_num = this->lattice->get_nn_number(current_elem_i);
	double Ek = 0;
	for (int nei = 0; nei < nei_num; nei++)
	{
		const int where_neighbor = this->lattice->get_nn(current_elem_i, nei);
		Ek += this->green_down(current_elem_i, where_neighbor);
		Ek += this->green_down(where_neighbor, current_elem_i);
		Ek += this->green_up(current_elem_i, where_neighbor);
		Ek += this->green_up(where_neighbor, current_elem_i);
	}
	return sign * this->t[current_elem_i] * Ek;
}

double hubbard::HubbardModel::cal_occupation(int sign, int current_elem_i) const
{
	return (sign * (1.0 - this->green_down(current_elem_i, current_elem_i)) + sign * (1.0 - this->green_up(current_elem_i, current_elem_i)));
}

double hubbard::HubbardModel::cal_occupation_corr(int sign, int current_elem_i, int current_elem_j) const
{
	return sign * ((this->green_down(current_elem_j, current_elem_i) + this->green_up(current_elem_j, current_elem_i)));
}

double hubbard::HubbardModel::cal_mz2(int sign, int current_elem_i) const
{
	return sign * (((1.0 - this->green_up(current_elem_i, current_elem_i)) * (1.0 - this->green_up(current_elem_i, current_elem_i)))
		+ ((1.0 - this->green_up(current_elem_i, current_elem_i)) * (this->green_up(current_elem_i, current_elem_i)))
		- ((1.0 - this->green_up(current_elem_i, current_elem_i)) * (1.0 - this->green_down(current_elem_i, current_elem_i)))
		- ((1.0 - this->green_down(current_elem_i, current_elem_i)) * (1.0 - this->green_up(current_elem_i, current_elem_i)))
		+ ((1.0 - this->green_down(current_elem_i, current_elem_i)) * (1.0 - this->green_down(current_elem_i, current_elem_i)))
		+ ((1.0 - this->green_down(current_elem_i, current_elem_i)) * (this->green_down(current_elem_i, current_elem_i))));
}

double hubbard::HubbardModel::cal_mz2_corr(int sign, int current_elem_i, int current_elem_j) const
{
	double delta_ij = 0.0L;
	if (current_elem_i == current_elem_j) {
		delta_ij = 1.0L;
	}
	//this->green_down.print("TEST");
	return sign * (((1.0L - this->green_up(current_elem_i, current_elem_i)) * (1.0L - this->green_up(current_elem_j, current_elem_j)))
		+ ((delta_ij - this->green_up(current_elem_j, current_elem_i)) * (this->green_up(current_elem_i, current_elem_j)))
		- ((1.0L - this->green_up(current_elem_i, current_elem_i)) * (1.0L - this->green_down(current_elem_j, current_elem_j)))
		- ((1.0L - this->green_down(current_elem_i, current_elem_i)) * (1.0L - this->green_up(current_elem_j, current_elem_j)))
		+ ((1.0L - this->green_down(current_elem_i, current_elem_i)) * (1.0L - this->green_down(current_elem_j, current_elem_j)))
		+ ((delta_ij - this->green_down(current_elem_j, current_elem_i)) * (this->green_down(current_elem_i, current_elem_j))));
}

double hubbard::HubbardModel::cal_my2(int sign, int current_elem_i) const
{
	return 0;
}

double hubbard::HubbardModel::cal_mx2(int sign, int current_elem_i) const
{
	return sign * (1.0 - this->green_up(current_elem_i, current_elem_i)) * (this->green_down(current_elem_i, current_elem_i))
		+ sign * (1.0 - this->green_down(current_elem_i, current_elem_i)) * (this->green_up(current_elem_i, current_elem_i));
}

double hubbard::HubbardModel::cal_ch_correlation(int sign, int current_elem_i, int current_elem_j) const
{
	double delta_ij = 0.0L;
	if (current_elem_i == current_elem_j) {
		delta_ij = 1.0L;
	}
	return sign * (((1 - this->green_up(current_elem_i, current_elem_i)) * (1 - this->green_up(current_elem_j, current_elem_j))					//sigma = sigma' = up
		+ (1 - this->green_down(current_elem_i, current_elem_i)) * (1 - this->green_down(current_elem_j, current_elem_j))				//sigma = sigma' = down
		+ (1 - this->green_down(current_elem_i, current_elem_i)) * (1 - this->green_up(current_elem_j, current_elem_j))					//sigma = down, sigma' = up
		+ (1 - this->green_up(current_elem_i, current_elem_i)) * (1 - this->green_down(current_elem_j, current_elem_j))					//sigma = up, sigma' = down
		+ ((delta_ij - this->green_up(current_elem_j, current_elem_i)) * this->green_up(current_elem_i, current_elem_j))				//sigma = sigma' = up
		+ ((delta_ij - this->green_down(current_elem_j, current_elem_i)) * this->green_down(current_elem_i, current_elem_j))));			//sigma = sigma' = down
}


// ---------------------------------------------------------------------------------------------------------------- PUBLIC CALCULATORS ----------------------------------------------------------------------------------------------------------------

/// <summary>
/// Equilivrate the simulation
/// </summary>
/// <param name="algorithm">type of equilibration algorithm</param>
/// <param name="mcSteps">Number of Monte Carlo steps</param>
/// <param name="conf">Shall print configurations?</param>
/// <param name="quiet">Shall be quiet?</param>
void hubbard::HubbardModel::relaxation(impDef::algMC algorithm, int mcSteps, bool conf, bool quiet)
{
	auto start = std::chrono::high_resolution_clock::now();											// starting timer for averages

	switch (algorithm)
	{
	case impDef::algMC::heat_bath:
		this->heat_bath_eq(mcSteps, conf, quiet);
		break;
	default:
		std::cout << "Didn't choose the algorithm type\n";
		exit(-1);
		break;
	}

	auto stop = std::chrono::high_resolution_clock::now();											// finishing timer for relaxation
	if (mcSteps != 1) {
#pragma omp critical
		stout << "For: " << this->get_info() << "->\n\t\tRelaxation Time taken: " << \
			(std::chrono::duration_cast<std::chrono::seconds>(stop - start)).count() << \
			" seconds. With average sign = " << \
			1.0 * (this->pos_num - this->neg_num) / (this->pos_num + this->neg_num) << std::endl;
	}
}

/// <summary>
/// Collect the averages from the simulation
/// </summary>
/// <param name="algorithm">type of equilibration algorithm</param>
/// <param name="corr_time">how many times to wait for correlations breakout</param>
/// <param name="avNum">number of averages to take</param>
/// <param name="bootStraps">Number of bootstraps - NOT IMPLEMENTED </param>
/// <param name="quiet">Shall be quiet?</param>
/// <param name="times"></param>
void hubbard::HubbardModel::average(impDef::algMC algorithm, int corr_time, int avNum, int bootStraps, bool quiet, int times)
{
	auto start = std::chrono::high_resolution_clock::now();											// starting timer for averages
	switch (algorithm)
	{
	case impDef::algMC::heat_bath:
		this->heat_bath_av(corr_time, avNum, quiet, times);
		break;
	default:
		std::cout << "Didn't choose the algorithm type\n";
		exit(-1);
		break;
	}
	auto stop = std::chrono::high_resolution_clock::now();											// finishing timer for relaxation
#pragma omp critical
	stout << "For: " << this->get_info() << "->\n\t\tAverages time taken: " << \
		(std::chrono::duration_cast<std::chrono::seconds>(stop - start)).count() << \
		" seconds. With average sign = " << \
		this->avs->av_sign << "\n\t\t->or with other measure : " << (this->pos_num - this->neg_num) / (1.0*static_cast<long long>(this->pos_num + this->neg_num)) \
		<<  std::endl;
}








