#include "include/hubbard.h"
// -------------------------------------------------------- HUBBARD MODEL -------------------------------------------------------- */
// -------------------------------------------------------- HELPERS
/// <summary>
/// A single step for calculating averages inside a loop
/// </summary>
/// <param name="current_elem_i"> Current Green matrix element in averages</param>
void hubbard::HubbardModel::av_single_step(int current_elem_i, int sign)
{
	this->avs->av_sign += sign;
	// m_z
	const double m_z2 = this->cal_mz2(sign, current_elem_i);
	this->avs->av_M2z += m_z2;
	this->avs->sd_M2z += m_z2 * m_z2;
	// m_x
	const double m_x2 = this->cal_mx2(sign, current_elem_i);
	this->avs->av_M2x += m_x2;
	this->avs->sd_M2x += m_x2 * m_x2;
	// occupation
	const double occ = this->cal_occupation(sign, current_elem_i);
	this->avs->av_occupation += occ;
	this->avs->sd_occupation += occ * occ;
	// kinetic energy
	const double Ek = this->cal_kinetic_en(sign, current_elem_i);
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
		this->avs->av_M2z_corr[x][y][z] += this->cal_mz2_corr(sign, current_elem_i, current_elem_j);
		this->avs->av_occupation_corr[x][y][z] += this->cal_occupation_corr(sign, current_elem_i, current_elem_j);
		this->avs->av_ch2_corr[x][y][z] += this->cal_ch_correlation(sign, current_elem_i, current_elem_j) / (this->Ns * 2.0);
	}
}

/// <summary>
/// Normalise all the averages taken during simulation
/// </summary>
/// <param name="avNum">Number of avs taken</param>
/// <param name="times">If the non-equal time properties were calculated</param>
void hubbard::HubbardModel::av_normalise(int avNum, bool times)
{
	const double normalization = static_cast<double>(avNum * this->M * this->Ns);								// average points taken
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
			this->hsFields[l][i] = this->ran.randomReal_uni() < 0.5 ? -1 : 1;		// set the hs fields to uniform -1 or 1
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

// -------------------------------------------------------- GETTERS

// -------------------------------------------------------- CALCULATORS

/// <summary>
/// Function to calculate the hopping matrix exponential (with nn for now)
/// </summary>
void hubbard::HubbardModel::cal_hopping_exp()
{
#pragma omp parallel for num_threads(this->inner_threads)
	for (int i = 0; i < this->Ns; i++) {
		this->hopping_exp(i, i) = this->dtau * this->mu;													// diagonal elements
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

/// <summary>
/// Function to calculate the interaction exponential at all times, each column represents the given Trotter time
/// </summary>
void hubbard::HubbardModel::cal_int_exp() {
	if (this->U > 0) {
		const double exp_plus = exp(this->lambda);				// plus exponent for faster computation
		const double exp_minus = exp(-this->lambda);			// minus exponent for faster computation
		// Repulsive case 
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
		/* Attractive case */
		for (int l = 0; l < M; l++) {
			/* Trotter times */
			for (int i = 0; i < Ns; i++) {
				/* Lattice sites */
				this->int_exp_down(i, l) = exp(-(1.0 / dtau) * lambda * hsFields[l][i] - (this->mu + (abs(this->U) / 2.0)) + 0.5 * (lambda * hsFields[l][i] + abs(this->U / 2) * this->dtau));
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
#pragma omp parallel for num_threads(this->inner_threads)
	for (int l = 0; l < this->M; l++) {
		// Trotter times 
		//this->b_mat_down[l] = arma::diagmat(this->int_exp_down.col(l)) * this->hopping_exp;
		//this->b_mat_up[l] = arma::diagmat(this->int_exp_up.col(l)) * this->hopping_exp;
		this->b_mat_down[l] = this->hopping_exp * arma::diagmat(this->int_exp_down.col(l));
		this->b_mat_up[l] = this->hopping_exp * arma::diagmat(this->int_exp_up.col(l));
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

// -------------------------------------------------------- HUBBARD MODEL WITH QR DECOMPOSITION --------------------------------------------------------

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
	this->gammaExp = { std::exp(2 * this->lambda), std::exp(-2 * this->lambda) };			// 0 -> sigma * hsfield = -1, 1 -> sigma * hsfield = 1

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
	this->b_mat_up = std::vector<arma::mat>(this->M, arma::mat(this->Ns, this->Ns, arma::fill::zeros));
	this->b_mat_down = std::vector<arma::mat>(this->M, arma::mat(this->Ns, this->Ns, arma::fill::zeros));
	this->b_up_condensed = std::vector<arma::mat>(this->p, arma::mat(this->Ns, this->Ns, arma::fill::zeros));
	this->b_down_condensed = std::vector<arma::mat>(this->p, arma::mat(this->Ns, this->Ns, arma::fill::zeros));

	// all times hs fields for real spin up and down
	this->hsFields = std::vector(this->M, std::vector<short>(this->Ns, 1));

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

// -------------------------------------------------------- OTHER METHODS

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

/// <summary>
/// Calculate Green with QR decomposition using LOH : doi:10.1016/j.laa.2010.06.023
/// For more look into :
/// "Advancing Large Scale Many-Body QMC Simulations on GPU Accelerated Multicore Systems"
/// In order to do that the M_0 and p variables will be used to divide the multiplication into smaller chunks of matrices
/// </summary>
/// <param name="which_time"></param>
void hubbard::HubbardQR::cal_green_mat(int which_time) {
	//stout << "STARTING CALCULATING GREEN FOR : " << which_time << std::endl;
	// if we can use precalculated version we do!
	if (which_time % this->M_0 == 0) {
		// find the sector to start
		int sector = static_cast<int>(which_time / (1.0 * this->M_0));
		//stout << sector << std::endl;
		if (!arma::qr(Q_up, R_up, P_up, this->b_up_condensed[sector], "matrix")) throw "decomposition failed\n";
		if (!arma::qr(Q_down, R_down, P_down, this->b_down_condensed[sector], "matrix")) throw "decomposition failed\n";
		D_down = diagvec(R_down);
		D_up = diagvec(R_up);
		T_down = (diagmat(D_down).i()) * R_down * (P_down.t());
		T_up = (diagmat(D_up).i()) * R_up * (P_up.t());

		for (int i = 1; i < this->p; i++) {
			sector++;
			if(sector == this->p) sector = 0;
			//stout << sector << std::endl;

			this->green_up = (this->b_up_condensed[sector] * Q_up) * diagmat(D_up);				// multiply by the former ones
			this->green_down = (this->b_down_condensed[sector] * Q_down) * diagmat(D_down);		// multiply by the former ones

			if (!arma::qr(Q_up, R_up, P_up, this->green_up)) throw "decomposition failed\n";
			if (!arma::qr(Q_down, R_down, P_down, this->green_down)) throw "decomposition failed\n";

			D_up = diagvec(R_up);
			D_down = diagvec(R_down);

			T_up = ((diagmat(D_up).i()) * R_up) * P_up.t() * T_up;
			T_down = ((diagmat(D_down).i()) * R_down) * P_down.t() * T_down;
		}
	}
	else {
		stout << "\t\t\tCalculating Green not in M_0 cycle\n";
		auto multiplier = [&](auto& tim) mutable {
			this->green_up = this->b_mat_up[tim];
			this->green_down = this->b_mat_down[tim];
			//stout << tim << std::endl;
			for (int j = 1; j < this->M_0; j++) {
				tim++;
				if (tim == this->M) tim = 0;
				//stout << tim << std::endl;
				this->green_up = this->b_mat_up[tim] * this->green_up;
				this->green_down = this->b_mat_down[tim] * this->green_down;
				//this->green_up = this->green_up * this->b_mat_up[tim];
				//this->green_down = this->green_down * this->b_mat_down[tim];
			}
			//stout << std::endl;
		};
		int tim = (which_time);
		multiplier(tim);

		if (!arma::qr(Q_up, R_up, P_up, this->green_up, "matrix")) throw "decomposition failed\n";
		if (!arma::qr(Q_down, R_down, P_down, this->green_down, "matrix")) throw "decomposition failed\n";

		D_down = diagvec(R_down);
		D_up = diagvec(R_up);
		T_down = (diagmat(D_down).i()) * R_down * (P_down.t());
		T_up = (diagmat(D_up).i()) * R_up * (P_up.t());

		for (int i = 1; i < this->p; i++)
		{
			// starting the multiplication
			tim = (which_time + i * this->M_0) % this->M;
			multiplier(tim);
			this->green_up = (this->green_up * Q_up) * diagmat(D_up);				// multiply by the former ones
			this->green_down = (this->green_down * Q_down) * diagmat(D_down);		// multiply by the former ones
			//this->green_down.print();

			if (!arma::qr(Q_up, R_up, P_up, this->green_up)) throw "decomposition failed\n";
			if (!arma::qr(Q_down, R_down, P_down, this->green_down)) throw "decomposition failed\n";

			D_up = diagvec(R_up);
			D_down = diagvec(R_down);
			//D_up.print();

			T_up = ((diagmat(D_up).i()) * R_up) * P_up.t() * T_up;
			T_down = ((diagmat(D_down).i()) * R_down) * P_down.t() * T_down;
		}
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

/// <summary>
/// Function to calculate the change in the potential exponential
/// Attractive case needs to be done
/// </summary>
/// <param name="lat_site">lattice site on which the change has been made</param>
/// <returns>A tuple for gammas for two spin channels, 0 is spin up, 1 is spin down</returns>
std::tuple<double, double> hubbard::HubbardQR::cal_gamma(int lat_site) const
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
std::tuple<double, double> hubbard::HubbardQR::cal_proba(int lat_site, double gamma_up, double gamma_down) const
{
	//double tmp_up = exp(-2*this->lambda * this->hsFields[this->current_time][lat_site]) - 1.0;
	//double tmp_down = exp(2*this->lambda * this->hsFields[this->current_time][lat_site]) - 1.0;
	//stout << "up: " << tmp_up << "\tdown: " << tmp_down << std::endl;
	//stout << "gammaUp: " << gamma_up << "\tgamma_down: " << gamma_down << std::endl << std::endl << std::endl;
	// SPIN UP
	const double p_up = 1.0 + gamma_up * (1.0L - this->green_up(lat_site, lat_site));
	// SPIN DOWN
	const double p_down = 1.0 + gamma_down * (1.0L - this->green_down(lat_site, lat_site));

	return std::make_tuple(p_up, p_down);
}

/// <summary>
/// Update the interaction matrix for current spin whenever the given lattice site HS field is changed
/// Only for testing purpose
/// </summary>
/// <param name="lat_site">the site of changed HS field</param>
/// <param name="delta_sigma">difference between changed and not</param>
/// <param name="sigma">spin channel</param>
void hubbard::HubbardQR::upd_int_exp(int lat_site, double delta_up, double delta_down)
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
void hubbard::HubbardQR::upd_B_mat(int lat_site, double delta_up, double delta_down) {
		this->b_mat_up[this->current_time].col(lat_site) *= delta_up;				// spin up
		this->b_mat_down[this->current_time].col(lat_site) *= delta_down;			// spin down
		//this->b_mat_up[this->current_time](j, lat_site) *= delta_up;						// spin up
		//this->b_mat_down[this->current_time](j, lat_site) *= delta_down;					// spin down
}

/// <summary>
/// After changing one spin we need to update the Green matrices via the Dyson equation
/// </summary>
/// <param name="lat_site">the site on which HS field has been changed</param>
/// <param name="prob_up">the changing probability for up channel</param>
/// <param name="prob_down">the changing probability for down channel</param>
/// <param name="gamma_up">changing parameter gamma for up channel</param>
/// <param name="gamma_down">changing probability for down channel</param>
void hubbard::HubbardQR::upd_equal_green(int lat_site, double prob_up, double prob_down, double gamma_up, double gamma_down)
{
	const double gamma_over_prob_up = gamma_up / prob_up;
	const double gamma_over_prob_down = gamma_down / prob_down;
	// create temporaries as the elements cannot change inplace
	this->tempGreen_up = this->green_up;
	this->tempGreen_down = this->green_down;
#pragma omp parallel for num_threads(this->inner_threads)
	for (int a = 0; a < this->Ns; a++) {
		const int delta = (lat_site == a) ? 1 : 0;
		for (int b = 0; b < this->Ns; b++) {
			// SPIN UP
			this->green_up(a,b) = tempGreen_up(a,b) - (delta - tempGreen_up(a,lat_site))*gamma_over_prob_up * tempGreen_up(lat_site,b);
			//this->green_up(a, b) += ((tempGreen_up(a, lat_site) - delta) * gamma_over_prob_up * tempGreen_up(lat_site, b));
			// SPIN DOWN
			this->green_down(a,b) = tempGreen_down(a,b) - (delta - tempGreen_down(a,lat_site))*gamma_over_prob_down * tempGreen_down(lat_site,b);
			//this->green_down(a, b) += ((tempGreen_down(a, lat_site) - delta) * gamma_over_prob_down * tempGreen_down(lat_site, b));
		}
	}
}

/// <summary>
/// Update the Green's matrices after going to next Trotter time, remember, the time is taken to be the previous one
/// <param name="which_time">updating to which_time + 1</param>
/// </summary>
void hubbard::HubbardQR::upd_next_green(int which_time) {
	this->green_up = (this->b_mat_up[which_time] * this->green_up) * this->b_mat_up[which_time].i();						// LEFT INCREASE
	this->green_down = (this->b_mat_down[which_time] * this->green_down) * this->b_mat_down[which_time].i();				// LEFT INCREASE;

	//this->green_up = (this->b_mat_up[which_time].i() * this->green_up) * this->b_mat_up[which_time];						// RIGHT DECREASE
	//this->green_down = (this->b_mat_down[which_time].i() * this->green_down) * this->b_mat_down[which_time];				// RIGHT DECREASE;
	//this->green_up.print("green up after update from t= " + std::to_string(which_time));
}

// -------------------------------------------------------- CALCULATORS
/// <summary>
/// Single step for the candidate to flip the HS field
/// </summary>
/// <param name="lat_site">the candidate lattice site</param>
/// <returns>sign of probability</returns>
int hubbard::HubbardQR::heat_bath_single_step(int lat_site)
{
	//this->hsFields[this->current_time][lat_site] = -this->hsFields[this->current_time][lat_site];	// try to flip before, why not
	const auto [gamma_up, gamma_down] = this->cal_gamma(lat_site);									// first up then down
	const auto [proba_up, proba_down] = this->cal_proba(lat_site, gamma_up, gamma_down);			// take the probabilities
	double proba = proba_up * proba_down;															// Metropolis probability
	//double multiplier = exp(2*hsFields[current_time][lat_site]*lambda);									// https://iopscience.iop.org/article/10.1088/1742-6596/1483/1/012002/pdf
	//proba = proba / (1.0 + proba);																	// heat-bath probability
	const int sign = proba >= 0 ? 1 : -1;
	if (this->ran.randomReal_uni(0,1) < sign * proba) {
		this->upd_int_exp(lat_site, gamma_up + 1, gamma_down + 1);
		this->upd_B_mat(lat_site, gamma_up + 1, gamma_down + 1);									// update the B matrices
		this->upd_equal_green(lat_site, proba_up, proba_down, gamma_up, gamma_down);				// update Greens via Dyson
		this->hsFields[this->current_time][lat_site] = -this->hsFields[this->current_time][lat_site];

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
		this->print_hs_fields(file_conf, this->current_time, lat_site, this->hsFields[this->current_time][lat_site]);
		file_conf.close();
		file_log << name_conf << "\t" << proba << "\t" << sign << std::endl;
		file_log.close();
	}
	// continue with a standard approach
	if (this->ran.randomReal_uni() < abs(proba)) {
		this->hsFields[this->current_time][lat_site] *= -1;
		this->upd_B_mat(lat_site, gamma_up + 1, gamma_down + 1);								// update the B matrices
		this->upd_equal_green(lat_site, proba_up, proba_down, gamma_up, gamma_down);		// update Greens via Dyson
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
		stout << "\t\t----> STARTING RELAXING FOR : " + this->info << std::endl;
		this->neg_num = 0;																				// counter of negative signs
		this->pos_num = 0;																				// counter of positive signs
	}
	// Progress bar
	auto progress = pBar();
	const double percentage = 5;
	const int percentage_steps = static_cast<int>(percentage * mcSteps / 100.0);

	// function
	int (HubbardQR:: * ptfptr)(int);																	// pointer to a single step function depending on whether we do configs or not

	if (conf) {
		stout << "Saving configurations of Hubbard Stratonovich fields\n";
		ptfptr = &HubbardQR::heat_bath_single_step_conf;												// pointer to saving configs
	}
	else
		ptfptr = &HubbardQR::heat_bath_single_step;														// pointer to non-saving configs

	for (int step = 0; step < mcSteps; step++) {
		// Monte Carlo steps
		for (int time_im = 0; time_im < this->M; time_im++) {
			// imaginary Trotter times
			//stout << "mc_step = " << step << ", time_im = " << time_im << std::endl;
			this->current_time = time_im;
			if (this->current_time % this->M_0 == 0) {
				// we have updated the sector of B matrices right befor the new one starting form current time!
				const int sector_to_upd = myModuloEuclidean(static_cast<int>(this->current_time / double(this->M_0)) - 1, this->p);
				this->cal_B_mat_cond(sector_to_upd);
				this->cal_green_mat(this->current_time);
				// compare Green's functions
				// this->compare_green_direct(this->current_time, 1e-6, false);
			}
			else {
				this->upd_next_green(this->current_time - 1);
				// compare Green's functions
				// this->compare_green_direct(this->current_time, 1e-6, false);
			}
			for (int j = 0; j < this->Ns; j++) {
				int sign = (this->*ptfptr)(j);															// get current sign and make single step
				sign > 0 ? this->pos_num++ : this->neg_num++;											// increase sign
				// compare Green's functions
				// this->compare_green_direct(this->current_time, 1e-6, false);
			}
		}
		if (mcSteps != 1 && step % percentage_steps == 0) {
			stout << "\t\t\t\t-> time: " << tim_s(start) << " -> RELAXATION PROGRESS for " << this->info  << " : \n";
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
void hubbard::HubbardQR::heat_bath_av(int corr_time, int avNum, bool quiet, bool times)
{
	auto start = std::chrono::high_resolution_clock::now();
#pragma omp critical
	stout << "\t\t----> STARTING AVERAGING FOR : " + this->info << std::endl;
	this->neg_num = 0L;																				// counter of negative signs
	this->pos_num = 0L;																				// counter of positive signs

	// Progress bar
	auto progress = pBar();
	const double percentage = 5;
	const int percentage_steps = static_cast<int>(percentage * avNum / 100.0);

	for (int step = 0; step < avNum; step++) {
		// Monte Carlo steps
		for (int time_im = 0; time_im < this->M; ++time_im) {
			// imaginary Trotter times
			this->current_time = time_im;
			if (this->current_time % this->M_0 == 0) {
				// we have updated the sector of B matrices right befor the new one starting form current time!
				int sector_to_upd = myModuloEuclidean(static_cast<int>(this->current_time / double(this->M_0)) - 1, this->p);
				this->cal_B_mat_cond(sector_to_upd);
				this->cal_green_mat(this->current_time);
			}
			else
				this->upd_next_green(this->current_time-1);
			for (int i = 0; i < this->Ns; i++) {
				// go through the lattice
				const int sign = this->heat_bath_single_step(i);
				sign >= 0 ? this->pos_num += 1L : this->neg_num += 1L;								// increase sign
				// stout << "sign: " << sign << ", pos: " << this->pos_num << ", neg: " << this->neg_num << std::endl;
				this->av_single_step(i, sign);														// collect all averages
			}
			this->avs->av_gr_down += this->green_down;
			this->avs->av_gr_up += this->green_up;
		}
		// erease correlations
		for (int corr = 1; corr < corr_time; corr++) {
			this->heat_bath_eq(1, false, true);
		}
		if (step % percentage_steps == 0) {
#pragma omp critical
			stout << "\t\t\t\t-> time: " << tim_s(start) << " -> AVERAGES PROGRESS for " << this->info  << " : \n";
#pragma omp critical
			progress.print();
#pragma omp critical
			stout << std::endl;
			progress.update(percentage);
		}

	}
	// After
	this->av_normalise(avNum, times);
}

// -------------------------------------------------------- PUBLIC CALCULATORS

/// <summary>
/// Equilivrate the simulation
/// </summary>
/// <param name="algorithm">type of equilibration algorithm</param>
/// <param name="mcSteps">Number of Monte Carlo steps</param>
/// <param name="conf">Shall print configurations?</param>
/// <param name="quiet">Shall be quiet?</param>
void hubbard::HubbardQR::relaxation(impDef::algMC algorithm, int mcSteps, bool conf, bool quiet)
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
void hubbard::HubbardQR::average(impDef::algMC algorithm, int corr_time, int avNum, int bootStraps, bool quiet, int times)
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

/* ---------------------------- HUBBARD MODEL WITH SPACE TIME FORMULATION ---------------------------- */