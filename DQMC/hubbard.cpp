#include "include/hubbard.h"
/* ---------------------------- HUBBARD MODEL ---------------------------- */
// ---- HELPERS
/// <summary>
/// A single step for calculating averages inside a loop
/// </summary>
/// <param name="current_elem_i"> Current Green matrix element in averages</param>
void hubbard::HubbardModel::av_single_step(int current_elem_i, int sign)
{
	// m_z
	const double m_z2 = this->cal_mz2(sign, current_elem_i);
	this->avs->av_M2z += m_z2;
	this->avs->sd_M2z += m_z2 * m_z2;
	// m_x 
	const double m_x2 = this->cal_mz2(sign, current_elem_i);
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
		this->avs->av_M2z_corr[x][y][z] += this->cal_mz2_corr(sign,current_elem_i,current_elem_j);
		this->avs->av_occupation_corr[x][y][z] += this->cal_occupation_corr(sign,current_elem_i,current_elem_j);
		this->avs->av_ch2_corr[x][y][z] += this->cal_ch_correlation(sign,current_elem_i, current_elem_j) / (this->Ns * 2.0);
	}
}
/// <summary>
/// Normalise all the averages taken during simulation
/// </summary>
/// <param name="avNum">Number of avs taken</param>
/// <param name="times">If the non-equal time properties were calculated</param>
void hubbard::HubbardModel::av_normalise(int avNum, bool times)
{
	const long double normalization = static_cast<long double>(avNum * this->M * this->Ns);						// average points taken
	this->avs->av_sign = (this->pos_num - this->neg_num) / 1.0*(this->pos_num + this->neg_num);					// average sign is needed
	this->avs->sd_sign = sqrt((1.0 - (this->avs->av_sign * this->avs->av_sign)) / normalization);					
	const long double normalisation_sign = normalization * this->avs->av_sign;									// we divide by average sign actually
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

// ---- SETTERS
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


// ---- GETTERS
/// <summary>
/// Get the number of Trotter times
/// </summary>
/// <returns>the number of Trotter times</returns>
int hubbard::HubbardModel::get_M() const
{
	return this->M;
}
/// <summary>
/// Get the number of QR decompositions or number of Trotter times for reduced Green's matrix block
/// </summary>
/// <returns>the number of QR decompositions or number of Trotter times for reduced Green's matrix block</returns>
int hubbard::HubbardModel::get_M_0() const
{
	return this->M_0;
}

std::string hubbard::HubbardModel::get_info() const
{
	return this->info;
}

// ---- SETTERS
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
	this->neg_log = this->neg_dir.substr(0,\
		this->neg_dir.length() - 9) + "neg_log," + info + ".dat";								// for storing the labels of negative files in csv for ML
	this->pos_log = this->pos_dir.substr(0,\
		this->pos_dir.length() - 9) + "pos_log," + info + ".dat";								// for storing the labels of positive files in csv for ML
	fileN.open(this->neg_log);
	fileP.open(this->pos_log);
	fileN.close();																				// close just to create file neg
	fileP.close();																				// close just to create file pos
}

// ---- CALCULATORS
/// <summary>
/// Function to calculate the hopping matrix exponential (with nn for now)
/// </summary>
void hubbard::HubbardModel::cal_hopping_exp()
{
	for (int i = 0; i < this->Ns; i++) {
		this->hopping_exp(i, i) = this->dtau * this->mu;								// diagonal elements
		const int n_of_neigh = this->lattice->get_nn_number(i);							// take number of nn at given site
		for (int j = 0; j < n_of_neigh; j++) {
			const int where_neighbor = this->lattice->get_nn(i, j);						// get given nn
			this->hopping_exp(i, where_neighbor) = this->dtau * this->t[i];				// assign non-diagonal elements
		}
	}
	this->hopping_exp = arma::expmat(this->hopping_exp);								// take the exponential
}
/// <summary>
/// Function to calculate the interaction exponential at all times, each column represents the given Trotter time
/// </summary>
void hubbard::HubbardModel::cal_int_exp(){
	if(this->U > 0){
	/* Repulsive case */
		for (int l = 0; l < this->M; l++) {
		/* Trotter times */
			for (int i = 0; i < this->Ns; i++){
			/* Lattice sites */
				this->int_exp_up(i,l) = exp(this->lambda * hsFields[l][i]);						// diagonal up spin channel
				this->int_exp_down(i,l) = exp(-this->lambda * hsFields[l][i]);					// diagonal down spin channel
			}
		}
	}
	else if(U < 0){
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
void hubbard::HubbardModel::cal_B_mat(){
	for (int l = 0; l < this->M; l++) {
		/* Trotter times */
			this->b_mat_down[l] = this->hopping_exp * arma::diagmat(this->int_exp_down.col(l));
			this->b_mat_up[l] = this->hopping_exp * arma::diagmat(this->int_exp_up.col(l));
	}
}

// ---- PRINTERS 

void hubbard::HubbardModel::print_hs_fields(std::ostream& output, int which_time_caused, int which_site_caused, short this_site_spin, std::string separator)
{
	for (int i = 0; i < this->M; i++) {
		for (int j = 0; j < this->Ns; j++) {
			if(j==which_site_caused && i == which_time_caused)
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

// ---- EQUAL TIME AVERAGES

double hubbard::HubbardModel::cal_kinetic_en(int sign, int current_elem_i)
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

double hubbard::HubbardModel::cal_occupation(int sign, int current_elem_i)
{
	return (sign * (1.0 - this->green_down(current_elem_i, current_elem_i)) + sign * (1.0 - this->green_up(current_elem_i, current_elem_i)));
}

double hubbard::HubbardModel::cal_occupation_corr(int sign, int current_elem_i, int current_elem_j)
{
	return sign * ((this->green_down(current_elem_j, current_elem_i) + this->green_up(current_elem_j, current_elem_i)));
}

double hubbard::HubbardModel::cal_mz2(int sign, int current_elem_i)
{
	return sign * (((1.0 - this->green_up(current_elem_i, current_elem_i)) * (1.0 - this->green_up(current_elem_i, current_elem_i)))
		+ ((1.0 - this->green_up(current_elem_i, current_elem_i)) * (this->green_up(current_elem_i, current_elem_i)))
		- ((1.0 - this->green_up(current_elem_i, current_elem_i)) * (1.0 - this->green_down(current_elem_i, current_elem_i)))
		- ((1.0 - this->green_down(current_elem_i, current_elem_i)) * (1.0 - this->green_up(current_elem_i, current_elem_i)))
		+ ((1.0 - this->green_down(current_elem_i, current_elem_i)) * (1.0 - this->green_down(current_elem_i, current_elem_i)))
		+ ((1.0 - this->green_down(current_elem_i, current_elem_i)) * (this->green_down(current_elem_i, current_elem_i))));
}

double hubbard::HubbardModel::cal_mz2_corr(int sign, int current_elem_i, int current_elem_j)
{
	long double delta_ij = 0.0L;
	if (current_elem_i == current_elem_j) {
		delta_ij = 1.0L;
	}
	//this->green_down.print("TEST");
	return sign*(((1.0L - this->green_up(current_elem_i, current_elem_i)) * (1.0L - this->green_up(current_elem_j, current_elem_j)))
		+ ((delta_ij - this->green_up(current_elem_j, current_elem_i)) * (this->green_up(current_elem_i, current_elem_j)))
		- ((1.0L - this->green_up(current_elem_i, current_elem_i)) * (1.0L - this->green_down(current_elem_j, current_elem_j)))
		- ((1.0L - this->green_down(current_elem_i, current_elem_i)) * (1.0L - this->green_up(current_elem_j, current_elem_j)))
		+ ((1.0L - this->green_down(current_elem_i, current_elem_i)) * (1.0L - this->green_down(current_elem_j, current_elem_j)))
		+ ((delta_ij - this->green_down(current_elem_j, current_elem_i)) * (this->green_down(current_elem_i, current_elem_j))));
}

double hubbard::HubbardModel::cal_my2(int sign, int current_elem_i)
{
	return 0;
}

double hubbard::HubbardModel::cal_mx2(int sign, int current_elem_i)
{
	return sign * (1.0 - this->green_up(current_elem_i, current_elem_i)) * (this->green_down(current_elem_i, current_elem_i))
		+ sign * (1.0 - this->green_down(current_elem_i, current_elem_i)) * (this->green_up(current_elem_i, current_elem_i));
}

double hubbard::HubbardModel::cal_ch_correlation(int sign, int current_elem_i, int current_elem_j)
{
	long double delta_ij = 0.0L;
	if (current_elem_i == current_elem_j) {
		delta_ij = 1.0L;
	}
	return sign*(((1 - this->green_up(current_elem_i, current_elem_i)) * (1 - this->green_up(current_elem_j, current_elem_j))					//sigma = sigma' = up
		+ (1 - this->green_down(current_elem_i, current_elem_i)) * (1 - this->green_down(current_elem_j, current_elem_j))				//sigma = sigma' = down
		+ (1 - this->green_down(current_elem_i, current_elem_i)) * (1 - this->green_up(current_elem_j, current_elem_j))					//sigma = down, sigma' = up
		+ (1 - this->green_up(current_elem_i, current_elem_i)) * (1 - this->green_down(current_elem_j, current_elem_j))					//sigma = up, sigma' = down
		+ ((delta_ij - this->green_up(current_elem_j, current_elem_i)) * this->green_up(current_elem_i, current_elem_j))				//sigma = sigma' = up
		+ ((delta_ij - this->green_down(current_elem_j, current_elem_i)) * this->green_down(current_elem_i, current_elem_j))));			//sigma = sigma' = down
}

/* ---------------------------- HUBBARD MODEL WITH QR DECOMPOSITION ---------------------------- */

// ---- CONSTRUCTORS
hubbard::HubbardQR::HubbardQR(const std::vector<double>& t, int M_0, double U, double mu, double beta, std::shared_ptr<Lattice> lattice)
{
	this->lattice = lattice;
	this->avs = {};
	this->t = t;
	/* Params */
	this->U = U;
	this->mu = mu;
	this->beta = beta;
	this->T = 1.0 / this->beta;
	this->Ns = this->lattice->get_Ns();
	this->ran = randomGen();																// random number generator initialization
	/* Trotter */
	this->dtau = dtau;
	this->M = static_cast<int>(this->beta / this->dtau);									// number of Trotter times
	this->M_0 = M_0;
	this->p = (this->M / this->M_0);														// number of QR decompositions
	/* Calculate alghorithm parameters */
	this->lambda = std::acoshl(std::expl((abs(this->U) * this->dtau )/2));
	/* Calculate changing exponents before, not to calculate exp all the time */
	this->gammaExp = {std::expl(2L * this->lambda), std::expl(-2L * this->lambda)};			// 0 -> sigma * hsfield = -1, 1 -> sigma * hsfield = 1
	/* Helping params */
	this->from_scratch = 8;
	this->pos_num = 0;
	this->neg_num = 0;
	/* Say hi to the world */
#pragma omp critical
	PLOG_INFO << "CREATING THE HUBBARD MODEL WITH QR DECOMPOSITION WITH PARAMETERS:" << std::endl \
	// decomposition
	<< "->M = " << this->M << std::endl \
	<< "->M_0 = " << this->M_0 << std::endl \
	<< "->p = " << this->p << std::endl \
	// physical
	<< "->beta = " << this->beta << std::endl \
	<< "->U = " << this->U << std::endl \
	<< "->dtau = " << this->dtau << std::endl \
	<< "->mu = " << this->mu << std::endl \
	// lattice
	<< "->dimension = " << this->getDim() << std::endl \
	<< "->Lx = " << this->lattice->get_Lx() << std::endl \
	<< "->Ly = " << this->lattice->get_Ly() << std::endl \
	<< "->Lz = " << this->lattice->get_Lz() << std::endl \
	<< "->lambda = " << this->lambda << std::endl << std::endl;
	/* Setting info about the model for files */
	this->info = "M=" + std::to_string(this->M) + ",M_0=" + std::to_string(this->M_0) +\
		",dtau=" + to_string_prec(this->dtau) + ",Lx=" + std::to_string(this->lattice->get_Lx()) +\
		",Ly=" + std::to_string(this->lattice->get_Ly()) + ",Lz=" + std::to_string(this->lattice->get_Lz()) +\
		",beta=" + to_string_prec(this->beta) + ",U=" + to_string_prec(this->U) +\
		",mu=" + to_string_prec(this->mu);

	/* Initialize memory */
	this->hopping_exp = arma::mat(this->Ns, this->Ns, arma::fill::zeros);
	// interaction for all times
	this->int_exp_down = arma::mat(this->Ns, this->M, arma::fill::zeros);
	this->int_exp_up = arma::mat(this->Ns, this->M, arma::fill::zeros);
	// all times exponents multiplication
	this->b_mat_up = std::vector<arma::mat>(this->M, arma::mat(this->Ns, this->Ns, arma::fill::zeros));
	this->b_mat_down = std::vector<arma::mat>(this->M, arma::mat(this->Ns, this->Ns, arma::fill::zeros));
	// all times hs fields for real spin up and down
	this->hsFields = std::vector(this->M, std::vector<short>(this->Ns, 1));
	// Green's function matrix
	this->green_up = arma::eye(this->Ns, this->Ns);
	this->green_down = arma::eye(this->Ns, this->Ns);
	/* Set HS fields */
	this->set_hs();
	/* Calculate something */
	this->cal_hopping_exp();
	this->cal_int_exp();
	this->cal_B_mat();
}

// ---- OTHER METHODS
/// <summary>
/// Calculate Green with QR decomposition using LOH : doi:10.1016/j.laa.2010.06.023
/// For more look into :
/// "Advancing Large Scale Many-Body QMC Simulations on GPU Accelerated Multicore Systems"
/// In order to do that the M_0 and p variables will be used to divide the multiplication into smaller chunks of matrices
/// </summary>
/// <param name="which_time"></param>
void hubbard::HubbardQR::cal_green_mat(int which_time){
	/* QR HELPING MATRICES */
	arma::mat Q_down(this->Ns, this->Ns);
	arma::mat Q_up(this->Ns, this->Ns);

	arma::vec D_down(this->Ns, arma::fill::ones);
	arma::vec D_up(this->Ns, arma::fill::ones);

	arma::mat T_down(this->Ns, this->Ns, arma::fill::ones);
	arma::mat T_up(this->Ns, this->Ns, arma::fill::ones);

	int time = which_time;
	// Stable solutions of linear systems involving long chain of matrix multiplications: doi:10.1016/j.laa.2010.06.023

	for(int i =0; i < this->p;i++)
	{
		arma::umat P_down;													// permutation matrix for spin down
		arma::umat P_up;													// permutation matrix for spin up
		arma::mat R_down;													// right triangular matrix down
		arma::mat R_up;														// right triangular matrix up

		// starting the multiplication
		this->green_up = this->b_mat_up[time];
		this->green_down = this->b_mat_down[time];

		time = (time == this->M-1) ? 0 : time + 1;

		// multiplication is taken from higher time on left to lower on the right
		for (int j = 1; j < this->M_0; j++) {
			this->green_up =  this->b_mat_up[time] * this->green_up;
			this->green_down = this->b_mat_down[time] * this->green_down;

			time = (time == this->M-1) ? 0 : time + 1;
		}
		this->green_up = (this->green_up*Q_up)*diagmat(D_up);				// multiply by the former ones
		this->green_down = (this->green_down*Q_down)*diagmat(D_down);		// multiply by the former ones

		arma::qr(Q_up, R_up, P_up, this->green_up, "matrix");
		arma::qr(Q_down, R_down, P_down, this->green_down, "matrix");

		D_up = diagvec(R_up);
		D_down = diagvec(R_down);
		T_up = ((arma::inv(diagmat(D_up)) * R_up) * (P_up.t() *T_up));
		T_down = ((arma::inv(diagmat(D_down)) * R_down) * (P_down.t()*T_down));
	}
	/* Correction terms */
	arma::vec Ds_up = D_up;
	arma::vec Ds_down = D_down;

	arma::vec Db_up = D_up;
	arma::vec Db_down = D_down;

	for(int i = 0; i < this->Ns;i++)
	{
		if(abs(D_up(i)) > 1)
		{
			Db_up(i) = D_up(i);
			Ds_up(i) = 1;
		}
		else
		{
			Db_up(i) =1;
			Ds_up(i) = D_up(i);
		}
		if(abs(D_down(i)) > 1)
		{
			Db_down(i) = D_down(i);
			Ds_down(i) = 1;
		}
		else
		{
			Db_down(i) =1;
			Ds_down(i) = D_down(i);
		}
	}

	this->green_up = arma::solve(arma::diagmat(Db_up).i() * Q_up.t() + arma::diagmat(Ds_up) *T_up, arma::diagmat(Db_up).i()*Q_up.t());
	this->green_down = arma::solve(arma::diagmat(Db_down).i() * Q_down.t() + arma::diagmat(Ds_down) *T_down, arma::diagmat(Db_down).i()*Q_down.t());
}
/// <summary>
/// Function to calculate the change in the potential exponential
/// Attractive case needs to be done
/// </summary>
/// <param name="lat_site">lattice site on which the change has been made</param>
/// <returns>A tuple for gammas for two spin channels, 0 is spin up, 1 is spin down</returns>
std::tuple<long double, long double> hubbard::HubbardQR::cal_gamma(int lat_site)
{
	std::tuple< double,  double> tmp(0, 0);								// here we will save the gammas
	if (this->U > 0) {
		/* Repulsive case */
		if (this->hsFields[this->current_time][lat_site] > 0)
		{
			tmp = std::make_tuple(this->gammaExp[1] - 1.0, this->gammaExp[0] - 1.0);
		}
		else
		{
			tmp = std::make_tuple(this->gammaExp[0] - 1.0, this->gammaExp[1] - 1.0);
		}
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
std::tuple<long double, long double> hubbard::HubbardQR::cal_proba(int lat_stie, long double gamma_up, long double gamma_down)
{
	/* SPIN UP */
	const double p_up = (1.0 + ((1.0 - this->green_up(lat_stie, lat_stie)) * gamma_up));
	/* SPIN DOWN */
	const double p_down = (1.0 + ((1.0 - this->green_down(lat_stie, lat_stie)) * gamma_down));

	return std::make_tuple(p_up, p_down);
}
/// <summary>
/// Update the interaction matrix for current spin whenever the given lattice site HS field is changed
/// Only for testing purpose
/// </summary>
/// <param name="lat_site">the site of changed HS field</param>
/// <param name="delta_sigma">difference between changed and not</param>
/// <param name="sigma">spin channel</param>
void hubbard::HubbardQR::upd_int_exp(int lat_site, long double delta_sigma, short sigma)
{
	if (sigma > 0) {
		this->int_exp_up(lat_site, this->current_time) *= delta_sigma;
	}
	else {
		this->int_exp_down(lat_site, this->current_time) *= delta_sigma;
	}
}
/// <summary>
/// After accepting spin change update the B matrix by multiplying it by diagonal element ( the delta )
/// </summary>
/// <param name="lat_site"></param>
/// <param name="delta_up"></param>
/// <param name="delta_down"></param>
void hubbard::HubbardQR::upd_B_mat(int lat_site, long double delta_up, long double delta_down){
	for (int j = 0; j < this->Ns; j++)
	{
		this->b_mat_up[this->current_time](j, lat_site) *= delta_up;			// spin up
		this->b_mat_down[this->current_time](j, lat_site) *= delta_down;		// spin down
	}
}
/// <summary>
/// After changing one spin we need to update the Green matrices via the Dyson equation 
/// </summary>
/// <param name="lat_site">the site on which HS field has been changed</param>
/// <param name="prob_up">the changing probability for up channel</param>
/// <param name="prob_down">the changing probability for down channel</param>
/// <param name="gamma_up">changing parameter gamma for up channel</param>
/// <param name="gamma_down">changing probability for down channel</param>
void hubbard::HubbardQR::upd_equal_green(int lat_site, long double prob_up, long double prob_down, long double gamma_up, long double gamma_down)
{
	const long double gamma_over_prob_up = gamma_up / prob_up;
	const long double gamma_over_prob_down = gamma_down / prob_down;
	// create temporaries as the elements cannot change inplace
	const arma::mat tempGreen_up = this->green_up;
	const arma::mat tempGreen_down = this->green_down;
		for (int a = 0; a < this->Ns; a++) {
		const int delta = (lat_site == a) ? 1 : 0;
		for (int b = 0; b < this->Ns; b++) {
			// SPIN UP
			this->green_up(a, b) +=  ((tempGreen_up(a, lat_site) - delta) * gamma_over_prob_up * tempGreen_up(lat_site, b));
			// SPIN DOWN
			this->green_down(a, b) +=  ((tempGreen_down(a, lat_site) - delta) * gamma_over_prob_down * tempGreen_down(lat_site, b));		}
	}
}
/// <summary>
/// Update the Green's matrices after going to next Trotter time, remember, the time is taken to be the previous one
/// <param name="which_time">updating to which_time + 1</param>
/// </summary>
void hubbard::HubbardQR::upd_next_green(int which_time){
	this->green_up = (this->b_mat_up[which_time] * this->green_up)* arma::inv(this->b_mat_up[which_time]);
	this->green_down = (this->b_mat_down[which_time] * this->green_down) * arma::inv(this->b_mat_down[which_time]);
}


// CALCULATORS
/// <summary>
/// Single step for the candidate to flip the HS field
/// </summary>
/// <param name="lat_site">the candidate lattice site</param>
/// <returns>sign of probability</returns>
int hubbard::HubbardQR::heat_bath_single_step(int lat_site)
{
	const auto [gamma_up,gamma_down] = this->cal_gamma(lat_site);							// first up then down
	const auto [proba_up,proba_down] = this->cal_proba(lat_site,gamma_up, gamma_down);		// take the probabilities
	long double proba = proba_up*proba_down;												// Metropolis probability
	proba = proba/(1.0L+proba);																// heat-bath probability
	if(this->ran.randomReal_uni() < abs(proba)){
		this->hsFields[this->current_time][lat_site] *= -1;
		this->upd_B_mat(lat_site, gamma_up +1, gamma_down + 1);								// update the B matrices
		this->upd_equal_green(lat_site, proba_up, proba_down, gamma_up, gamma_down);		// update Greens via Dyson
	}
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
	const auto [gamma_up,gamma_down] = this->cal_gamma(lat_site);							// first up then down
	const auto [proba_up,proba_down] = this->cal_proba(lat_site,gamma_up, gamma_down);		// take the probabilities
	long double proba = proba_up*proba_down;												// Metropolis probability
	proba = proba/(1.0L+proba);																// heat-bath probability
	int sign = 1;
	if(proba < 0){
		sign = -1;
		proba = -proba;																		// set abs(proba)
		if (this->neg_num <= this->pos_num){												// to maitain the same number of both signs
			name_conf = this->neg_dir + "negative_" + this->info + \
			",n=" + std::to_string(this->neg_num) + ".dat";
			name_log = this->neg_log;
		}
	}
	else{
		if (this->neg_num <= this->pos_num){												// to maitain the same number of both signs
			name_conf = this->pos_dir + "positive_" + this->info + \
			",n=" + std::to_string(this->pos_num) + ".dat";
			name_log = this->pos_log;
		}
	}
	// open files
	file_log.open(name_log);
	file_conf.open(name_conf);
	if(!file_conf.is_open() || !file_log.is_open()){
		PLOG_ERROR  << "Couldn't open either: " + name_log + " , or " + name_conf + "\n";
		throw -1;
		exit(-1);
	}
	else{
		this->print_hs_fields(file_conf, this->current_time, lat_site, this->hsFields[this->current_time][lat_site]);
		file_conf.close();
		file_log << name_conf << "\t" << proba << "\t" << sign << std::endl;
		file_log.close();
	}
	// continue with a standard approach
	if(this->ran.randomReal_uni() < abs(proba)){
		this->hsFields[this->current_time][lat_site] *= -1;
		this->upd_B_mat(lat_site, gamma_up +1, gamma_down + 1);								// update the B matrices
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
	this->neg_num = 0;																				// counter of negative signs
	this->pos_num = 0;																				// counter of positive signs
	int (HubbardQR::* ptfptr)(int);																	// pointer to a single step function depending on 
	int sign = 1;

	if(conf){
		PLOG_INFO << "Saving configurations of Hubbard Stratonovich fields\n";
		ptfptr = &HubbardQR::heat_bath_single_step_conf;											// pointer to saving configs
	}
	else
		ptfptr = &HubbardQR::heat_bath_single_step;													// pointer to non-saving configs

	for (int step = 0; step < mcSteps; step++) {
	// Monte Carlo steps
		for (int time_im = 0; time_im < this->M; time_im++){
		// imaginary Trotter times
			this->current_time = time_im;
			if (time_im % this->from_scratch == 0) this->cal_green_mat(this->current_time);			// calculate the Greens from scratch
			else if(time_im != this->M - 1) this->upd_next_green(this->current_time - 1);			// if it's possible update the Greens
			for (int i = 0; i < this->Ns; i++) {
			// go through the lattice
				sign = (this->*ptfptr)(i);															// get current sign and make single step
				sign>0 ? this->pos_num++ : this->neg_num++;											// increase sign
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
	this->neg_num = 0;																				// counter of negative signs
	this->pos_num = 0;																				// counter of positive signs
	int sign = 1;
	
	// For future purposes
	const int Lx = this->lattice->get_Lx();
	const int Ly = this->lattice->get_Ly();
	const int Lz = this->lattice->get_Lz();
	// Correlations - depend on the dimension - equal time
	this->avs->av_occupation_corr = v_3d<double>(2 * Lx - 1,  v_2d<double>(2 * Ly - 1,  v_1d<double>(2 * Lz - 1, 0.0)));
	this->avs->av_M2z_corr = this->avs->av_occupation_corr;
	this->avs->av_ch2_corr = this->avs->av_occupation_corr;
	// Setting av Greens
	this->avs->av_gr_down = arma::zeros(this->Ns, this->Ns);
	this->avs->av_gr_up = arma::zeros(this->Ns, this->Ns);


	for (int step = 0; step < avNum; step++) {
	// Monte Carlo steps
		uint64_t neg_points =0;
		uint64_t pos_points =0;
		for (int time_im = 0; time_im < this->M; time_im++){
		// imaginary Trotter times
			this->current_time = time_im;
			if (time_im % this->from_scratch == 0) this->cal_green_mat(this->current_time);			// calculate the Greens from scratch
			else if(time_im != this->M - 1) this->upd_next_green(this->current_time - 1);			// if it's possible update the Greens
			for (int i = 0; i < this->Ns; i++) {
			// go through the lattice
				sign = this->heat_bath_single_step(i);												// get current sign and make single step
				sign>0 ? pos_points++ : neg_points++;												// increase sign
				const int current_elem_i = i;														// set current Green i element for averages
				this->av_single_step(current_elem_i,sign);											// collect all averages
			}
			this->avs->av_gr_down += this->green_down;
			this->avs->av_gr_up += this->green_up;
		}
		// erease correlations
		for (int corr = 0; corr < corr_time-1; corr++) {
			this->heat_bath_eq(1, false, true);
		}
	}
	// After
	this->av_normalise(avNum, times);

}



// PUBLIC CALCULATORS
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
		PLOG_WARNING << "Didn't choose the algorithm type\n";
		exit(-1);
		break;
	}

	auto stop = std::chrono::high_resolution_clock::now();											// finishing timer for relaxation
	if(mcSteps!=1) 
		PLOG_INFO << "Relaxation Time taken: " << \
		(std::chrono::duration_cast<std::chrono::seconds>(stop - start)).count() << \
		" seconds. With average sign = " << \
		1.0 * (this->pos_num - this->neg_num) / (this->pos_num + this->neg_num) << std::endl;

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
		PLOG_WARNING << "Didn't choose the algorithm type\n";
		exit(-1);
		break;
	}
	auto stop = std::chrono::high_resolution_clock::now();											// finishing timer for relaxation
	
	PLOG_INFO << "Averages time taken: " << \
	(std::chrono::duration_cast<std::chrono::seconds>(stop - start)).count() << \
	" seconds. With average sign = " << \
	1.0 * (this->pos_num - this->neg_num) / (this->pos_num + this->neg_num) << std::endl;
}















/* ---------------------------- HUBBARD MODEL WITH SPACE TIME FORMULATION ---------------------------- */