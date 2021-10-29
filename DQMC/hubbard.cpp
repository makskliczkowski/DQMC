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
	const double normalization = static_cast<double>(avNum * timesNum * this->Ns);								// average points taken
	//this->avs->av_sign /= double(avNum);																			// average sign is needed
	this->avs->av_sign = (this->pos_num - this->neg_num) /double(this->pos_num + this->neg_num);
	this->avs->sd_sign = sqrt((1.0 - (this->avs->av_sign * this->avs->av_sign)) / double(avNum));
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

/// <summary>
/// 
/// </summary>
/// <param name="useWrapping"></param>
void hubbard::HubbardModel::averageUnequalGreens(bool useWrapping, const v_1d<arma::mat>& up, const v_1d<arma::mat>& down, u32 bucketnum)
{
	const int Lx = this->lattice->get_Lx();
	const int Ly = this->lattice->get_Ly();

	// average 
	u32 tau1 = 0;
	for (u32 tau2 = 0; tau2 < this->M; tau2++) {
		std::vector<std::vector<double>> norma = std::vector<std::vector<double>>(2*Lx-1, v_1d<double>(2*Ly-1, 0));
		this->g_up_diffs[tau2].zeros();
		this->g_down_diffs[tau2].zeros();
		for (int a = 0; a < this->Ns; a++) {
			for (int b = 0; b < this->Ns; b++) {
				const int j_minus_i_y = this->lattice->get_coordinates(b, 1) - this->lattice->get_coordinates(a, 1);
				const int j_minus_i_x = this->lattice->get_coordinates(b, 0) - this->lattice->get_coordinates(a, 0);
				const int y = j_minus_i_y + Ly - 1;
				const int x = j_minus_i_x + Lx - 1;
				g_up_diffs[tau2](x,y) += up[tau2](b,a);
				g_down_diffs[tau2](x,y) += down[tau2](b,a);
				norma[x][y] += 1;
			}
		}
		for (int i = 0; i < 2*Lx - 1; i++) {
			for (int j = 0; j < 2*Ly - 1; j++) {
				g_up_diffs[tau2](i,j) /= (norma[i][j] * bucketnum);
				g_down_diffs[tau2](i,j) /= (norma[i][j] * bucketnum);
			}
		}
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="filenum"></param>
/// <param name="useWrapping"></param>
void hubbard::HubbardModel::save_unequal_greens(int filenum, bool useWrapping, const v_1d<arma::mat>& up, const v_1d<arma::mat>& down, u32 bucketnum)
{
	std::string information = " Some version\n\n This is the file that contains real space Green's functions for different times.\n";
	information += " The structure of each is we average over time differences and first row\n";
	information += " before each Green matrix <cicj^+(t1, t2)> is an information about the difference\n";


	std::ofstream fileUp; openFile(fileUp, this->dir->time_greens_dir + std::to_string(filenum) + "-up" + this->dir->nameGreensTime);
	std::ofstream fileDown; openFile(fileDown, this->dir->time_greens_dir + std::to_string(filenum) + "-down" + this->dir->nameGreensTime);
	fileUp << " Up different time, real-space Greens\n" << information;
	fileDown << " Down different time, real-space Greens\n" << information;
	std::initializer_list<std::string> enter_params = { "n =\t", std::to_string(this->lattice->get_Lx()),"\n",
		"l =\t",std::to_string(this->M),"\n",
		"tausk =\t",std::to_string(this->M),"\n",
		"doall =\t",std::string("don't know what is that"),"\n",
		"densw =\t",std::string("don't know what is that"),"\n",
		"histn =\t",std::string("don't know what is that"),"\n",
		"iran =\t",std::string("don't know what is that"),"\n",
		"t  =\t",to_string_prec(this->t[0],5),"\n",
		"mu  =\t",std::to_string(this->mu),"\n",
		"delmu =\t",std::string("don't know what is that"),"\n",
		"bpar  =\t",std::string("don't know what is that"),"\n",
		"dtau  =\t",std::to_string(this->dtau),"\n",
		"warms  =\t",std::to_string(2000),		"\n",
		"sweeps =\t",std::to_string(2000),"\n",
		"u =\t", std::to_string(this->U),"\n",
		"nwrap =\t",std::to_string(this->M_0),"\n",
		"difflim =\t",std::string("don't know what is that"),"\n",
		"errrat =\t",std::string("don't know what is that"),	"\n",	
		"doauto =\t",std::to_string(this->M_0),"\n",
		"orthlen =\t",std::string("don't know what is that"),"\n",
		"eorth =\t",std::string("don't know what is that"),"\n",
		"dopair =\t",std::string("don't know what is that"),"\n",
		"numpair =\t",std::string("don't know what is that"),"\n",		
		"lambda =\t", to_string_prec(this->lambda, 4),"\n",
		"start = \t",std::to_string(0), "\n",
		"sign = \t",to_string_prec(this->config_sign,5), "\n"};
	printSeparated(fileUp, " ", enter_params, 30);
	printSeparated(fileDown, " ",enter_params, 30);
	//stout<<this->dir->time_greens_dir + std::to_string(filenum) + ",up" + this->dir->nameGreensTime << std::endl;
	const int Lx = this->lattice->get_Lx();
	const int Ly = this->lattice->get_Ly();

	this->averageUnequalGreens(useWrapping, up, down, bucketnum);
	const u16 width = 8;
	for (u32 i = 0; i <= Lx/2; i++) {
		// we also have negative vectors included
		const u32 x = i + Lx - 1;
		for (u32 j = i; j <= Ly/2; j++) {
			const u32 y = j + Ly - 1;
			printSeparated(fileUp, "\t", {std::string(" G(nx,ny,ti):\n nx ="), std::to_string(i), std::string("ny ="), std::to_string(j)},6);
			printSeparated(fileDown, "\t", {std::string(" G(nx,ny,ti):\n nx ="), std::to_string(i), std::string("ny ="), std::to_string(j)},6);
			for (int tau1 = 0; tau1 < this->M; tau1++)
			{
				printSeparated(fileUp, "\t", {std::to_string(tau1)}, 4,0);
				printSeparated(fileUp, "\t", {to_string_prec(this->g_up_diffs[tau1](x,y),width)}, width+5,0);
				printSeparated(fileUp, "\t", {std::string("+-"), std::to_string(0)}, 5);

				printSeparated(fileDown, "\t", {std::to_string(tau1)}, 4,0);
				printSeparated(fileDown, "\t", {to_string_prec(this->g_down_diffs[tau1](x,y),width)}, width+5,0);
				printSeparated(fileDown, "\t", {std::string("+-"), std::to_string(0)}, 5);
			}
		}
	}

	const double kxStep = TWOPI / Lx;
	const double kyStep = TWOPI / Ly;
	// fourier transform
	for (u32 i = 0; i <= Lx / 2; i++) {
		double kx = -PI + i * kxStep;
		for (u32 j = i; j <= Ly / 2; j++) {
			double ky = -PI + j * kyStep;
			printSeparated(fileUp, "\t", {std::string(" G(qx,qy,ti):\n qx ="), std::to_string(i), std::string("qy ="), std::to_string(j)},5);
			printSeparated(fileDown, "\t", {std::string(" G(qx,qy,ti):\n qx ="), std::to_string(i), std::string("qy ="), std::to_string(j)},5);
			for (int tau1 = 0; tau1 < this->M; tau1++)
			{
				cx_double sum_up = 0;
				cx_double sum_down = 0;

				int row = 0;
				int col = 0;
				for (int yy = -Ly + 1; yy < Ly; yy++)
				{
					const u32 yelem = yy + Ly - 1;
					for (int xx = -Lx + 1; xx < Lx; xx++)
					{
						const u32 xelem = xx + Lx - 1;
						auto exponential = std::exp(-im_num * (kx * xx + ky * yy));
						sum_up += exponential * this->g_up_diffs[tau1](xelem, yelem);
						sum_down += exponential * this->g_down_diffs[tau1](xelem, yelem);
					}
				}
				printSeparated(fileUp, "\t", {std::to_string(tau1)}, 3,0);
				printSeparated(fileUp, "\t", {to_string_prec(real(sum_up),width)}, width+5,0);
				printSeparated(fileUp, "\t", {std::string("+-"), std::to_string(0)}, 5);

				printSeparated(fileDown, "\t", {std::to_string(tau1)}, 3,0);
				printSeparated(fileDown, "\t", {to_string_prec(real(sum_down),width)}, width+5,0);
				printSeparated(fileDown, "\t", {std::string("+-"), std::to_string(0)}, 5);
			}
		}
	}

	fileUp.close();
	fileDown.close();
}


// -------------------------------------------------------- SETTERS
/// <summary>
/// Setting the Hubbard - Stratonovich fields
/// </summary>
void hubbard::HubbardModel::set_hs()
{

	for (int i = 0; i < this->Ns; i++) {
		for (int l = 0; l < this->M; l++) {
			//int elem = this->ran.bernoulli(0.5) ? -1 : 1;
			int elem = this->ran.randomReal_uni(0, 1) > 0.5 ? 1 : -1;
			this->hsFields(l, i) = elem;	// set the hs fields to uniform -1 or 1
			//this->hsFields_img[l][i] = elem > 0 ? " " : "|";
		}
	}
}

/// <summary>
/// Sets the directories for saving configurations of Hubbard - Stratonovich fields. It adds /negative/ and /positive/ to dir
/// </summary>
/// <param name="dir">directory to be used for configurations</param>
void hubbard::HubbardModel::setConfDir() {
	this->dir->neg_dir = this->dir->conf_dir + kPSepS + "negative";
	this->dir->pos_dir = this->dir->conf_dir + kPSepS + "positive";
	// create directories
	fs::create_directories(this->dir->neg_dir);
	fs::create_directories(this->dir->pos_dir);
	// add a separator
	this->dir->neg_dir += kPSepS;
	this->dir->pos_dir += kPSepS;
	// for .log files
	std::ofstream fileN, fileP;																	// files for saving the configurations
	this->dir->neg_log = this->dir->neg_dir.substr(0, \
		this->dir->neg_dir.length() - 9) + "negLog," + info + ".dat";							// for storing the labels of negative files in csv for ML
	this->dir->pos_log = this->dir->pos_dir.substr(0, \
		this->dir->pos_dir.length() - 9) + "posLog," + info + ".dat";							// for storing the labels of positive files in csv for ML
	fileN.open(this->dir->neg_log);
	fileP.open(this->dir->pos_log);
	fileN.close();																				// close just to create file neg
	fileP.close();																				// close just to create file pos
}

/// <summary>
/// 
/// </summary>
/// <param name="working_directory"></param>
/// <returns></returns>
void hubbard::HubbardModel::setDirs(std::string working_directory)
{
	using namespace std;
	int Lx = this->lattice->get_Lx();
	int Ly = this->lattice->get_Ly();
	int Lz = this->lattice->get_Lz();
		// -------------------------------------------------------------- file handler ---------------------------------------------------------------
	this->dir->info = this->info;
	this->dir->LxLyLz = "Lx=" + std::to_string(Lx) + ",Ly=" + to_string(Ly) + ",Lz=" + to_string(Lz);

	this->dir->lat_type = this->lattice->get_type() + kPSepS;																// making folder for given lattice type
	this->dir->working_dir = working_directory + this->dir->lat_type + \
		to_string(this->lattice->get_Dim()) + \
		"D" + kPSepS + this->dir->LxLyLz + kPSepS;																		// name of the working directory

	// CREATE DIRECTORIES 
	this->dir->fourier_dir = this->dir->working_dir + "fouriers";
	fs::create_directories(this->dir->fourier_dir);																								// create folder for fourier based parameters
	fs::create_directories(this->dir->fourier_dir + kPSepS + "times");																	// and with different times
	this->dir->fourier_dir += kPSepS;

	this->dir->params_dir = this->dir->working_dir + "params";																					// rea; space based parameters directory
	this->dir->greens_dir = this->dir->working_dir + "greens";																		// greens directory
	fs::create_directories(this->dir->greens_dir);
	this->dir->greens_dir += kPSepS + this->dir->info;
	this->dir->time_greens_dir = this->dir->greens_dir + kPSepS + "times";
	fs::create_directories(this->dir->params_dir);
	fs::create_directories(this->dir->params_dir + kPSepS + "times");
	fs::create_directories(this->dir->greens_dir);
	fs::create_directories(this->dir->time_greens_dir);
	this->dir->greens_dir += kPSepS;
	this->dir->time_greens_dir += kPSepS;
	this->dir->params_dir += kPSepS;

	this->dir->conf_dir = this->dir->working_dir + "configurations" + kPSepS;
	
	// FILES
	this->setConfDir();
	this->dir->setFileNames();
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
	std::tuple< double, double> tmp;								// here we will save the gammas
	if (this->U > 0) {
		// Repulsive case
		bool up = this->hsFields(this->current_time, lat_site) == 1 ? 0 : 1;
		bool down = !up;
		tmp = std::make_tuple(this->gammaExp[u16(up)], this->gammaExp[u16(down)]);

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
	double p_up = 1.0 + (gamma_up) * (1.0 - this->green_up(lat_site, lat_site));;
	double p_down = 0;
	// SPIN DOWN
	if(this->U > 0)
		p_down = 1.0 + (gamma_down) * (1.0 - this->green_down(lat_site, lat_site));
	else
		p_down = 1.0 + (gamma_up) * (1.0 - this->green_down(lat_site, lat_site));
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

	for (int i = 0; i < this->Ns; i++) {
		this->b_mat_up[this->current_time](i,lat_site) *= delta_up;
		this->b_mat_down[this->current_time](i,lat_site) *= delta_down;
		this->b_mat_up_inv[this->current_time](lat_site, i) *= delta_down;
		this->b_mat_down_inv[this->current_time](lat_site, i) *= delta_up;
	}

	//this->b_mat_up[this->current_time].col(lat_site) *= delta_up;				// spin up
	//this->b_mat_down[this->current_time].col(lat_site) *= delta_down;			// spin down
		
	// the names are opposite because we need to divide in principle, not multiply
	//this->b_mat_up_inv[this->current_time].row(lat_site) *= delta_down;			// spin up
	//this->b_mat_down_inv[this->current_time].row(lat_site) *= delta_up;		// spin down

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
	bool checkerboard = false;
	const int Lx = this->lattice->get_Lx();
	const int Ly = this->lattice->get_Ly();

	// USE CHECKERBOARD
	const int dim = this->lattice->get_Dim();
	if (checkerboard && this->getDim() == 2 && Lx == Ly) {
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
		/*arma::mat K(Ns, Ns, arma::fill::zeros);
		for(int x = 0; x < Lx; ++x) {
		    for(int y = 0; y < Ly; ++y) {
		        // chemical potential 'mu' on the diagonal
		        //K(x + Lx * y, x + Lx * y) -= this->mu;
		        K(x + Lx * y, ((x + 1) % Lx) + Lx * y) = this->t[0];
		        K(((x + 1) % Lx) + Lx * y, x + Lx * y) = this->t[0];
		        K(x + Lx * y, x + Lx * ((y + 1) % Lx)) = this->t[0];
		        K(x + Lx * ((y + 1) % Lx), x + Lx * y) = this->t[0];
		    }
		}*/


		//Kx_a.print("Kx a:");
		//Kx_b.print("Kx b:");
		//Ky_a.print("Ky a:");
		//Ky_b.print("Ky b:");
		//this->hopping_exp = Kx_a + Kx_b + Ky_a + Ky_b;
		//(this->hopping_exp - K).print();
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
	else 
	{
	for (int i = 0; i < this->Ns; i++) {
		//this->hopping_exp(i, i) = this->dtau * this->mu;														// diagonal elements
		const int n_of_neigh = this->lattice->get_nn_number(i);												// take number of nn at given site
		for (int j = 0; j < n_of_neigh; j++) {
			const int where_neighbor = this->lattice->get_nn(i, j);											// get given nn
			this->hopping_exp(i, where_neighbor) = this->dtau * this->t[i];								// assign non-diagonal elements
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
	this->hopping_exp = arma::expmat_sym(this->hopping_exp);												// take the exponential
	//this->hopping_exp.print("hopping after exponentiation");

	}
}

/// <summary>
/// Function to calculate the interaction exponential at all times, each column represents the given Trotter time
/// </summary>
void hubbard::HubbardModel::cal_int_exp() {
	const arma::vec dtau_vec = arma::ones(this->Ns) *  this->dtau * (this->mu);
	if (this->U > 0) {
		// Repulsive case
		for (int l = 0; l < this->M; l++) {
			// Trotter times 
			this->int_exp_up.col(l) = arma::exp(dtau_vec + this->hsFields.row(l).t() * (this->lambda));
			this->int_exp_down.col(l) = arma::exp(dtau_vec + this->hsFields.row(l).t() * (-this->lambda));
		}
	}
	else if (U < 0) {
		//const double exp_plus = exp(this->lambda + this->dtau * this->mu);				// plus exponent for faster computation
		//const double exp_minus = exp(-this->lambda + this->dtau * this->mu);			// minus exponent for faster computation
		// Attractive case
		for (int l = 0; l < M; l++) {
			// Trotter times 
			for (int i = 0; i < Ns; i++) {
				// Lattice sites 
				this->int_exp_down.col(l) = arma::exp(dtau_vec + this->hsFields.row(l).t() * this->lambda);
				this->int_exp_up.col(l) = this->int_exp_down.col(l);
			}
		}
	}
	else {
		this->int_exp_down = arma::eye(this->Ns, this->Ns);
		this->int_exp_up = arma::eye(this->Ns, this->Ns);
	}
	//this->int_exp_up.print();
}

/// <summary>
/// Function to calculate all B exponents for a given model. Those are used for the Gibbs weights
/// </summary>
void hubbard::HubbardModel::cal_B_mat() {
//#pragma omp parallel for num_threads(this->inner_threads)
	arma::mat tmp_up(this->Ns, this->Ns, arma::fill::zeros);
	arma::mat tmp_down(this->Ns, this->Ns, arma::fill::zeros);
	for (int l = 0; l < this->M; l++) {
		// Trotter times 
		this->b_mat_down[l] = arma::diagmat(this->int_exp_down.col(l)) * this->hopping_exp;
		this->b_mat_up[l] = arma::diagmat(this->int_exp_up.col(l)) * this->hopping_exp;
		//this->b_mat_down[l] = this->hopping_exp * arma::diagmat(this->int_exp_down.col(l));
		//this->b_mat_up[l] = this->hopping_exp * arma::diagmat(this->int_exp_up.col(l));

		this->b_mat_up_inv[l] = this->b_mat_up[l].i();
		this->b_mat_down_inv[l] = this->b_mat_down[l].i();

		//this->b_mat_down[l] = this->hopping_exp * arma::diagmat(this->int_exp_down.col(l)) * this->hopping_exp;
		//this->b_mat_up[l] = this->hopping_exp * arma::diagmat(this->int_exp_up.col(l)) * this->hopping_exp;
	}
	//b_mat_down[0].print("B_mat_down in t = 0");
}

/// <summary>
/// Function to calculate all B exponents for a given model at a given time. Those are used for the Gibbs weights
/// </summary>
void hubbard::HubbardModel::cal_B_mat(int which_time)
{
	//this->b_mat_down[which_time] = this->hopping_exp * arma::diagmat(this->int_exp_down.col(which_time));
	//this->b_mat_up[which_time] = this->hopping_exp * arma::diagmat(this->int_exp_up.col(which_time));
	this->b_mat_down[which_time] = arma::diagmat(this->int_exp_down.col(which_time)) * this->hopping_exp;
	this->b_mat_up[which_time] = arma::diagmat(this->int_exp_up.col(which_time)) * this->hopping_exp;

	this->b_mat_up_inv[which_time] = this->b_mat_up[which_time].i();
	this->b_mat_down_inv[which_time] = this->b_mat_down[which_time].i();
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
				output << ((this->hsFields(i,j) == 1) ? 0.75 : 0.25) << separator;
			}
		}
		output << "\n";
	}
}


// -------------------------------------------------------- EQUAL TIME AVERAGES

double hubbard::HubbardModel::cal_kinetic_en(int sign, int current_elem_i, const mat& g_up, const mat& g_down) const
{
	const int nei_num = this->lattice->get_nn_number(current_elem_i);
	double Ek = 0;
	for (int nei = 0; nei < nei_num; nei++)
	{
		const int where_neighbor = this->lattice->get_nn(current_elem_i, nei);
		Ek += g_down(current_elem_i, where_neighbor);
		Ek += g_down(where_neighbor, current_elem_i);
		Ek += g_up(current_elem_i, where_neighbor);
		Ek += g_up(where_neighbor, current_elem_i);
	}
	return sign * this->t[current_elem_i] * Ek;
}

double hubbard::HubbardModel::cal_occupation(int sign, int current_elem_i, const mat& g_up, const mat& g_down) const
{
	return (sign * (1.0 - g_down(current_elem_i, current_elem_i)) + sign * (1.0 - g_up(current_elem_i, current_elem_i)));
}

double hubbard::HubbardModel::cal_occupation_corr(int sign, int current_elem_i, int current_elem_j, const mat& g_up, const mat& g_down) const
{
	return sign * ((g_down(current_elem_j, current_elem_i) + g_up(current_elem_j, current_elem_i)));
}

double hubbard::HubbardModel::cal_mz2(int sign, int current_elem_i, const mat& g_up, const mat& g_down) const
{
	return sign * (((1.0 - g_up(current_elem_i, current_elem_i)) * (1.0 - g_up(current_elem_i, current_elem_i)))
		+ ((1.0 - g_up(current_elem_i, current_elem_i)) * (g_up(current_elem_i, current_elem_i)))
		- ((1.0 - g_up(current_elem_i, current_elem_i)) * (1.0 - g_down(current_elem_i, current_elem_i)))
		- ((1.0 - g_down(current_elem_i, current_elem_i)) * (1.0 - g_up(current_elem_i, current_elem_i)))
		+ ((1.0 - g_down(current_elem_i, current_elem_i)) * (1.0 - g_down(current_elem_i, current_elem_i)))
		+ ((1.0 - g_down(current_elem_i, current_elem_i)) * (g_down(current_elem_i, current_elem_i))));
}

double hubbard::HubbardModel::cal_mz2_corr(int sign, int current_elem_i, int current_elem_j, const mat& g_up, const mat& g_down) const
{
	double delta_ij = 0.0L;
	if (current_elem_i == current_elem_j) {
		delta_ij = 1.0L;
	}
	//g_down.print("TEST");
	return sign * (((1.0L - g_up(current_elem_i, current_elem_i)) * (1.0L - g_up(current_elem_j, current_elem_j)))
		+ ((delta_ij - g_up(current_elem_j, current_elem_i)) * (g_up(current_elem_i, current_elem_j)))
		- ((1.0L - g_up(current_elem_i, current_elem_i)) * (1.0L - g_down(current_elem_j, current_elem_j)))
		- ((1.0L - g_down(current_elem_i, current_elem_i)) * (1.0L - g_up(current_elem_j, current_elem_j)))
		+ ((1.0L - g_down(current_elem_i, current_elem_i)) * (1.0L - g_down(current_elem_j, current_elem_j)))
		+ ((delta_ij - g_down(current_elem_j, current_elem_i)) * (g_down(current_elem_i, current_elem_j))));
}

double hubbard::HubbardModel::cal_my2(int sign, int current_elem_i, const mat& g_up, const mat& g_down) const
{
	return 0;
}

double hubbard::HubbardModel::cal_mx2(int sign, int current_elem_i, const mat& g_up, const mat& g_down) const
{
	return sign * (1.0 - g_up(current_elem_i, current_elem_i)) * (g_down(current_elem_i, current_elem_i))
		+ sign * (1.0 - g_down(current_elem_i, current_elem_i)) * (g_up(current_elem_i, current_elem_i));
}

double hubbard::HubbardModel::cal_ch_correlation(int sign, int current_elem_i, int current_elem_j, const mat& g_up, const mat& g_down) const
{
	double delta_ij = 0.0L;
	if (current_elem_i == current_elem_j) {
		delta_ij = 1.0L;
	}
	return sign * (((1 - g_up(current_elem_i, current_elem_i)) * (1 - g_up(current_elem_j, current_elem_j))					//sigma = sigma' = up
		+ (1 - g_down(current_elem_i, current_elem_i)) * (1 - g_down(current_elem_j, current_elem_j))				//sigma = sigma' = down
		+ (1 - g_down(current_elem_i, current_elem_i)) * (1 - g_up(current_elem_j, current_elem_j))					//sigma = down, sigma' = up
		+ (1 - g_up(current_elem_i, current_elem_i)) * (1 - g_down(current_elem_j, current_elem_j))					//sigma = up, sigma' = down
		+ ((delta_ij - g_up(current_elem_j, current_elem_i)) * g_up(current_elem_i, current_elem_j))				//sigma = sigma' = up
		+ ((delta_ij - g_down(current_elem_j, current_elem_i)) * g_down(current_elem_i, current_elem_j))));			//sigma = sigma' = down
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

	if (!quiet && mcSteps != 1) {
#pragma omp critical
		stout << "For: " << this->get_info() << "->\n\t\t\t\tRelax time taken: " << tim_s(start) << " seconds. With sign: " << (pos_num - neg_num)/(1.0*(pos_num + neg_num)) << "\n";
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
#pragma omp critical
	stout << "For: " << this->get_info() << "->\n\t\t\t\tAverages time taken: " << tim_s(start) <<  std::endl;
}








