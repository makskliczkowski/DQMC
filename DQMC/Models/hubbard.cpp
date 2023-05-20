#include "../include/Models/hubbard.h"
#include <execution>
#include <numeric>
#include <utility>

// ################################################# I N I T I A L I Z E R S ######################################################

/*
* @brief initializes the memory for all of the matrices used later
*/
void Hubbard::init()
{
	// set the write lock
	WriteLock lock(this->Mutex);

	// hopping exponent
	this->TExp_.zeros(this->Ns_, this->Ns_);

	// HS transformation fields
	this->HSFields_.ones(this->M_, this->Ns_);

	// all the spin matrices
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; _SPIN_++)
	{
		// Green's matrix
		this->G_		[_SPIN_].zeros(this->Ns_, this->Ns_);
		// interaction
		this->IExp_		[_SPIN_].zeros(this->Ns_, this->M_);
		// propagators
		this->B_		[_SPIN_]	=	v_1d<arma::mat>(this->M_, ZEROM(this->Ns_));
		this->iB_		[_SPIN_]	=	v_1d<arma::mat>(this->M_, ZEROM(this->Ns_));
		this->Bcond_	[_SPIN_]	=	v_1d<arma::mat>(this->p_, ZEROM(this->Ns_));
		// initialize UDT decomposition
		this->udt_		[_SPIN_].reset(new algebra::UDT_QR(this->G_[_SPIN_]));

#ifdef CAL_TIMES

#endif // CAL_TIMES
	}
}

// ################################################## C A L C U L A T O R S #######################################################

/*
* @brief Function to calculate the change in the interaction exponent
* @param _site site on which the change has been made
* @returns A pair for gammas for two spin channels, 0 is spin up, 1 is spin down
*/
auto Hubbard::calGamma(uint _site) -> void
{
	if (this->REPULSIVE_)
		this->currentGamma_ = (this->HSFields_(this->tau_, _site) == 1) ? this->gammaExp_[0] : this->gammaExp_[1];
	else
		this->currentGamma_ = { this->gammaExp_[0][0], this->gammaExp_[0][0] };
}

/*
* @brief Allows to calculate the change in the interaction exponent
*/
auto Hubbard::calDelta() -> spinTuple_
{
	spinTuple_ _out;
	std::transform	(		
						this->currentGamma_.begin(),
						this->currentGamma_.end(),
						_out.begin(),
						[&](auto& i) { return i + 1; }
					);
	return _out;
}

/*
* @brief Return probabilities of spin flip for both spin channels
* @param _site flipping candidate site
* @param gii gammas for both spin channels
* @returns tuple for probabilities on both spin channels, 
* @warning remember, 0 is spin up, 1 is spin down
*/
auto Hubbard::calProba(uint _site) -> spinTuple_
{
	return	{
				1.0		+	currentGamma_[_UP_]	*	(1.0 - this->G_[SPINNUM::_UP_](_site, _site)),
				1.0		+	currentGamma_[_DN_]	*	(1.0 - this->G_[SPINNUM::_DN_](_site, _site))
			};
}

// #################################################### H A M I L T O N I A N ######################################################

/*
* @brief Function to calculate the hopping matrix exponential.
*/
auto Hubbard::calQuadratic() -> void
{
	// cacluate the hopping matrix
	this->TExp_.zeros(this->Ns_, this->Ns_);
	for (int _site = 0; _site < this->Ns_; _site++)
	{
		const auto neiSize			=	this->lat_->get_nn(_site);
		for (int neiNum = 0; neiNum < neiSize; neiNum++) {
			const auto nei			=	this->lat_->get_nn(_site, neiNum);								// get given nn
			this->TExp_(_site, nei) +=	this->dtau_ * this->t_[_site];									// assign non-diagonal elements
		}
	}
#pragma omp critical
	this->TExp_						=	arma::expmat(this->TExp_);
}

/*
* @brief Function to calculate the interaction exponential at all Trotter times, each column represents the given Trotter time.
*/
auto Hubbard::calInteracts() -> void
{
	const arma::Col<double> _dtauVec		=		arma::ones(this->Ns_) * this->dtau_ * (this->mu_);
	if (this->U_ > 0)
		// Repulsive case
		for (int l = 0; l < this->M_; l++) {
			// Trotter times
			this->IExp_[_UP_].col(l)		=		arma::exp(_dtauVec + this->HSFields_.row(l).t() * (	this->lambda_));
			this->IExp_[_DN_].col(l)		=		arma::exp(_dtauVec + this->HSFields_.row(l).t() * (-this->lambda_));
		}
	else if (this->U_ < 0)
		// Attractive case
		for (int l = 0; l < this->M_; l++) {
			// Trotter times
			this->IExp_[_UP_].col(l)		=		arma::exp(_dtauVec + this->HSFields_.row(l).t() * (	this->lambda_));
			this->IExp_[_DN_].col(l)		=		this->IExp_[_UP_].col(l);
		}
	else 
	{
		this->IExp_[_UP_]					=		arma::eye(this->Ns_, this->Ns_);
		this->IExp_[_DN_]					=		arma::eye(this->Ns_, this->Ns_);
	}
}

/*
* @brief Function to calculate all B propagators for a Hubbard model. Those are used for the Gibbs weights.
*/
auto Hubbard::calPropagatB() -> void
{
	for(auto _spin = 0; _spin < this->spinNumber_; _spin++)
		for (int l = 0; l < this->M_; l++) {
			// Trotter times
			this->B_[_spin][l]		=		this->TExp_ * DIAG(this->IExp_[_spin].col(l));
			this->iB_[_spin][l]		=		this->B_[_spin][l].i();
		}
}

/*
* @brief Function to calculate all B propagators for a Hubbard model. Those are used for the Gibbs weights.
* @param _tau Specific Trotter imaginary time
*/
auto Hubbard::calPropagatB(uint _tau) -> void
{
	for (auto _spin = 0; _spin < this->spinNumber_; _spin++) 
	{
		this->B_[_spin][_tau]		=		this->TExp_ * DIAG(this->IExp_[_spin].col(_tau));
		this->iB_[_spin][_tau]		=		this->B_[_spin][_tau].i();
	}
}

/*
* @brief Precalculate the multiplications of B matrices according to M0 stable ones
* @param _sec the sector to calculate the stable multiplication
*/
void Hubbard::calPropagatBC(uint _sec)
{
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; _SPIN_++) {
		auto _time						=		_sec * this->M0_;
		this->Bcond_[_SPIN_][_sec]		=		this->B_[_SPIN_][_time];
		for (int i = _time; i < _time + this->M0_; i++)
			this->Bcond_[_SPIN_][_sec]	=		this->B_[_SPIN_][i] * this->Bcond_[_SPIN_][_sec];
	}
	
}

/*
* Calculate Green with QR decomposition using LOH : doi:10.1016/j.laa.2010.06.023 with premultiplied B matrices.
* For more look into :
* @copydetails "Advancing Large Scale Many-Body QMC Simulations on GPU Accelerated Multicore Systems".
* In order to do that the M_0 and p variables will be used to divide the multiplication into smaller chunks of matrices.
* @param _tau starting time sector - always marks the beginning of the sector
*/
void Hubbard::calGreensFun(uint _tau)
{
	auto _time					= _tau;
	auto _sector				= _tau / this->M0_;

	// decompose the matrices
	this->udt_[SPINNUM::_UP_]->decompose(this->Bcond_[SPINNUM::_UP_][_sector]);
	this->udt_[SPINNUM::_DN_]->decompose(this->Bcond_[SPINNUM::_DN_][_sector]);

	// go through each sector
	for (int i = 1; i < this->p_; i++) {
		_sector++;
		if (_sector == this->p_)
			_sector				= 0;
		this->udt_[SPINNUM::_UP_]->factMult(this->Bcond_[SPINNUM::_UP_][_sector]);
		this->udt_[SPINNUM::_DN_]->factMult(this->Bcond_[SPINNUM::_DN_][_sector]);
	}
	// making two scales for the decomposition following Loh
	this->udt_[SPINNUM::_UP_]->loh_inplace();
	this->udt_[SPINNUM::_DN_]->loh_inplace();

	// save the Green's
	this->udt_[SPINNUM::_UP_]->inv1P(this->G_[SPINNUM::_UP_]);
	this->udt_[SPINNUM::_DN_]->inv1P(this->G_[SPINNUM::_DN_]);
}


// ######################################################## S E T T E R S ###########################################################

/*
* @brief Sets the initial state of the Hubbard-Stratonovich fields
*/
auto Hubbard::setHS(HS_CONF_TYPES _t) -> void
{
	switch (_t)
	{
	case HIGH_T:
		for (int i = 0; i < this->Ns_; i++) 
			for (int l = 0; l < this->M_; l++) 
				this->HSFields_(l, i) = this->ran_.random(0, 1) > 0.5 ? 1 : -1;
		break;
	case LOW_T:
		this->HSFields_.ones(this->M_, this->Ns_);
		break;
	}
}

auto Hubbard::setDir(std::string _m) -> void
{
	return;
}

// ####################################################### U P D A T E R S ##########################################################

/*
* @brief Update the interaction matrix for current spin whenever the given lattice site HS field is changed.
*/
void Hubbard::updInteracts(uint _site, uint _t)
{
	auto _delta							=	this->calDelta();
	for(auto _spin = 0; _spin < this->spinNumber_; _spin++)
		this->IExp_[_spin](_site, _t)	*=	_delta[_spin];
}

/*
* @brief After accepting spin change update the B matrix by multiplying it by the diagonal element (the delta)
* @param _site current lattice site for the update
*/
void Hubbard::updPropagatB(uint _site, uint _t)
{
	auto _delta = this->calDelta();
	for (int i = 0; i < this->Ns_; i++) 
	{
		this->B_[0][_t]	(i, _site)	*=	_delta[0];
		this->B_[1][_t]	(i, _site)	*=	_delta[1];

		this->iB_[0][_t](i, _site)	*=	_delta[1];
		this->iB_[1][_t](i, _site)	*=	_delta[0];
	}
}

// ##################################################### S I M U L A T I O N ########################################################

/*
* @brief heat - bath based algorithm for the propositon of HS field spin flip
* @param _site lattice site at which we try a flip
* @returns sign of the probility
*/
int Hubbard::eqSingleStep(int _site)
{
	this->calGamma(_site);
	auto _probaTuple	=		this->calProba(_site);
	this->proba_		=		std::reduce(_probaTuple.begin(), _probaTuple.end(), 1, std::multiplies<int>());
	this->proba_		*=		this->proba_ / (1.0 + this->proba_);
	const int _sign		=		(this->proba_ >= 0) ? 1 : -1;
	if (this->ran_.bernoulli(proba_) <= _sign * this->proba_)
	{
		this->HSFields_(this->tau_, _site) *= -1;
		this->updPropagatB(_site, this->tau_);
		this->updEqlGreens(_site, _probaTuple);
	}
	return _sign;
}






/*
* @brief Normalise all the averages taken during simulation
* @param avNum number of avs taken
* @param timesNum number of Trotter times used
* @param times if the non-equal time properties were calculated
*/
void hubbard::HubbardModel::av_normalise(int avNum, int timesNum)
{
	const auto normalization = static_cast<double>(avNum * timesNum * this->Ns);						// average points taken
	this->avs->av_sign = (this->pos_num - this->neg_num) / double(this->pos_num + this->neg_num);
	this->avs->sd_sign = variance(static_cast<ld>(avNum), this->avs->av_sign, avNum);
	const double normalisation_sign = normalization * this->avs->av_sign;								// we divide by average sign actually
	// with minus
	//this->avs->av_gr_down /= normalisation_sign / this->Ns;
	//this->avs->av_gr_up /= normalisation_sign / this->Ns;

	this->avs->av_occupation /= normalisation_sign;
	this->avs->sd_occupation = variance(this->avs->sd_occupation, this->avs->av_occupation, normalisation_sign);

	this->avs->av_M2z /= normalisation_sign;
	this->avs->sd_M2z = variance(this->avs->sd_M2z, this->avs->av_M2z, normalisation_sign);
	this->avs->av_M2x /= normalisation_sign;
	this->avs->sd_M2x = variance(this->avs->sd_M2x, this->avs->av_M2x, normalisation_sign);
	// Ek
	this->avs->av_Ek /= normalisation_sign;
	this->avs->sd_Ek = variance(this->avs->av_Ek2, this->avs->av_Ek, normalisation_sign);


	auto [x_num, y_num, z_num] = this->lattice->getNumElems();
	// correlations
	for (int i = 0; i < x_num; i++) {
		for (int j = 0; j < y_num; j++) {
			for (int k = 0; k < z_num; k++) {

				this->avs->av_M2z_corr[i][j][k] /= normalisation_sign;
				this->avs->av_ch2_corr[i][j][k] /= normalisation_sign;
				this->avs->av_occupation_corr[i][j][k] = this->Ns * this->avs->av_occupation_corr[i][j][k] / normalisation_sign;
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

/*
* saves the unequal times Green's functions in a special form
* @param filenum
* @param useWrapping
*/
void hubbard::HubbardModel::save_unequal_greens(int filenum, const vec& signs)
{
	this->avs->normaliseGreens(this->lattice);
	auto [x_num, y_num, z_num] = this->lattice->getNumElems();
	const std::string sign = this->config_sign == 1 ? "+" : "-";

#ifndef SAVE_UNEQUAL_HDF5
	std::string information = " Some version\n\n This is the file that contains real space Green's functions for different times.\n";
	information += " The structure of each is we average over time differences and first row\n";
	information += " before each Green matrix <cicj^+(t1, t2)> is an information about the difference\n";

	std::ofstream fileUp;
	openFile(fileUp, this->dir->time_greens_dir + STR(filenum) + "-up" + sign + this->dir->nameGreensTime);
	std::ofstream fileDown;
	openFile(fileDown, this->dir->time_greens_dir + STR(filenum) + "-down" + sign + this->dir->nameGreensTime);

	fileUp << " Up different time, real-space Greens\n" << information;
	fileDown << " Down different time, real-space Greens\n" << information;
	std::initializer_list<std::string> enter_params = { "n =\t",
		STR(this->lattice->get_Lx()),"\n",
		"l =\t",STR(this->M),"\n",
		"tausk =\t",STR(this->p),"\n",
		"doall =\t",std::string("don't know what is that"),"\n",
		"denswp =\t",STR(BUCKET_NUM),"\n",
		"histn =\t",std::string("don't know what is that"),"\n",
		"iran =\t",std::string("don't know what is that"),"\n",
		"t  =\t",STRP(this->t[0],5),"\n",
		"mu =\t",STRP(this->mu, 5),"\n",
		"delmu =\t",std::string("don't know what is that"),"\n",
		"bpar  =\t",std::string("don't know what is that"),"\n",
		"dtau = \t",STRP(this->dtau,5),"\n",
		"warms  =\t",STR(1000),		"\n",
		"sweeps =\t",STR(2000),"\n",
		"u =\t",STRP(this->U,5),"\n",
		"nwrap =\t",STR(this->M_0),"\n",
		"difflim =\t",std::string("don't know what is that"),"\n",
		"errrat =\t",std::string("don't know what is that"),	"\n",
		"doauto = \t0","\n",
		"orthlen =\t",std::string("don't know what is that"),"\n",
		"eorth =\t",std::string("don't know what is that"),"\n",
		"dopair =\t",std::string("don't know what is that"),"\n",
		"numpair =\t",std::string("don't know what is that"),"\n",
		"lambda=\t",STRP(this->lambda,4),"\n",
		"start = \t0", "\n",
		"signs=\n" };

	printSeparated(fileUp, ' ', enter_params, 30);
	printSeparated(fileDown, ' ', enter_params, 30);
	fileUp << signs.t() << "\n\n";
	fileDown << signs.t() << "\n\n";
	const u16 width = 12;
	printSeparated(fileUp, '\t', { std::string(" G(nx,ny,ti):") });
	printSeparated(fileDown, '\t', { std::string(" G(nx,ny,ti):") });

	for (int nx = 0; nx < x_num; nx++) {
		for (int ny = 0; ny < y_num; ny++) {
			auto [x, y, z] = this->lattice->getSymPosInv(nx, ny, 0);
			printSeparated(fileUp, '\t', 6, true, VEQ(x), VEQ(y));
			printSeparated(fileDown, '\t', 6, true, VEQ(x), VEQ(y));
			for (int tau1 = 0; tau1 < this->M; tau1++)
			{
				printSeparated(fileUp, '\t', 4, false, tau1);
				printSeparated(fileUp, '\t', width + 5, false, STRP(this->avs->g_up_diffs[tau1](nx, ny), width));
				printSeparated(fileUp, '\t', 5, false, "+-");
				printSeparated(fileUp, '\t', width + 5, true, STRP(this->avs->sd_g_up_diffs[tau1](nx, ny), width));

				printSeparated(fileDown, '\t', 4, false, tau1);
				printSeparated(fileDown, '\t', width + 5, false, STRP(this->avs->g_down_diffs[tau1](nx, ny), width));
				printSeparated(fileDown, '\t', 5, false, "+-");
				printSeparated(fileDown, '\t', width + 5, true, STRP(this->avs->sd_g_down_diffs[tau1](nx, ny), width));
			}
		}
	}
	fileUp.close();
	fileDown.close();
#else
	for (int tau = 0; tau < this->M; tau++) {
		this->tempGreen_down = (this->avs->g_up_diffs[tau] + this->avs->g_down_diffs[tau]) / 2.0;
		this->tempGreen_down.save(arma::hdf5_name(this->dir->time_greens_dir + STR(filenum) + "_" + sign + "_" + this->dir->nameGreensTimeH5,
			STR(tau), arma::hdf5_opts::append));
	}
#endif
}

//! -------------------------------------------------------- SETTERS


/*
* Sets the directories for saving configurations of Hubbard - Stratonovich fields. It adds /negative/ and /positive/ to dir
* @param dir directory to be used for configurations
*/
//void hubbard::HubbardModel::setConfDir() {
//	this->dir->neg_dir = this->dir->conf_dir + kPS + this->info;
//	this->dir->pos_dir = this->dir->conf_dir + kPS + this->info;
//	// create directories
//
//	this->dir->neg_dir += kPS + "negative";
//	this->dir->pos_dir += kPS + "positive";
//
//	fs::create_directories(this->dir->neg_dir);
//	fs::create_directories(this->dir->pos_dir);
//
//	// add a separator
//	this->dir->neg_dir += kPS;
//	this->dir->pos_dir += kPS;
//	// for .log files
//	std::ofstream fileN, fileP;																	// files for saving the configurations
//	this->dir->neg_log = this->dir->neg_dir.substr(0, \
//		this->dir->neg_dir.length() - 9) + "negLog," + info + ".dat";							// for storing the labels of negative files in csv for ML
//	this->dir->pos_log = this->dir->pos_dir.substr(0, \
//		this->dir->pos_dir.length() - 9) + "posLog," + info + ".dat";							// for storing the labels of positive files in csv for ML
//	fileN.open(this->dir->neg_log);
//	fileP.open(this->dir->pos_log);
//	fileN.close();																				// close just to create file neg
//	fileP.close();																				// close just to create file pos
//}

/*
* @brief setting the model directories
* @param working_directory current working directory
*/
//void hubbard::HubbardModel::setDirs(std::string working_directory)
//{
//	using namespace std;
//	int Lx = this->lattice->get_Lx();
//	int Ly = this->lattice->get_Ly();
//	int Lz = this->lattice->get_Lz();
//
//	// set the unique token for file names
//	const auto token = clk::now().time_since_epoch().count();
//	this->dir->token = STR(token % this->ran.randomInt_uni(0, 1e6));
//
//	// -------------------------------------------------------------- file handler ---------------------------------------------------------------
//	this->dir->info = this->info;
//	this->dir->LxLyLz = "Lx=" + STR(Lx) + ",Ly=" + STR(Ly) + ",Lz=" + STR(Lz);
//
//	this->dir->lat_type = this->lattice->get_type() + kPS;																// making folder for given lattice type
//	this->dir->working_dir = working_directory + this->dir->lat_type + \
//		STR(this->lattice->get_Dim()) + \
//		"D" + kPS + this->dir->LxLyLz + kPS;																		// name of the working directory
//
//	// CREATE DIRECTORIES
//	this->dir->fourier_dir = this->dir->working_dir + "fouriers";
//	fs::create_directories(this->dir->fourier_dir);																								// create folder for fourier based parameters
//	fs::create_directories(this->dir->fourier_dir + kPS + "times");																	// and with different times
//	this->dir->fourier_dir += kPS;
//
//	this->dir->params_dir = this->dir->working_dir + "params";																					// rea; space based parameters directory
//	this->dir->greens_dir = this->dir->working_dir + "greens";																		// greens directory
//	fs::create_directories(this->dir->greens_dir);
//	this->dir->greens_dir += kPS + this->dir->info;
//	this->dir->time_greens_dir = this->dir->greens_dir + kPS + "times";
//	fs::create_directories(this->dir->params_dir + kPS + "times");
//	fs::create_directories(this->dir->time_greens_dir);
//	this->dir->greens_dir += kPS;
//	this->dir->time_greens_dir += kPS;
//	this->dir->params_dir += kPS;
//
//	this->dir->conf_dir = this->dir->working_dir + "configurations" + kPS;
//
//	// FILES
//	this->setConfDir();
//	this->dir->setFileNames();
//}

// -------------------------------------------------------- UPDATERS --------------------------------------------------------



//! -------------------------------------------------------- GETTERS

//! -------------------------------------------------------- CALCULATORS

// TODO ------------------------>

//void hubbard::HubbardModel::cal_hopping_exp()
//{
//	bool checkerboard = false;
//	const int Lx = this->lattice->get_Lx();
//	const int Ly = this->lattice->get_Ly();
//
//	// USE CHECKERBOARD
//	const int dim = this->lattice->get_Dim();
//	if (checkerboard && this->getDim() == 2 && Lx == Ly) {
//		arma::mat Kx_a, Kx_b, Ky_a, Ky_b, Kz_a, Kz_b;
//		Kx_a.zeros(this->Ns, this->Ns);
//		Kx_b = Kx_a;
//		if (dim >= 2) {
//			// 2D
//			Ky_a = Kx_a;
//			Ky_b = Ky_a;
//			if (dim == 3) {
//				// 3D
//				Kz_a = Kx_a;
//				Kz_b = Kz_a;
//			}
//		}
//		// set elements
//		for (int i = 0; i < this->Ns; i++) {
//			const int n_of_neigh = this->lattice->get_nn_number(i);												// take number of nn at given site
//			for (int j = 0; j < n_of_neigh; j++) {
//				const int where_neighbor = this->lattice->get_nn(i, j);											// get given nn
//				const int y = i / Lx;
//				const int x = i - y * Lx;
//				const int y_nei = where_neighbor / Ly;
//				const int x_nei = where_neighbor - y_nei * Lx;
//				if (y_nei == y) {
//					// even rows
//					if (i % 2 == 0) {
//						if (x_nei == (x + 1) % Lx) {
//							Kx_a(i, where_neighbor) = 1;
//						}
//						else {
//							Kx_b(i, where_neighbor) = 1;
//						}
//					}
//					// odd rows
//					else {
//						if (x_nei == (x + 1) % Lx) {
//							Kx_b(i, where_neighbor) = 1;
//						}
//						else {
//							Kx_a(i, where_neighbor) = 1;
//						}
//					}
//				}
//				else {
//					// ky
//					if (where_neighbor % 2 == 0) {
//						Ky_a(i, where_neighbor) = 1;
//						Ky_a(where_neighbor, i) = 1;
//					}
//					else {
//						Ky_b(i, where_neighbor) = 1;
//						Ky_b(where_neighbor, i) = 1;
//					}
//				}
//			}
//		}
//		/*arma::mat K(Ns, Ns, arma::fill::zeros);
//		for(int x = 0; x < Lx; ++x) {
//			for(int y = 0; y < Ly; ++y) {
//				// chemical potential 'mu' on the diagonal
//				//K(x + Lx * y, x + Lx * y) -= this->mu;
//				K(x + Lx * y, ((x + 1) % Lx) + Lx * y) = this->t[0];
//				K(((x + 1) % Lx) + Lx * y, x + Lx * y) = this->t[0];
//				K(x + Lx * y, x + Lx * ((y + 1) % Lx)) = this->t[0];
//				K(x + Lx * ((y + 1) % Lx), x + Lx * y) = this->t[0];
//			}
//		}*/
//
//		//Kx_a.print("Kx a:");
//		//Kx_b.print("Kx b:");
//		//Ky_a.print("Ky a:");
//		//Ky_b.print("Ky b:");
//		//this->hopping_exp = Kx_a + Kx_b + Ky_a + Ky_b;
//		//(this->hopping_exp - K).print();
//		//this->hopping_exp.print("HOPPING MATRIX:");
//
//		//arma::mat tmp_exp = arma::expmat(this->hopping_exp);
//		//tmp_exp.print("NORMALLY CALCULATED EXPONENT");
//
//		arma::mat one = arma::eye(this->Ns, this->Ns);
//		one *= cosh(this->dtau * t[0]);
//		const double sinus = sinh(this->dtau * this->t[0]);
//
//		Kx_a = (Kx_a * sinus + one);
//		Kx_b = (Kx_b * sinus + one);
//		Ky_a = (Ky_a * sinus + one);
//		Ky_b = (Ky_b * sinus + one);
//		this->hopping_exp = Ky_a * Kx_a * Ky_b * Kx_b;
//		//this->hopping_exp.print("BETTER CALCULATED EXP");
//		return;
//	}
//	else
//	{
//		for (int i = 0; i < this->Ns; i++) {
//			//this->hopping_exp(i, i) = this->dtau * this->mu;														// diagonal elements
//			const auto n_of_neigh = this->lattice->get_nn_number(i);											// take number of nn at given site
//			for (int j = 0; j < n_of_neigh; j++) {
//				const int where_neighbor = this->lattice->get_nn(i, j);											// get given nn
//				this->hopping_exp(i, where_neighbor) = this->dtau * this->t[0];									// assign non-diagonal elements
//			}
//		}
//		//this->hopping_exp.print("hopping before exponentiation");
//		//arma::vec eigval;
//		//arma::mat eigvec;
//		//arma::eig_sym(eigval, eigvec, this->hopping_exp);
//		//stout << "eigenvalues:\n" << eigval.t() << std::endl;
//		//arma::mat jordan = eigvec.i() * this->hopping_exp * eigvec;
//		//jordan = arma::expmat_sym(jordan);
//		//this->hopping_exp = eigvec * this->hopping_exp * eigvec.i();
//#pragma omp critical
//		this->hopping_exp = arma::expmat(this->hopping_exp);													// take the exponential
//		//this->hopping_exp.print("hopping after exponentiation");
//	}
//}



//! -------------------------------------------------------- PRINTERS --------------------------------------------------------
// TODO ----------->
/*
* TODO
* @param output
* @param which_time_caused
* @param which_site_caused
* @param this_site_spin
* @param separator
*/
//void hubbard::HubbardModel::print_hs_fields(std::string separator) const
//{
//	std::ofstream file_conf, file_log;														// savefiles
//	std::string name_conf, name_log;														// filenames to save
//	if (this->config_sign < 0) {
//		name_conf = this->dir->neg_dir + "neg_" + this->info + \
//			",n=" + STR(this->neg_num) + ".dat";
//		name_log = this->dir->neg_log;
//	}
//	else {
//		name_conf = this->dir->pos_dir + "pos_" + this->info + \
//			",n=" + STR(this->pos_num) + ".dat";
//		name_log = this->dir->pos_log;
//	}
//	// open files
//	openFile(file_log, name_log, ios::app);
//	openFile(file_conf, name_conf);
//	printSeparated(file_log, ',', { name_conf, str_p(this->probability, 4), STR(this->config_sign) }, 26);
//
//	for (int i = 0; i < this->M; i++) {
//		for (int j = 0; j < this->Ns; j++) {
//			file_conf << (this->hsFields(i, j) > 0 ? 1 : 0) << separator;
//		}
//		file_conf << "\n";
//	}
//	file_conf.close();
//	file_log.close();
//}

/*
*
* @param separator
* @param toPrint
*/
//void hubbard::HubbardModel::print_hs_fields(std::string separator, const arma::mat& toPrint) const
//{
//	std::ofstream file_conf, file_log;														// savefiles
//	std::string name_config = "", name_log = "";												// filenames to save
//	if (this->config_sign < 0) {
//		name_config = this->dir->neg_dir + "neg_" + this->info + ",n=" + STR(this->neg_num) + ".dat";
//		name_log = this->dir->neg_log;
//	}
//	else {
//		name_config = this->dir->pos_dir + "pos_" + this->info + ",n=" + STR(this->pos_num) + ".dat";
//		name_log = this->dir->pos_log;
//	}
//	// open files
//	openFile(file_log, name_log, ios::app);
//	openFile(file_conf, name_config);
//	printSeparated(file_log, ',', { name_config, str_p(this->probability, 4), STR(this->config_sign) }, 26);
//
//	for (int i = 0; i < this->M; i++) {
//		for (int j = 0; j < this->Ns; j++) {
//			file_conf << (toPrint(i, j) > 0 ? 1 : 0) << separator;
//		}
//		file_conf << "\n";
//	}
//	file_conf.close();
//	file_log.close();
//}

// -------------------------------------------------------- EQUAL TIME AVERAGES --------------------------------------------------------

double hubbard::HubbardModel::cal_kinetic_en(int sign, int current_elem_i, const mat& g_up, const mat& g_down)
{
	const auto nei_num = this->lattice->get_nn_number(current_elem_i);
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

double hubbard::HubbardModel::cal_occupation(int sign, int current_elem_i, const mat& g_up, const mat& g_down)
{
	return (sign * (1.0 - g_down(current_elem_i, current_elem_i)) + sign * (1.0 - g_up(current_elem_i, current_elem_i)));
}

double hubbard::HubbardModel::cal_occupation_corr(int sign, int current_elem_i, int current_elem_j, const mat& g_up, const mat& g_down)
{
	return sign * ((g_down(current_elem_j, current_elem_i) + g_up(current_elem_j, current_elem_i)));
}

double hubbard::HubbardModel::cal_mz2(int sign, int current_elem_i, const mat& g_up, const mat& g_down)
{
	return sign * (((1.0 - g_up(current_elem_i, current_elem_i)) * (1.0 - g_up(current_elem_i, current_elem_i)))
		+ ((1.0 - g_up(current_elem_i, current_elem_i)) * (g_up(current_elem_i, current_elem_i)))
		- ((1.0 - g_up(current_elem_i, current_elem_i)) * (1.0 - g_down(current_elem_i, current_elem_i)))
		- ((1.0 - g_down(current_elem_i, current_elem_i)) * (1.0 - g_up(current_elem_i, current_elem_i)))
		+ ((1.0 - g_down(current_elem_i, current_elem_i)) * (1.0 - g_down(current_elem_i, current_elem_i)))
		+ ((1.0 - g_down(current_elem_i, current_elem_i)) * (g_down(current_elem_i, current_elem_i))));
}

double hubbard::HubbardModel::cal_mz2_corr(int sign, int current_elem_i, int current_elem_j, const mat& g_up, const mat& g_down)
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

double hubbard::HubbardModel::cal_my2(int sign, int current_elem_i, const mat& g_up, const mat& g_down)
{
	return 0;
}

double hubbard::HubbardModel::cal_mx2(int sign, int current_elem_i, const mat& g_up, const mat& g_down)
{
	return sign * (1.0 - g_up(current_elem_i, current_elem_i)) * (g_down(current_elem_i, current_elem_i))
		+ sign * (1.0 - g_down(current_elem_i, current_elem_i)) * (g_up(current_elem_i, current_elem_i));
}

double hubbard::HubbardModel::cal_ch_correlation(int sign, int current_elem_i, int current_elem_j, const mat& g_up, const mat& g_down)
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

/*
* @brief Equilivrate the simulation
* @param algorithm type of equilibration algorithm
* @param mcSteps Number of Monte Carlo steps
* @param conf Shall print configurations?
* @param quiet Shall be quiet?
*/
void hubbard::HubbardModel::relaxation(impDef::algMC algorithm, int mcSteps, bool conf, bool quiet)
{
	auto start = std::chrono::high_resolution_clock::now();											// starting timer for averages
	this->equalibrate = false;
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
		stout << "For: " << this->get_info() << "->\n\t\t\t\tRelax time taken: " << tim_s(start) << " seconds. With sign: " << (pos_num - neg_num) / (1.0 * (pos_num + neg_num)) << "\n";
	}
}

/*
* Collect the averages from the simulation
* @param algorithm type of equilibration algorithm
* @param corr_time how many times to wait for correlations breakout
* @param avNum number of averages to take
* @param bootStraps Number of bootstraps - NOT IMPLEMENTED
* @param quiet shall be quiet?
*/
void hubbard::HubbardModel::average(impDef::algMC algorithm, int corr_time, int avNum, int bootStraps, bool quiet)
{
	auto start = std::chrono::high_resolution_clock::now();											// starting timer for averages
	this->equalibrate = false;
	//this->cal_B_mat();
	switch (algorithm)
	{
	case impDef::algMC::heat_bath:
		this->heat_bath_av(corr_time, avNum, quiet);
		break;
	default:
		std::cout << "Didn't choose the algorithm type\n";
		exit(-1);
		break;
	}
#pragma omp critical
	stout << "For: " << this->get_info() << "->\n\t\t\t\tAverages time taken: " << tim_s(start) << std::endl;
}