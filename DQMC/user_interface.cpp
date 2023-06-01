#include "include/user_interface.h"
int LASTLVL = 0;

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief model parser
* @param argc number of line arguments
* @param argv line arguments
*/
void UI::parseModel(int argc, cmdArg& argv)
{
	// --------- HELP
	if (std::string option = this->getCmdOption(argv, "-hlp"); option != "")
		this->exitWithHelp();

	// set default at first
	this->setDefault();

	std::string choosen_option = "";

	// -------------------- SIMULATION PARAMETERS --------------------
	SETOPTION(		simP, mcS			);
	SETOPTION(		simP, mcA			);
	SETOPTION(		simP, mcC			);
	// ---------- LATTICE ----------
	SETOPTIONV(		latP, typ, "l"		);
	SETOPTIONV(		latP, dim, "d"		);
	SETOPTION(		latP, Lx			);
	SETOPTION(		latP, Ly			);
	SETOPTION(		latP, Lz			);
	SETOPTION(		latP, bc			);
	if (!this->defineLattice())
		throw std::runtime_error("Couldn't create a lattice\n");

	// ---------- MODEL ----------

	// model type
	SETOPTIONV(		modP, modTyp, "mod"	);
	// --- Hubbard ---
	SETOPTION(		modP, U				);
	SETOPTION(		modP, dtau			);
	SETOPTION(		modP, beta			);
	SETOPTION(		modP, mu			);
	SETOPTION(		modP, M0			);
	this->modP.Ns_			=			this->latP.lat->get_Ns();
	this->modP.T_			=			1.0 / this->modP.beta_;
	this->modP.M_			=			this->modP.beta_ / this->modP.dtau_;
	this->modP.t_			=			v_1d<double>(this->modP.Ns_, this->modP.t_[0]);
	// ---------- OTHERS
	this->setOption(this->quiet		, argv, "q"	);
	this->setOption(this->threadNum	, argv, "th"	);

	// later function choice
	this->setOption(this->chosenFun	, argv, "fun"	);

	//---------- DIRECTORY

	bool setDir		[[maybe_unused]] =	this->setOption(this->mainDir, argv, "dir");
	this->mainDir	=	fs::current_path().string() + kPS + "DATA" + kPS + this->mainDir + kPS;

	// create the directories
	createDir(this->mainDir);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief  Setting parameters to default.
*/
void UI::setDefault()
{
	// lattice stuff
	this->latP.setDefault();

	// define basic model
	this->modP.setDefault();

	// others 
	this->threadNum = 1;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief chooses the method to be used later based on input -fun argument
*/
void UI::funChoice()
{
	LOGINFO_CH_LVL(0);
	switch (this->chosenFun)
	{
	case -1:
		// default case of showing the help
		this->exitWithHelp();
		break;
	case 11:
		// this option utilizes the Hamiltonian with NQS ansatz calculation
		LOGINFO("SIMULATION: HAMILTONIAN WITH DQMC QR", LOG_TYPES::CHOICE, 1);
		this->makeSim();
		break;
	default:
		// default case of showing the help
		this->exitWithHelp();
		break;
	}
	LOGINFO("USING #THREADS=" + STR(this->threadNum), LOG_TYPES::CHOICE, 1);

}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Depending on the lattice type, define it!
*/
bool UI::defineLattice()
{
	switch (this->latP.typ_)
	{
	case LatticeTypes::SQ:
		this->latP.lat = std::make_shared<SquareLattice>(this->latP.Lx_, this->latP.Ly_, this->latP.Lz_,
			this->latP.dim_, this->latP.bc_);
		break;
	case LatticeTypes::HEX:
		this->latP.lat = std::make_shared<HexagonalLattice>(this->latP.Lx_, this->latP.Ly_, this->latP.Lz_,
			this->latP.dim_, this->latP.bc_);
		break;
	default:
		this->latP.lat = std::make_shared<SquareLattice>(this->latP.Lx_, this->latP.Ly_, this->latP.Lz_,
			this->latP.dim_, this->latP.bc_);
		break;
	};
	return true;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bool UI::defineModels(bool _createLat)
{
	// create lattice
	if (_createLat && !this->latP.lat)
		this->defineLattice();

	return this->defineModel();
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bool UI::defineModel()
{
	switch (this->modP.modTyp_)
	{
	case MY_MODELS::HUBBARD_M:
		this->mod_s2_	= std::make_shared<Hubbard>(this->modP.T_, this->latP.lat, this->modP.M_, this->modP.M0_,
													this->modP.t_, this->modP.U_, this->modP.dtau_);
		break;
	default:
		this->mod_s2_ = std::make_shared<Hubbard>(this->modP.T_, this->latP.lat, this->modP.M_, this->modP.M0_,
			this->modP.t_, this->modP.U_, this->modP.dtau_);
		break;
	}
	return true;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void UI::makeSim()
{
	BEGIN_CATCH_HANDLER
	if (!this->defineModels(true))
		return;
	this->mod_s2_->setDir(this->mainDir);
	this->mod_s2_->relaxes(this->simP.mcS_, this->quiet);
	LOGINFO(LOG_TYPES::TIME, 1);
	this->mod_s2_->average(this->simP.mcS_, this->simP.mcC_, this->simP.mcA_, 1, this->quiet);
	LOGINFO(LOG_TYPES::TIME, 1);
	this->mod_s2_->saveAverages();
	END_CATCH_HANDLER(std::string(__FUNCTION__));
}
// -------------------------------------------------------- PARSERS

///*
//* @brief model parser
//* @param argc number of line arguments
//* @param argv line arguments
//*/
//void hubbard::ui::parseModel(int argc, const v_1d<std::string>& argv)
//{
//	this->set_default();
//
//	std::string choosen_option = "";
//
//	//---------- SIMULATION PARAMETERS
//	// monte carlo steps
//	choosen_option = "-m";
//	this->set_option(this->mcSteps, argv, choosen_option);
//	// dimension
//	choosen_option = "-d";
//	this->set_option(this->dim, argv, choosen_option, false);
//	if (this->dim >= 3 || this->dim < 1)
//		this->set_default_msg(this->dim, choosen_option.substr(1), \
//			"Wrong dimmension\n", default_params);
//	// correlation time
//	choosen_option = "-c";
//	this->set_option(this->corrTime, argv, choosen_option);
//	// number of averages
//	choosen_option = "-a";
//	this->set_option(this->avsNum, argv, choosen_option);
//	// Trotter subintervals
//	choosen_option = "-m0";
//	this->set_option(this->M_0, argv, choosen_option);
//	// ---------- Trotter time difference
//	choosen_option = "-dt";
//	this->set_option(this->dtau, argv, choosen_option);
//	// Trotter time differences number
//	choosen_option = "-dtn";
//	this->set_option(this->dtau_num, argv, choosen_option);
//	// Trotter time differences step
//	choosen_option = "-dts";
//	this->set_option(this->dtau_step, argv, choosen_option);
//	// ---------- beta
//	choosen_option = "-b";
//	this->set_option(this->beta, argv, choosen_option);
//	// beta step
//	choosen_option = "-bs";
//	this->set_option(this->beta_step, argv, choosen_option);
//	// betas number
//	choosen_option = "-bn";
//	this->set_option(this->beta_num, argv, choosen_option);
//	// ---------- U
//	choosen_option = "-u";
//	this->set_option(this->U, argv, choosen_option, false);
//	// U step
//	choosen_option = "-us";
//	this->set_option(this->U_step, argv, choosen_option, false);
//	// U number
//	choosen_option = "-un";
//	this->set_option(this->U_num, argv, choosen_option);
//	// ---------- mu
//	choosen_option = "-mu";
//	this->set_option(this->mu, argv, choosen_option, false);
//	// mu step
//	choosen_option = "-mus";
//	this->set_option(this->mu_step, argv, choosen_option, false);
//	// mu number
//	choosen_option = "-mun";
//	this->set_option(this->mu_num, argv, choosen_option);
//	// ---------- LATTICE PARAMETERS
//	// lx
//	choosen_option = "-lx";
//	this->set_option(this->lx, argv, choosen_option);
//	// lx_step
//	choosen_option = "-lxs";
//	this->set_option(this->lx_step, argv, choosen_option);
//	// lx_num
//	choosen_option = "-lxn";
//	this->set_option(this->lx_num, argv, choosen_option);
//	// ly
//	choosen_option = "-ly";
//	this->set_option(this->ly, argv, choosen_option);
//	// ly_step
//	choosen_option = "-lys";
//	this->set_option(this->ly_step, argv, choosen_option);
//	// ly_num
//	choosen_option = "-lyn";
//	this->set_option(this->ly_num, argv, choosen_option);
//	// lz
//	choosen_option = "-lz";
//	this->set_option(this->lz, argv, choosen_option);
//	// lz_step
//	choosen_option = "-lzs";
//	this->set_option(this->lz_step, argv, choosen_option);
//	// lz_num
//	choosen_option = "-lzn";
//	this->set_option(this->lz_num, argv, choosen_option);
//
//	// double T = 1/this->beta;
//	int Ns = this->lx * this->ly * this->lz;
//	//---------- OTHERS
//	// quiet
//	choosen_option = "-q";
//	this->set_option(this->quiet, argv, choosen_option, false);
//	// outer thread number
//	choosen_option = "-th";
//	this->set_option(this->outer_threads, argv, choosen_option, false);
//	if (this->outer_threads <= 0 || this->outer_threads * this->inner_threads > this->thread_number)
//		this->set_default_msg(this->outer_threads, choosen_option.substr(1), \
//			"Wrong number of threads\n", default_params);
//	// inner thread number
//	choosen_option = "-ti";
//	this->set_option(this->inner_threads, argv, choosen_option, false);
//	if (this->inner_threads <= 0 || this->outer_threads * this->inner_threads > this->thread_number)
//		this->set_default_msg(this->inner_threads, choosen_option.substr(1), \
//			"Wrong number of threads\n", default_params);
//	// qr
//	choosen_option = "-hs";
//	this->set_option(this->useHirsh, argv, choosen_option, false);
//	// calculate different times
//	choosen_option = "-ct";
//	this->set_option(this->cal_times, argv, choosen_option, false);
//	// save configurations
//	choosen_option = "-cg";
//	this->set_option(this->save_conf, argv, choosen_option, false);
//	// self learning
//	choosen_option = "-sf";
//	this->set_option(this->sf, argv, choosen_option, false);
//	if (this->sf < 0 || this->sf > 2)
//		this->set_default_msg(this->sf, choosen_option.substr(1), \
//			"Wrong input for self learning\n", default_params);
//	// self learning number
//	choosen_option = "-sfn";
//	this->set_option(this->sfn, argv, choosen_option, false);
//	// hopping coefficients
//	choosen_option = "-t";
//	this->set_option(this->t_fill, argv, choosen_option, false);
//	this->t = std::vector<double>(Ns, t_fill);
//	// lattice type
//	choosen_option = "-l";
//	this->set_option(this->lattice_type, argv, choosen_option, false);
//
//
//	// get help
//	choosen_option = "-h";
//	if (std::string option = this->getCmdOption(argv, choosen_option); option != "")
//		exit_with_help();
//
//	bool set_dir = false;
//	choosen_option = "-dir";
//	if (std::string option = this->getCmdOption(argv, choosen_option); option != "") {
//		this->set_option(this->saving_dir, argv, choosen_option, false);
//		set_dir = true;
//	}
//	if (!set_dir)
//		this->saving_dir = fs::current_path().string() + kPS + "results" + kPS;
//
//	fs::create_directories(this->saving_dir);
//	//std::string folder = "." + kPS + "results" + kPS;
//	//if (!argv[argc - 1].empty() && argc % 2 != 0) {
//	//	// only if the last command is non-even
//	//	folder = argv[argc - 1];
//	//	if (!fs::create_directories(folder))											// creating the directory for saving the files with results
//	//		this->saving_dir = folder;																// if can create dir this is is
//	//}
//	//else {
//	//	this->saving_dir = folder;																	// if can create dir this is is
//	//}
//	//omp_set_num_threads(outer_threads);
//	omp_set_num_threads(outer_threads * inner_threads);
//}

/// <summary>
///
/// </summary>
//void hubbard::ui::make_simulation()
//{
//	stout << "STARTING THE SIMULATION AND USING OUTER THREADS = " << outer_threads << ", INNER THREADS = " << inner_threads << std::endl;
//
//	// save the log file
//#pragma omp single
//	{
//		std::fstream fileLog(this->saving_dir + "HubbardLog.csv", std::ios::in | std::ios::app);
//		fileLog.seekg(0, std::ios::end);
//		if (fileLog.tellg() == 0) {
//			fileLog.clear();
//			fileLog.seekg(0, std::ios::beg);
//			printSeparated(fileLog, ',', 20, true, "lattice_type", "mcsteps", "avsNum", "corrTime", "M", "M0", "dtau", "Lx", \
//				"Ly", "Lz", "beta", "U", "mu", "occ", "sd(occ)", "av_sgn", "sd(sgn)", "Ekin", \
//				"sd(Ekin)", "m^2_z", "sd(m^2_z)", "m^2_x", "time taken", "token");
//		}
//		fileLog.close();
//	}
//
//
//	v_1d<HubbardParams> paramList;
//
//	for (int bi = 0; bi < this->beta_num; bi++) {
//		// over different betas
//		for (int ui = 0; ui < this->U_num; ui++) {
//			// over interactions
//			for (int mui = 0; mui < this->mu_num; mui++) {
//				// over chemical potentials
//				for (int Li = 0; Li < this->lx_num; Li++) {
//					// over lattice sizes (currently square)
//					for (int dtaui = 0; dtaui < this->dtau_num; dtaui++) {
//						// PARAMS LOCALLY FOR THREADS
//						auto Lx_i = this->lx + Li * this->lx_step;
//						auto Ly_i = this->ly + Li * this->ly_step;
//						auto Lz_i = 1;
//
//						auto beta_i = this->beta + bi * this->beta_step;
//						auto U_i = this->U + ui * this->U_step;
//						auto mu_i = this->mu + mui * mu_step;
//						auto dtau_i = this->dtau + dtaui * this->dtau_step;
//						//double dtaui = dtau;//sqrt(0.125 / Ui);
//						paramList.push_back(HubbardParams(dim, beta_i, mu_i, U_i, Lx_i, Ly_i, Lz_i, dtau_i, 1, this->M_0, 1));
//
//					}
//				}
//			}
//		}
//	}
//
//#pragma omp parallel for num_threads(this->outer_threads)
//	for (int i = 0; i < paramList.size(); i++) {
//		paramList[i].M = static_cast<int>(1.0 * paramList[i].beta / paramList[i].dtau);
//		paramList[i].p = std::ceil(1.0 * paramList[i].M / paramList[i].M0);
//		paramList[i].M = paramList[i].p * paramList[i].M0;
//		paramList[i].dtau = 1.0 * paramList[i].beta / paramList[i].M;
//		collectAvs(paramList[i]);
//	}
//
//	std::cout << "FINISHED EVERY THREAD" << std::endl;
//}

// -------------------------------------------------------- HELPERS


//void hubbard::ui::collectRealSpace(std::string name_times, std::string name, const HubbardParams& params, std::shared_ptr<averages_par> avs, std::shared_ptr<Lattice> lat)
//{
//	using namespace std;
//	std::ofstream fileP, fileP_time;
//	const auto prec = 8;
//	auto& [dim, beta, mu, U, Lx, Ly, Lz, M, M0, p, dtau] = params;
//
//	openFile(fileP, name);// dirs->nameNormal);
//	printSeparated(fileP, ',', 8, 0, "x", "y", "z");
//	printSeparated(fileP, ',', prec + 3, true, "avM2z_corr", "avCharge_corr");
//
//	auto [x_num, y_num, z_num] = lat->getNumElems();
//	// FILES CORRELATIONS
//	for (int x = 0; x < x_num; x++)
//		for (int y = 0; y < y_num; y++)
//			for (int z = 0; z < z_num; z++) {
//				auto [xx, yy, zz] = lat->getSymPosInv(x, y, z);
//				printSeparated(fileP, ',', prec, false, xx, yy, zz);
//				printSeparatedP(fileP, ',', prec + 6, true, prec, avs->av_M2z_corr[x][y][z], avs->av_ch2_corr[x][y][z]);
//				//if (times) {
//				//	for (int i = 0; i < M; i++) {
//						//fileP_time << x << "\t" << y << "\t" << z << "\t" << i << "\t" << (avs.av_M2z_corr_uneqTime[x_pos][y_pos][z_pos][i]) << "\t" << (avs.av_Charge2_corr_uneqTime[x_pos][y_pos][z_pos][i]) << endl;
//						//fileP_time << x << "\t" << y << "\t" << z << "\t" << i << "\t" << this->gree << "\t" << (avs.av_Charge2_corr_uneqTime[x_pos][y_pos][z_pos][i]) << endl;
//				//	}
//			}
//	fileP.close();
//}

//void hubbard::ui::collectFouriers(std::string name_times, std::string name, const HubbardParams& params, std::shared_ptr<averages_par> avs, std::shared_ptr<Lattice> lat)
//{
//	using namespace std;
//	std::ofstream file_fouriers, file_fouriers_time, file_response;
//	auto& [dim, beta, mu, U, Lx, Ly, Lz, M, M0, p, dtau] = params;
//	const auto N = lat->get_Ns();
//
//
//	openFile(file_fouriers, name);
//	/*if (times) {
//		file_fouriers_time.open(filenameTimes + "_time.dat");
//		if (!file_fouriers_time.is_open()) {
//			cout << "Couldn't open a file\n";
//			exit(-1);
//		}
//		//file_fouriers_time << "kx\tky\tkz\tdtau\toccupation_fourier\tgreen_up\tgreen_down\tmagnetic_susc\tcharge_susc" << endl;
//		file_fouriers_time << "kx\tky\tkz\ttau\tgreen_up\tgreen_down" << endl;
//	}*/
//	printSeparated(file_fouriers, ',', 12, false, "kx", "ky", "kz");
//	printSeparated(file_fouriers, ',', 16, true, "occ(k)", "spin_str_fac", "ch_str_fac");
//
//	auto [x_num, y_num, z_num] = lat->getNumElems();
//	for (int iter = 0; iter < Lx * Ly * Lz; iter++) {
//
//		/* Fourier occupation */
//		arma::cx_double spin_structure_factor = 0;
//		arma::cx_double charge_structure_factor = 0;
//		//arma::cx_double charge_susc = 0;
//		//arma::cx_double mag_susc = 0;
//		arma::cx_double occupation_fourier = 0;
//		//arma::cx_vec green_up(M_0, arma::fill::zeros);
//		//arma::cx_vec green_down(M_0, arma::fill::zeros);
//		const auto k_vec = lat->get_k_vectors(iter);
//
//		const auto kx = k_vec[0];
//		const auto ky = k_vec[1];
//		const auto kz = k_vec[2];
//		for (int i = 0; i < x_num; i++) {
//			for (int j = 0; j < y_num; j++) {
//				for (int k = 0; k < z_num; k++) {
//					auto [x, y, z] = lat->getSymPosInv(i, j, k);
//					const auto r = lat->get_real_space_vec(x, y, z);
//
//					arma::cx_double expa = exp(imn * dot(r, k_vec));
//
//					occupation_fourier += expa * avs->av_occupation_corr[i][j][k];
//					spin_structure_factor += expa * avs->av_M2z_corr[i][j][k];
//					charge_structure_factor += expa * avs->av_ch2_corr[i][j][k];
//					//spin_structure_factor += expa * avs.avM2z_corr[x][y][z];
//					//if (times) {
//					//	for (int l = 0; l < M_0; l++) {
//							/* DODAC LICZENIE TYCH WSZYSTKICH CZASOWYCH */
//					//		green_up[l] += expa * avs.av_green_up[x][y][z][l];
//					//		green_down[l] += expa * avs.av_green_down[x][y][z][l];
//							//avs.av_green_down[x][y][z][0] + avs.av_green_up[x][y][z][0]
//					//	}
//					//}
//				}
//			}
//		}
//		printSeparatedP(file_fouriers, ',', 12, false, 4, kx, ky, kz);
//		printSeparatedP(file_fouriers, ',', 16, true, 8, 1 - occupation_fourier.real() / (2.0 * N), spin_structure_factor.real(), charge_structure_factor.real());
//		//if (qx == 0 && qy == 0) {
//		//	file_response.open(this->saving_dir + "response_U=" + STR(this->U, 2) +",occ=" + STR(avs->av_occupation,2) + ".dat", std::ofstream::out | std::ofstream::app);
//		//	file_response << Lx << "\t" << Ly << "\t" << Lz << "\t" << beta << "\t" << real(spin_structure_factor) << std::endl;
//		//	file_response.close();
//		//}
//		//if (times) {
//		//	for (int l = 0; l < M_0; l++) {
//		//		file_fouriers_time << kx << "\t" << ky << "\t" << kz << "\t" << l << "\t" << green_up[l] << "\t" << green_down[l] << endl;
//		//	}
//		//}
//	}
//#pragma omp critical
//	file_fouriers.close();
//}