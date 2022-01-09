#include "src/user_interface.h"

// -------------------------------------------------------- USER INTERFACE --------------------------------------------------------

/// <summary>
///
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="value"></param>
/// <param name="argv"></param>
/// <param name="choosen_option"></param>
/// <param name="geq_0"></param>
template<typename T>
void user_interface::set_option(T& value, const v_1d<std::string>& argv, std::string choosen_option, bool geq_0)
{
	if (std::string option = this->getCmdOption(argv, choosen_option); option != "")
		value = static_cast<T>(stod(option));												// set value to an option
	if (geq_0 && value <= 0)																	// if the variable shall be bigger equal 0
		this->set_default_msg(value, choosen_option.substr(1), \
			choosen_option + " cannot be negative\n", hubbard::default_params);
}

/// <summary>
/// Set
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="value"></param>
/// <param name="option"></param>
/// <param name="message"></param>
template<typename T>
void user_interface::set_default_msg(T& value, std::string option, std::string message, const std::unordered_map <std::string, std::string>& map) const
{
	std::cout << message;																// print warning
	std::string value_str = "";																// we will set this to value
	auto it = map.find(option);
	if (it != map.end()) {
		value_str = it->second;															// if in table - we take the enum
	}
	value = stod(value_str);
}

/// <summary>
/// Find a given option in a vector of string given from cmd parser
/// </summary>
/// <param name="vec">vector of strings from cmd</param>
/// <param name="option">the option that we seek</param>
/// <returns>value for given option if exists, if not an empty string</returns>
std::string user_interface::getCmdOption(const v_1d<std::string>& vec, std::string option) const
{
	if (auto itr = std::find(vec.begin(), vec.end(), option); itr != vec.end() && ++itr != vec.end())
		return *itr;
	return std::string();
}

/// <summary>
/// If the commands are given from file, we must treat them the same as arguments
/// </summary>
/// <param name="filename"> the name of the file that contains the command line </param>
/// <returns></returns>
std::vector<std::string> user_interface::parseInputFile(std::string filename) {
	std::vector<std::string> commands;
	std::ifstream inputFile(filename);
	std::string line = "";
	if (!inputFile.is_open()) {
		std::cout << "Cannot open a file " + filename + " that I could parse. Setting all parameters to default. Sorry :c \n";
		this->set_default();
	}
	else {
		if (std::getline(inputFile, line)) {
			commands = split_str(line, " ");										// saving lines to out vector if it can be done, then the parser shall treat them normally
		}
	}
	return std::vector<std::string>(commands.begin(), commands.end());
}

// -------------------------------------------------------- HUBBARD --------------------------------------------------------
// -------------------------------------------------------- CONSTRUCTOR

/// <summary>
/// </summary>
/// <param name="argc">number of cmd parameters</param>
/// <param name="argv">cmd parameters</param>
hubbard::ui::ui(int argc, char** argv)
{
	auto input = change_input_to_vec_of_str(argc, argv);									// change standard input to vec of strings
	input = std::vector<std::string>(input.begin()++, input.end());							// skip the first element which is the name of file

	//plog::init(plog::info, "log.txt");														// initialize logger
	if (std::string option = this->getCmdOption(input, "-f"); option != "") {
		input = this->parseInputFile(option);												// parse input from file
	}
	this->parseModel(input.size(), input);													// parse input from CMD directly
}

// -------------------------------------------------------- PARSERS

/// <summary>
/// Prints help for a Hubbard interface
/// </summary>
void hubbard::ui::exit_with_help()
{
	printf(
		"Usage: name [options] outputDir \n"
		"options:\n"
		" The input can be both introduced with [options] described below or with giving the input directory \n"
		" (which also is the flag in the options) \n"
		" options:\n"
		"-f input file for all of the options : (default none) \n"
		"-m monte carlo steps : bigger than 0 (default 300) \n"
		"-d dimension : set dimension (default 2) \n"
		"	1 -- 1D \n"
		"	2 -- 2D \n"
		"	3 -- 3D -> NOT IMPLEMENTED YET \n"
		"-l lattice type : (default square) -> CHANGE NOT IMPLEMENTED YET \n"
		"   square \n"
		"-t exchange constant : set hopping (default 1) \n"
		"   is numeric? - constant value \n"
		"   'r' - uniform random on each (NOT IMPLEMENTED YET) \n"
		"-a averages number : set tolerance for statistics, bigger than 0 (default 50) \n"
		"-c correlation time : whether to wait to collect observables, bigger than 0 (default 1) \n"
		"-M0 all Trotter slices sections : this sets how many slices in division our Trotter times number has, bigger than 0 (default 10)\n"
		"-dt dtau : Trotter step (default 0.1)\n"
		"-dts dtau : Trotter step step (default 0)\n"
		"-dtn dtau : Trotter steps number(default 1)\n"
		// SIMULATIONS STEPS
		"-lx x-length : bigger than 0(default 4)\n"
		"-lxs x-length step : integer, bigger than 0 (default 1)\n"
		"-lxn x-length steps number : integer bigger than 0(default 1)\n"
		"\n"
		"-ly y-length : bigger than 0(default 4)\n"
		"-lys y-length step : integer, bigger than 0 (default 1)\n"
		"-lyn y-length steps number : integer bigger than 0(default 1)\n"
		"\n"
		"-lz z-length : bigger than 0(default 1)\n"
		"-lzs z-length step : integer, bigger than 0 (default 1)\n"
		"-lzn z-length steps number : integer bigger than 0(default 1)\n"
		"\n"
		"-b inversed temperature :bigger than 0 (default 6)\n"
		"-bs inversed temperature step : bigger than 0 (default 1)\n"
		"-bn inversed temperature number : integer bigger than 0(default 1)\n"
		"\n"
		"-u interaction U : (default 2) -> CURRENTLy ONLY U>0\n"
		"-us interaction step : bigger than 0, start from smaller (default 1)\n"
		"-un interaction number : integer bigger than 0(default 1)\n"
		"\n"
		"-mu chemical potential mu : (default 0 -> half filling)\n"
		"-mus chemical potential mu step : bigger than 0, start from smaller (default 1)\n"
		"-mun chemical potential mu number : integer bigger than 0(default 1)\n"
		"\n"
		"-th outer threads : number of outer threads (default 1)\n"
		"-ti inner threads : number of inner threads (default 1)\n"
		"-q : 0 or 1 -> quiet mode (no outputs) (default false)\n"
		"\n"
		"-cg save-config : for machine learning -> we will save all Trotter times at the same time(default false)\n"
		"-qr qr decomposition \n"
		"-ct collect time averages - slower \n"
		"\n"
		"-sf use self-learning : 0(default) - do not use, 1 (just train parameters and exit), 2 (just make simulation from the existing network)\n"
		"-sfn - the number of configurations saved to train(default 50)\n"
		"-h - help\n"
	);
	std::exit(1);
}

/// <summary>
/// Setting Hubbard parameters to default
/// </summary>
void hubbard::ui::set_default()
{
	this->inner_threads = 1;
	this->outer_threads = 1;
	this->thread_number = std::thread::hardware_concurrency();
	this->saving_dir = "." + kPS;
	this->sf = 0;
	this->sfn = 50;
	this->quiet = 0;

	this->save_conf = false;
	this->qr_dec = true;
	this->cal_times = false;

	this->dim = 2;
	this->lx = 4;
	this->ly = 4;
	this->lz = 1;
	this->lx_step = 2;
	this->ly_step = 2;
	this->lz_step = 2;
	this->lx_num = 1;
	this->ly_num = 1;
	this->lz_num = 1;
	this->boundary_conditions = 0;

	this->beta = 6;
	this->beta_step = 1;
	this->beta_num = 1;

	// hubbard
	this->U = 2.0;
	this->U_step = 1;
	this->U_num = 1;
	this->mu = 0.5 * U;
	this->mu_step = 1;
	this->mu_num = 1;
	this->t = std::vector<double>(lx * ly * lz, 1.0);
	this->t_fill = 1.0;
	// trotter
	this->M_0 = 10;
	this->p = 2;
	this->M = p * M_0;
	this->dtau = beta / (1.0 * M);
	this->dtau_num = 1;
	this->dtau_step = 0.1;

	// monte carlo
	this->mcSteps = 300;
	this->avsNum = 50;
	this->corrTime = 1;
}

/// <summary>
/// Hubbard model parser
/// </summary>
/// <param name="argc">number of line arguments</param>
/// <param name="argv">line arguments</param>
void hubbard::ui::parseModel(int argc, const v_1d<std::string>& argv)
{
	this->set_default();

	std::string choosen_option = "";

	//---------- SIMULATION PARAMETERS
	// monte carlo steps
	choosen_option = "-m";
	this->set_option(this->mcSteps, argv, choosen_option);
	// dimension
	choosen_option = "-d";
	this->set_option(this->dim, argv, choosen_option, false);
	if (this->dim >= 3 || this->dim < 1)
		this->set_default_msg(this->dim, choosen_option.substr(1), \
			"Wrong dimmension\n", hubbard::default_params);
	// correlation time
	choosen_option = "-c";
	this->set_option(this->corrTime, argv, choosen_option);
	// number of averages
	choosen_option = "-a";
	this->set_option(this->avsNum, argv, choosen_option);
	// Trotter subintervals
	choosen_option = "-m0";
	this->set_option(this->M_0, argv, choosen_option);
	// ---------- Trotter time difference
	choosen_option = "-dt";
	this->set_option(this->dtau, argv, choosen_option);
	// Trotter time differences number
	choosen_option = "-dtn";
	this->set_option(this->dtau_num, argv, choosen_option);
	// Trotter time differences step
	choosen_option = "-dts";
	this->set_option(this->dtau_step, argv, choosen_option);
	// ---------- beta
	choosen_option = "-b";
	this->set_option(this->beta, argv, choosen_option);
	// beta step
	choosen_option = "-bs";
	this->set_option(this->beta_step, argv, choosen_option);
	// betas number
	choosen_option = "-bn";
	this->set_option(this->beta_num, argv, choosen_option);
	// ---------- U
	choosen_option = "-u";
	this->set_option(this->U, argv, choosen_option, false);
	// U step
	choosen_option = "-us";
	this->set_option(this->U_step, argv, choosen_option, false);
	// U number
	choosen_option = "-un";
	this->set_option(this->U_num, argv, choosen_option);
	// ---------- mu
	choosen_option = "-mu";
	this->set_option(this->mu, argv, choosen_option, false);
	// mu step
	choosen_option = "-mus";
	this->set_option(this->mu_step, argv, choosen_option, false);
	// mu number
	choosen_option = "-mun";
	this->set_option(this->mu_num, argv, choosen_option);
	// ---------- LATTICE PARAMETERS
	// lx
	choosen_option = "-lx";
	this->set_option(this->lx, argv, choosen_option);
	// lx_step
	choosen_option = "-lxs";
	this->set_option(this->lx_step, argv, choosen_option);
	// lx_num
	choosen_option = "-lxn";
	this->set_option(this->lx_num, argv, choosen_option);
	// ly
	choosen_option = "-ly";
	this->set_option(this->ly, argv, choosen_option);
	// ly_step
	choosen_option = "-lys";
	this->set_option(this->ly_step, argv, choosen_option);
	// ly_num
	choosen_option = "-lyn";
	this->set_option(this->ly_num, argv, choosen_option);
	// lz
	choosen_option = "-lz";
	this->set_option(this->lz, argv, choosen_option);
	// lz_step
	choosen_option = "-lzs";
	this->set_option(this->lz_step, argv, choosen_option);
	// lz_num
	choosen_option = "-lzn";
	this->set_option(this->lz_num, argv, choosen_option);

	// double T = 1/this->beta;
	int Ns = this->lx * this->ly * this->lz;
	//---------- OTHERS
	// quiet
	choosen_option = "-q";
	this->set_option(this->quiet, argv, choosen_option, false);
	// outer thread number
	choosen_option = "-th";
	this->set_option(this->outer_threads, argv, choosen_option, false);
	if (this->outer_threads <= 0 || this->outer_threads * this->inner_threads > this->thread_number)
		this->set_default_msg(this->outer_threads, choosen_option.substr(1), \
			"Wrong number of threads\n", hubbard::default_params);
	// inner thread number
	choosen_option = "-ti";
	this->set_option(this->inner_threads, argv, choosen_option, false);
	if (this->inner_threads <= 0 || this->outer_threads * this->inner_threads > this->thread_number)
		this->set_default_msg(this->inner_threads, choosen_option.substr(1), \
			"Wrong number of threads\n", hubbard::default_params);
	// qr
	choosen_option = "-qr";
	this->set_option(this->qr_dec, argv, choosen_option, false);
	// calculate different times
	choosen_option = "-ct";
	this->set_option(this->cal_times, argv, choosen_option, false);
	// save configurations
	choosen_option = "-cg";
	this->set_option(this->save_conf, argv, choosen_option, false);
	// self learning
	choosen_option = "-sf";
	this->set_option(this->sf, argv, choosen_option, false);
	if (this->sf < 0 || this->sf > 2)
		this->set_default_msg(this->sf, choosen_option.substr(1), \
			"Wrong input for self learning\n", hubbard::default_params);
	// self learning number
	choosen_option = "-sfn";
	this->set_option(this->sfn, argv, choosen_option, false);
	// hopping coefficients
	choosen_option = "-t";
	this->set_option(this->t_fill, argv, choosen_option, false);
	this->t = std::vector<double>(Ns, t_fill);
	// get help
	choosen_option = "-h";
	if (std::string option = this->getCmdOption(argv, choosen_option); option != "")
		exit_with_help();

	std::string folder = "." + kPS + "results" + kPS;
	if (!argv[argc - 1].empty() && argc % 2 != 0) {
		// only if the last command is non-even
		folder = argv[argc - 1];
		if (fs::create_directories(folder))											// creating the directory for saving the files with results
			this->saving_dir = folder;																// if can create dir this is is
	}
	else {
		this->saving_dir = folder;																	// if can create dir this is is
	}
	//omp_set_num_threads(outer_threads);
	omp_set_num_threads(outer_threads * inner_threads);
}

/// <summary>
///
/// </summary>
void hubbard::ui::make_simulation()
{
	stout << "STARTING THE SIMULATION AND USING OUTER THREADS = " << outer_threads << ", INNER THREADS = " << inner_threads << std::endl;

	// save the log file
#pragma omp single
	{
		std::fstream fileLog(this->saving_dir + "HubbardLog.csv", std::ios::in | std::ios::app);
		fileLog.seekg(0, std::ios::end);
		if (fileLog.tellg() == 0) {
			fileLog.clear();
			fileLog.seekg(0, std::ios::beg);
			printSeparated(fileLog, ',', { "lattice_type","mcsteps","avsNum","corrTime", "M", "M0", "dtau", "Lx",\
				"Ly", "Lz", "beta", "U", "mu", "occ", "sd(occ)", "av_sgn", "sd(sgn)", "Ekin",\
				"sd(Ekin)", "m^2_z", "sd(m^2_z)", "m^2_x", "time taken" }, 14);
		}
		fileLog.close();
	}

	struct params {
		double beta;
		double mu;
		double U;
		int Lx, Ly, Lz;
		double dtau;
		params(double beta, double mu, double U, int Lx, int Ly, int Lz, double dtau) {
			this->beta = beta; this->mu = mu; this->U = U; this->Lx = Lx; this->Ly = Ly; this->Lz = Lz; this->dtau = dtau;
		}
	};
	v_1d<params> param_list;

	for (int bi = 0; bi < this->beta_num; bi++) {
		// over different betas
		for (int ui = 0; ui < this->U_num; ui++) {
			// over interactions
			for (int mui = 0; mui < this->mu_num; mui++) {
				// over chemical potentials
				for (int Li = 0; Li < this->lx_num; Li++) {
					// over lattice sizes (currently square)
					for (int dtaui = 0; dtaui < this->dtau_num; dtaui++) {
						// PARAMS LOCALLY FOR THREADS
						auto Lx_i = this->lx + Li * this->lx_step;
						auto Ly_i = this->ly + Li * this->ly_step;
						auto Lz_i = 1;

						auto beta_i = this->beta + bi * this->beta_step;
						auto U_i = this->U + ui * this->U_step;
						auto mu_i = this->mu + mui * mu_step;
						auto dtau_i = this->dtau + dtaui * this->dtau_step;
						//double dtaui = dtau;//sqrt(0.125 / Ui);
						param_list.push_back(params(beta_i, mu_i, U_i, Lx_i, Ly_i, Lz_i, dtau_i));
						//collectAvs(U_i, M0_i, dtau_i, p_i, beta_i, mu_i, Lx_i, Ly_i, Lz_i);
					}
				}
			}
		}
	}

#pragma omp parallel for num_threads(this->outer_threads)
	for (int i = 0; i < param_list.size(); i++) {
		auto M_i = static_cast<int>(1.0 * param_list[i].beta / param_list[i].dtau);
		const auto p_i = std::ceil(1.0 * M_i / this->M_0);
		M_i = p_i * this->M_0;
		param_list[i].dtau = 1.0 * param_list[i].beta / (M_i);
		collectAvs(param_list[i].U, this->M_0, param_list[i].dtau, p_i, param_list[i].beta, param_list[i].mu, param_list[i].Lx, param_list[i].Ly, param_list[i].Lz);
	}

	/*
	this->M = static_cast<int>(1.0 * this->beta / this->dtau);
	this->p = std::ceil(1.0 * M / M_0);
	M = p * M_0;
	dtau = beta / double(M);
	collectAvs(this->U, this->M_0, dtau, p, beta, mu, this->lx, this->ly, this->lz);
	*/
	std::cout << "FINISHED EVERY THREAD" << std::endl;
}

// -------------------------------------------------------- HELPERS

void hubbard::ui::collectAvs(double u, int M_0, double Dtau, int p, double Beta, double Mu, int Lx, int Ly, int Lz)
{
	using namespace std;
	auto start = chrono::high_resolution_clock::now();
	const int prec = 10;
	//plog::init(plog::debug, "logger.txt");																			// initialize plogger
	// parameters and constants
	const auto M = static_cast<int>(Beta / Dtau);
	std::shared_ptr<averages_par> avs;
	// model
	std::shared_ptr<Lattice> lat;
	// ------------------------------- set lattice --------------------------------
	switch (this->lattice_type) {
	case 0:
		lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, dim, this->boundary_conditions);
		break;
	default:
		lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, dim, this->boundary_conditions);
		break;
	}
	// ------------------------------- set model --------------------------------
	std::unique_ptr<hubbard::HubbardModel> model = std::make_unique<hubbard::HubbardQR>(this->t, Dtau, M_0, u, Mu, Beta, lat, this->inner_threads, this->cal_times);
	std::ofstream fileLog, fileP, fileP_time, fileGup, fileGdown, fileSignLog;
	// take all the directories needed
	model->setDirs(this->saving_dir);
	auto dirs = model->get_directories();
	// RELAX
	if (sf == 0)																														// without using machine learning to self learn
	{
		model->relaxation(impDef::algMC::heat_bath, this->mcSteps,  this->save_conf, this->quiet);										// this can also handle saving configurations
		if (!this->save_conf) {
			// FILES
			openFile(fileLog, this->saving_dir + "HubbardLog.csv", std::ios::in | std::ios::app);
			openFile(fileSignLog, this->saving_dir + "HubbardSignLog_" + dirs->LxLyLz + ",U=" + str_p(u, 2) +\
				",beta=" + str_p(Beta, 2) + ",dtau=" + str_p(Dtau, 4) +\
				".dat", std::ios::in | std::ios::app);
			openFile(fileP, dirs->nameNormal);
			printSeparated(fileP, ',', { "x","y","z" }, 8, 0);
			printSeparated(fileP, ',', { "avM2z_corr","avCharge_corr" }, prec + 3);
			// REST
			model->average(impDef::algMC::heat_bath, this->corrTime, this->avsNum, 1, this->quiet, this->cal_times);
			avs = model->get_avs();

			// SAVING TO STRING
			printSeparated(fileLog, ',', { lat->get_type(), STR(this->mcSteps), STR(this->avsNum),\
				STR(this->corrTime), STR(M), STR(M_0), str_p(Dtau, prec),\
				STR(Lx), STR(Ly), STR(Lz), STR(Beta), STR(u),\
				STR(Mu), str_p(avs->av_occupation, prec), str_p(avs->sd_occupation, prec),\
				str_p(avs->av_sign, prec), str_p(avs->sd_sign, prec),\
				str_p(avs->av_Ek, prec),str_p(avs->sd_Ek, prec),\
				str_p(avs->av_M2z, prec),str_p(avs->sd_M2z, prec),\
				str_p(avs->av_M2x, prec),str_p(tim_s(start)) }, 14);
			printSeparated(fileSignLog, '\t', { str_p(avs->av_occupation, prec),\
				str_p(avs->av_sign, prec), str_p(Mu, 3) }, 12);
#pragma omp critical
			std::cout << "\t--- Average occupation is n = " << avs->av_occupation << "\t---Average sign is sign= " << avs->av_sign << "\t---Average local moment is m_z^2= " << avs->av_M2z << "---" << endl << endl;

			// FILES CORRELATIONS
			for (int x = -Lx + 1; x < Lx; x++) {
				for (int y = -Ly + 1; y < Ly; y++) {
					for (int z = -Lz + 1; z < Lz; z++) {
						int x_pos = x + Lx - 1;
						int y_pos = y + Ly - 1;
						int z_pos = z + Lz - 1;
						printSeparated(fileP, ',', { x , y, z }, 8, 0);
						printSeparated(fileP, ',', { avs->av_M2z_corr[x_pos][y_pos][z_pos], (avs->av_ch2_corr[x_pos][y_pos][z_pos]) }, prec + 6);
						//if (times) {
						//	for (int i = 0; i < M; i++) {
								//fileP_time << x << "\t" << y << "\t" << z << "\t" << i << "\t" << (avs.av_M2z_corr_uneqTime[x_pos][y_pos][z_pos][i]) << "\t" << (avs.av_Charge2_corr_uneqTime[x_pos][y_pos][z_pos][i]) << endl;
								//fileP_time << x << "\t" << y << "\t" << z << "\t" << i << "\t" << this->gree << "\t" << (avs.av_Charge2_corr_uneqTime[x_pos][y_pos][z_pos][i]) << endl;
						//	}
					}
				}
			}
#pragma omp critical
			fileP.close();
#pragma omp critical
			fileLog.close();
#pragma omp critical
			fileSignLog.close();
			this->collectFouriers(dirs->nameFouriersTime, dirs->nameFouriers, Lx, Ly, Lz, M, beta, avs);
		}
	}
	std::cout << "FINISHED EVERYTHING - Time taken: " << (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start)).count() << " seconds" << endl;
}

void hubbard::ui::collectFouriers(std::string name_times, std::string name, int Lx, int Ly, int Lz, int M, double beta, std::shared_ptr<averages_par> avs)
{
	using namespace std;
	std::ofstream file_fouriers, file_fouriers_time, file_response;
	openFile(file_fouriers, name);
	/*if (times) {
		file_fouriers_time.open(filenameTimes + "_time.dat");
		if (!file_fouriers_time.is_open()) {
			cout << "Couldn't open a file\n";
			exit(-1);
		}
		//file_fouriers_time << "kx\tky\tkz\tdtau\toccupation_fourier\tgreen_up\tgreen_down\tmagnetic_susc\tcharge_susc" << endl;
		file_fouriers_time << "kx\tky\tkz\ttau\tgreen_up\tgreen_down" << endl;
	}*/
	printSeparated(file_fouriers, ',', { "kx","ky","kz" }, 12, 0);
	printSeparated(file_fouriers, ',', { "occ(k)","spin_str_fac","ch_str_fac" }, 16);

	int kx_num = Lx; int ky_num = Ly; int kz_num = Lz;
	int N = kx_num * ky_num * kz_num;
	const auto two_pi_over_Lx = TWOPI / kx_num;
	const auto two_pi_over_Ly = TWOPI / ky_num;
	const auto two_pi_over_Lz = TWOPI / kz_num;

	for (int qx = 0; qx < kx_num; qx++) {
		double kx = -PI + two_pi_over_Lx * qx;
		for (int qy = 0; qy < ky_num; qy++) {
			double ky = -PI + two_pi_over_Ly * qy;
			for (int qz = 0; qz < kz_num; qz++) {
				double kz = -PI + two_pi_over_Lz * qz;
				/* Fourier occupation */

				arma::cx_double spin_structure_factor = 0;
				arma::cx_double charge_structure_factor = 0;
				//arma::cx_double charge_susc = 0;
				//arma::cx_double mag_susc = 0;
				arma::cx_double occupation_fourier = 0;
				//arma::cx_vec green_up(M_0, arma::fill::zeros);
				//arma::cx_vec green_down(M_0, arma::fill::zeros);
				for (int i = -kx_num + 1; i < kx_num; i++) {
					for (int j = -ky_num + 1; j < ky_num; j++) {
						for (int k = -kz_num + 1; k < kz_num; k++) {
							int x = i + kx_num - 1;
							int y = j + ky_num - 1;
							int z = k + kz_num - 1;
							arma::cx_double expa = exp(imn * (kx * i + ky * j + kz * k));

							occupation_fourier += expa * avs->av_occupation_corr[x][y][z];
							spin_structure_factor += expa * avs->av_M2z_corr[x][y][z];
							charge_structure_factor += expa * avs->av_ch2_corr[x][y][z];
							//spin_structure_factor += expa * avs.avM2z_corr[x][y][z];
							//if (times) {
							//	for (int l = 0; l < M_0; l++) {
									/* DODAC LICZENIE TYCH WSZYSTKICH CZASOWYCH */
							//		green_up[l] += expa * avs.av_green_up[x][y][z][l];
							//		green_down[l] += expa * avs.av_green_down[x][y][z][l];
									//avs.av_green_down[x][y][z][0] + avs.av_green_up[x][y][z][0]
							//	}
							//}
						}
					}
				}
				printSeparated(file_fouriers, ',', { kx,ky,kz }, 12, 0);
				printSeparated(file_fouriers, ',', { 1 - occupation_fourier.real() / (2.0 * N) , spin_structure_factor.real() ,charge_structure_factor.real() }, 16);
				//if (qx == 0 && qy == 0) {
				//	file_response.open(this->saving_dir + "response_U=" + STR(this->U, 2) +",occ=" + STR(avs->av_occupation,2) + ".dat", std::ofstream::out | std::ofstream::app);
				//	file_response << Lx << "\t" << Ly << "\t" << Lz << "\t" << beta << "\t" << real(spin_structure_factor) << std::endl;
				//	file_response.close();
				//}
				//if (times) {
				//	for (int l = 0; l < M_0; l++) {
				//		file_fouriers_time << kx << "\t" << ky << "\t" << kz << "\t" << l << "\t" << green_up[l] << "\t" << green_down[l] << endl;
				//	}
				//}
			}
		}
	}
#pragma omp critical
	file_fouriers.close();
}