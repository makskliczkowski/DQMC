#include "src/user_interface.h"

/* ----------------------- USER INTERFACE --------------------- */
/// <summary>
/// If the commands are given from file, we must treat them the same as arguments
/// </summary>
/// <param name="filename"> the name of the file that contains the command line </param>
/// <returns></returns>
std::vector<std::string> user_interface::parseInputFile(std::string filename){
	std::vector<std::string> commands;
	std::ifstream inputFile(filename);
	std::string line = "";
	if(!inputFile.is_open()){
		std::cout << "Cannot open a file " + filename + " that I could parse. Setting all parameters to default. Sorry :c \n";
		this->set_default();
	}
	else{
		if(std::getline(inputFile, line)){
			commands = split_str(line, " ");														// saving lines to out vector if it can be done, then the parser shall treat them normally
		}
	}
	return std::vector<std::string>(commands.begin(),commands.end()); 
}
/* ----------------------- HUBBARD ----------------------- */
// --- CONSTRUCTOR
hubbard::ui::ui(int argc, char** argv)
{
	this->set_default();
	this->parseModel(argc, change_input_to_vec_of_str(argc, argv));
}
/// <summary>
/// Prints help for a Hubbard interface
/// </summary>
void hubbard::ui::exit_with_help()
{
	printf(
		"Usage: name [options] outputDir \n"
		"options:\n"
		"-m monte carlo steps : bigger than 0 (default 300)\n"
		"-d dimension : set dimension (default 2)\n"
		"	1 -- 1D \n"
		"	2 -- 2D \n"
		"	3 -- 3D -> NOT IMPLEMENTED YET \n"
		"-l lattice type : (default square) -> CHANGE NOT IMPLEMENTED YET \n"
		"   square \n"
		"-t exchange constant : set hopping (default -1) -> NEED TO MAKE IT MORE UNIVERSAL FOR VECTOR\n"
		"-a averages number : set tolerance for statistics, bigger than 0 (default 50)\n"
		"-c correlation time : whether to wait to collect observables, bigger than 0 (default 1)\n"
		"-M0 all Trotter slices sections : this sets how many slices in division our Trotter times number has, bigger than 0 (default 10)\n"
		"-dt dtau : Trotter step (default beta/2*M_0)\n"
		"-dts dtau : Trotter step step (default beta/2*M_0)\n"
		"-dtn dtau : Trotter steps number(default beta/2*M_0)\n"
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
		"-u interaction U : (default 2) -> CURRENTly ONly U>0\n"
		"-us interaction step : bigger than 0, start from smaller (default 1)\n"
		"-un interaction number : integer bigger than 0(default 1)\n"
		"\n"
		"-mu chemical potential mu : (default U/2 -> half filling)\n"
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
	);
	std::exit(1);
}
/// <summary>
/// Hubbard model parser
/// </summary>
/// <param name="argc">number of line arguments</param>
/// <param name="argv">line arguments</param>
void hubbard::ui::parseModel(int argc, v_1d<std::string> argv)
{
	double T = 1/this->beta;
	int i = 1;
	for (i = 1; i < argc; i++)
	{
		/* BREAKERS */
		if (argv[i][0] != '-') break;
		//if (reading_from_file) break;
		if (++i >= argc)
			this->exit_with_help();

		/* PARSE COMANDS */
		std::string argument = ((argv[i - 1])).substr(1, (argv[i - 1]).size() - 1);			// taking the argument to string
		auto it = table.find(argument);														// looking for argument iterator in map parser
		parsers enum_arg;																	// creating an instance of the enum class for switch-case
		if (it != table.end()) {
			enum_arg = it->second;															// if in table - we take the enum 
		}
		else {
			enum_arg = parsers::q;
			fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);						// exit if item is not in the parser
			exit_with_help();
		}
		switch (enum_arg)
		{
		case hubbard::parsers::m:
			/* MONTE CARLO STEPS */
			this->mcSteps = stoi(argv[i]);
			if (this->mcSteps <= 0) {
				std::cout << "Can't be negative! Setting default!\n";
				mcSteps = 300;
			}
			if (this->mcSteps <= 1) {
				this->mcSteps = 1;
				std::cout << "Min is 1!\n";
			}
			break;
		case hubbard::parsers::d:
			/* DIMENSION */
			this->dim = stoi(argv[i]);
			if (dim >= 3 || dim < 1) {
				std::cout << "Bad input! Setting default!\n";
				this->dim = 2;
			}
			break;
		case hubbard::parsers::c:
			/* DIMENSION */
			this->corrTime = stoi(argv[i]);
			if (this->corrTime < 0) {
				std::cout << "Bad input! Setting default!\n";
				this->corrTime = 1;
			}
			break;
		case hubbard::parsers::l:
			/* LATTICE TYPE */
			//type = stoi(argv[i]);
			break;
		case hubbard::parsers::t:
			/* EXCHANGE CONSTANT */
			this->t[0] = stof(argv[i]);
			break;
		case hubbard::parsers::a:
			/* AVERAGES NUMBER */
			this->avsNum = stoi(argv[i]);
			if (this->avsNum <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->avsNum = 50;
			}
			break;
		case hubbard::parsers::M0:
			/* M0 */
			this->M_0 = stoi(argv[i]);
			if (this->M_0 <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->M_0 = 10;
			}
			break;
		case hubbard::parsers::dt:
			/* TROTTER STEP -> NOT IMPLEMENTED YET FOR CHANGING EVERY TIME */
			this->dtau = stof(argv[i]);
			if (this->dtau <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->dtau = 0.15;
			}
			break;
		case hubbard::parsers::dtn:
			this->dtau_num = stoi(argv[i]);
			if (this->dtau_num <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->dtau_num = 1;
			}
			break;
		case hubbard::parsers::dts:
			this->dtau_step = stof(argv[i]);
			if (this->dtau_step <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->dtau_num = 1;
				this->dtau_step = 0.1;
			}
			break;
			//--------//
		case hubbard::parsers::lx:
			/* lx */
			this->lx = stoi(argv[i]);
			if (this->lx <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->lx = 4;
			}
			break;
		case hubbard::parsers::lxs:
			/* lx STEP */
			this->lx_step = stoi(argv[i]);
			if (this->lx_step <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->lx_step = 1;
			}
			break;
		case hubbard::parsers::lxn:
			/* lx NUMBER */
			this->lx_num = stoi(argv[i]);
			if (this->lx_num <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->lx_num = 1;
			}
			break;
		case hubbard::parsers::ly:
			/* ly */
			this->ly = stoi(argv[i]);
			if (this->ly <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->ly = 4;
			}
			break;
		case hubbard::parsers::lys:
			/* ly STEP */
			this->ly_step = stoi(argv[i]);
			if (this->ly_step <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->ly_step = 1;
			}
			break;
		case hubbard::parsers::lyn:
			/* ly NUMBER */
			this->ly_num = stoi(argv[i]);
			if (this->ly_num <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->ly_num = 1;
			}
			break;
		case hubbard::parsers::lz:
			/* lz */
			this->lz = stoi(argv[i]);
			if (this->lz <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->lz = 1;
			}
			break;
		case hubbard::parsers::lzs:
			/* lz STEP */
			this->lz_step = stoi(argv[i]);
			if (this->lz_step <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->lz_step = 1;
			}
			break;
		case hubbard::parsers::lzn:
			/* lz NUMBER */
			this->lz_num = stoi(argv[i]);
			if (this->lz_num <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->lz_num = 1;
			}
			break;
		case hubbard::parsers::b:
			/* BETA */
			this->beta = stof(argv[i]);
			if (this->beta <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->beta = 6;
			}
			T = 1.0 / beta;
			break;
		case hubbard::parsers::bs:
			/* BETA STEP */
			this->beta_step = stof(argv[i]);
			if (this->beta_step <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->beta_step = 1;
			}
			break;
		case hubbard::parsers::bn:
			/* BETA NUMBER */
			beta_num = stoi(argv[i]);
			if (beta_num <= 0) {
				std::cout << "Bad input! Setting default!\n";
				beta_num = 1;
			}
			break;
		case hubbard::parsers::u:
			/* INTERACTION */
			this->U = stof(argv[i]);
			break;
		case hubbard::parsers::us:
			/* INTERACTION STEP */
			this->U_step = stof(argv[i]);
			if (this->U_step <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->U_step = 1;
			}
			break;
		case hubbard::parsers::un:
			/* INTERACTION NUMBER */
			this->U_num = stoi(argv[i]);
			if (this->U_num <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->U_num = 1;
			}
			break;
		case hubbard::parsers::mu:
			/* CHEMICAL POTENTIAL */
			this->mu = stof(argv[i]);
			break;
		case hubbard::parsers::mus:
			/* CHEMICAL POTENTIAL STEP */
			this->mu_step = stof(argv[i]);
			if (this->mu_step <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->mu_step = 1;
			}
			break;
		case hubbard::parsers::mun:
			/* CHEMICAL POTENTIAL NUMBER */
			this->mu_num = stoi(argv[i]);
			if (this->mu_num <= 0) {
				std::cout << "Bad input! Setting default!\n";
				this->mu_num = 1;
			}
			break;
		case hubbard::parsers::q:
			/* QUIET MODE -> NOT READY YET */
			this->quiet = bool(stoi(argv[i]));
			break;
		case hubbard::parsers::th:
			/* OUTER THREADS NUMBER */
			this->outer_threads = stoi(argv[i]);
			if (this->outer_threads <= 0 || this->outer_threads > this->thread_number)
			{
				std::cout << "Bad input! Setting default!\n";
				this->outer_threads = 1;
			}
			break;
		case hubbard::parsers::ti:
			/* INNER THREADS NUMBER */
			this->inner_threads = stoi(argv[i]);
			if (this->inner_threads <= 0 || this->outer_threads * this->inner_threads > this->thread_number)
			{
				std::cout << "Bad input! Setting default!\n";
				this->inner_threads = 1;
			}
			break;
		case hubbard::parsers::qr:
			this->qr_dec = bool(stoi(argv[i]));
			break;
		case hubbard::parsers::times:
			this->cal_times = bool(stoi(argv[i]));
			break;
		case hubbard::parsers::config:
			this->save_conf = bool(stoi(argv[i]));
			break;
		case hubbard::parsers::self_learn:
			this->sf = stoi(argv[i]);
			if(this->sf <0 && sf > 2)
			{
				this->sf = 0;
				std::cout << "WRONG OPTION FOR SF! SETTING DEFAULT\n";
			}
			break;
		case hubbard::parsers::self_learn_n:
			this->sfn = stoi(argv[i]);
			if (sfn <= 0)
			{
				this->sfn = 50;
				std::cout << "WRONG OPTION FOR SF NUMBER! SETTING DEFAULT\n";
			}
			break;
		default:
			fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);
			//cout << "Setting default training model parameters\n";
			exit_with_help();
		}
	}
	if ((!((argv[i]).size()))==0){
		std::filesystem::create_directories(saving_dir);													// creating the directory for saving the files with results
		this->saving_dir = (argv[i]);
	}

	int Ns = lx*ly*lz;
	t = v_1d<double>(Ns, t[0]);
	std::cout << "USING OUTER THREADS = " << outer_threads << std::endl;
	omp_set_num_threads(outer_threads);
	#pragma omp parallel for num_threads(outer_threads) collapse(5)
	for (int bi = 0; bi < beta_num; bi++) {
		// over different betas
		for (int ui = 0; ui < U_num; ui++) {
			// over interactions
			for (int mui = 0; mui < mu_num; mui++) {
				// over chemical potentials
				for (int Li = 0; Li < lx_num; Li++) {
					// over lattice sizes
					/* PARAMS LOCALLY FOR THREADS */
					for (int dtau_i = 0; dtau_i < dtau_num; dtau_i++) {
						int Lxi = lx + Li * lx_step;
						int Lyi = ly + Li * ly_step;
						int Lzi = 1;
						double betai = beta + bi * beta_step;
						double Ui = U + ui * U_step;
						double muii = mu + mui * mu_step;
						double dtauii = dtau + dtau_i * dtau_step;
						/* TROTTER */
						//long double dtaui = dtau;//sqrt(0.125 / Ui);
						int M0i = M_0;
						int Mi = static_cast<double>(1.0 * betai / dtauii);
						int pi = int(1.0 * Mi / M0i);
						Mi = pi * M0i;
						dtauii = 1.0 * betai / static_cast<double>(Mi);
						/* CALCULATE */
						//collectAvs(save_dir, quiet, qr_dec, save_config, mcSteps, dimension, t, Ui, avsNum, corrTime, M_0, dtauii, pi, betai, muii, Lxi, Lyi, Lzi, col_times,sf ,sfn);
					}
				}
			}
		}
	}
	std::cout << "FINISHED EVERY THREAD" << std::endl;
}

void hubbard::ui::set_default()
{
	this->inner_threads = 1;
	this->outer_threads = 1;
	this->thread_number = std::thread::hardware_concurrency();
	this->saving_dir = "";
	this->sf = 0;
	this->sfn = 50;
	this->quiet = 0;

	this->save_conf = false;
	this->qr_dec = false;
	this->cal_times = false;

	this->dim = 2;
	this->lx = 4;
	this->ly = 4;
	this->lz = 1;
	this->lx_step = 2;
	this->ly_step = 2;
	this->lz_step = 0;
	this->lx_num = 1;
	this->ly_num = 1;
	this->lz_num = 1;

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
	std::vector<double> t(lx*ly*lz,1.0);
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
