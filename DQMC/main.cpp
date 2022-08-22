#include "include/user_interface.h"

int main(const int argc, char* argv[]) {
	//auto a = 3;
	//auto b = 4.2;
	//auto c = 4.13;
	//auto d = "I am d";
	//printSeparated(stout,',', 8, true,a, b, c, d);
	//printSeparated(stout, ',', 8, true, VEQ(a), VEQP(b, 2), VEQP(c, 3), VEQ(d));
	/*
	int Ns = 16;
	arma::mat U(Ns, Ns, arma::fill::zeros);
	arma::vec D(Ns, arma::fill::zeros);
	arma::mat V(Ns, Ns, arma::fill::zeros);
	arma::mat Vtmp(Ns, Ns, arma::fill::zeros);
	arma::mat B1(Ns, Ns, arma::fill::randn);
	arma::mat B2(Ns, Ns, arma::fill::randu);
	
	auto standard = B2 * B1;
	standard.print("\n\nstandard multiplication");

	svd(U, D, V, B1);
	multiplyMatricesSVDFromRight(B2, U, D, V, Vtmp);
	auto grr = U * DIAG(D) * V.t();
	grr.print("\n\nwith SVD");
	*/

	std::unique_ptr<user_interface> intface = std::make_unique<hubbard::ui>(argc, argv);
	intface->make_simulation();
	cin.get();

	return 0;
}