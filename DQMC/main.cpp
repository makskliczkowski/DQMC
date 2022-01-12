#include "src/user_interface.h"

int main(const int argc, char* argv[]) {
	//auto a = 3;
	//auto b = 4.2;
	//auto c = 4.13;
	//auto d = "I am d";
	//printSeparated(stout,',', 8, true,a, b, c, d);
	//printSeparated(stout, ',', 8, true, VEQ(a), VEQP(b, 2), VEQP(c, 3), VEQ(d));
	std::unique_ptr<user_interface> intface = std::make_unique<hubbard::ui>(argc, argv);
	intface->make_simulation();
	//cin.get();
	return 0;
}