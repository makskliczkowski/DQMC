#include "src/user_interface.h"

int main(const int argc, char* argv[]) {
	//stout << printSeparated(',', 3, a, b, c) << EL;
	std::unique_ptr<user_interface> intface = std::make_unique<hubbard::ui>(argc, argv);
	intface->make_simulation();
	cin.get();
	return 0;
}