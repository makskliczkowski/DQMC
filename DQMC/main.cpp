
#include "src/user_interface.h"

int main(const int argc, char* argv[]){

	std::unique_ptr<user_interface> intface = std::make_unique<hubbard::ui>(argc, argv);
	intface->make_simulation();
	return 0;
}