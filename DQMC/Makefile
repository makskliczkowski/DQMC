# DQMC makefile for Intel OneApi compiler
# compiler
CXX = icpx
RM = rm -f

# logfile
LOG = compile_log.txt

# compiler flags
CPPFLAGS = -pthread -lstdc++fs -qopenmp -fopenmp -qmkl=sequential -std=c++17 -O3

# linker flags
LDLIBS = $(shell root-config --libs)

# the target is
TARGET = dqmc

# FROM HUBBARD
HUBBARD_FILES = main.cpp general_model.cpp hubbard_dqmc_qr.cpp hubbard.cpp user_interface.cpp

# FROM source
source_dir = ./source/
SOURCE_FILES = ${source_dir}common.cpp ${source_dir}statistical.cpp ${source_dir}ui.cpp 
SOURCE_FILES_LAT = ${source_dir}Lattices/square.cpp ${source_dir}Lattices/hexagonal.cpp
#SOURCE_HEADERS = ./include/general_model.h ./include/hubbard_dqmc_qr.h ./include/hubbard.h ./include/user_interface.h ./source/src/common.h

# INCLUDE LIBRARIES
ARMA = $HOME/LIBRARIES_CPP/armadillo-11.0.1/include


# all: ${TARGET}

# ${TARGET}: ${TARGET}.o
# 	${CXX} -o ${TARGET} ${TARGET}.o >& ${LOG}

# ${TARGET}.o: ${SOURCE_FILES} ${SOURCE_FILES_LAT} ${SOURCE_HEADERS}
# 	${CXX} -c ${SOURCE_FILES} ${SOURCE_FILES_LAT} -I${ARMA} ${CPPFLAGS}

${CXX} -o ${TARGET}.o >& ${LOG} -c ${SOURCE_FILES} ${SOURCE_FILES_LAT} -I${ARMA} ${CPPFLAGS} >& ${LOG}