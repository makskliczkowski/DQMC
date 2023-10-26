#!/bin/bash

# DQMC makefile for Intel OneApi compiler
# compiler
source /opt/intel/oneapi
icpx -V

cd $HOME/Codes/DQMC/DQMC/

CXX="icpx"

# logfile
LOG=compile_log.txt

# compiler flags
CPPFLAGS="-pthread -qopenmp -fopenmp -qmkl=sequential -std=c++20 -O3"

# linker flags
LDLIBS="-lhdf5 -Wall -Wformat=0 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -lblas -ldl -lmkl_sequential -lstdc++fs"

# the target is
TARGET="dqmc"

# FROM HUBBARD
HUBBARD_FILES="main.cpp dqmc.cpp dqmc2.cpp Models/hubbard.cpp user_interface.cpp"

# FROM source
source_dir="./source/cpp/"
SOURCE_FILES="${source_dir}ui.cpp ${source_dir}str.cpp ${source_dir}exceptions.cpp ${source_dir}directories.cpp ${source_dir}common.cpp" 
SOURCE_FILES_LAT="${source_dir}Lattices/square.cpp ${source_dir}Lattices/hexagonal.cpp"
#SOURCE_HEADERS = ./include/general_model.h ./include/hubbard_dqmc_qr.h ./include/hubbard.h ./include/user_interface.h ./source/src/common.h

# INCLUDE LIBRARIES
ARMA="$HOME/LIBRARIES_CPP/armadillo-11.0.1/include/"


# all: ${TARGET}

# ${TARGET}: ${TARGET}.o
# 	${CXX} -o ${TARGET} ${TARGET}.o >& ${LOG}

# ${TARGET}.o: ${SOURCE_FILES} ${SOURCE_FILES_LAT} ${SOURCE_HEADERS}
# 	${CXX} -c ${SOURCE_FILES} ${SOURCE_FILES_LAT} -I${ARMA} ${CPPFLAGS}

icpx ${HUBBARD_FILES} ${SOURCE_FILES} ${SOURCE_FILES_LAT} -o ${TARGET}.o -I${ARMA} ${CPPFLAGS} ${LDLIBS} >& ${LOG}