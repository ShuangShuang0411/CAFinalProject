#usage: sh compile.sh filename.cpp

# filename
IN=${1}

# your compiler
CC=g++-11

# openmp flag of your compiler
OPENFLAG=-fopenmp
#OPENFLAG=-openmp
#OPENFLAG=-qopenmp

# set the output name
OUT=./bin/$(basename ${IN} .cpp).out

${CC} -I ./eigen-3.4.0 ${OPENFLAG} ${IN} -o ${OUT}
