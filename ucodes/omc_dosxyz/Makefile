################################################################################
# ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
# simulations
# 
# Copyright (C) 2018 Edgardo Doerner (edoerner@fis.puc.cl)
#
# Makefile for ompMC user codes.
#
################################################################################

# User code:
UCODE = omc_dosxyz

# Include folders needed for compilation:
INCLUDES = -I../../src

# Source files
SOURCES = $(UCODE).c ../../src/ompmc.c \
	  ../../src/omc_utilities.c \
	  ../../src/omc_random.c

# C compiler:
CC = gcc

# Compiler flags:
CFLAGS = -lm -O2 -mtune=native -fopenmp

# The build target executable:
TARGET = $(UCODE).exe

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCES)

clean:
	rm $(TARGET)
