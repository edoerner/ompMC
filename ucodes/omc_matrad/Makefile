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
UCODE = omc_matrad

# Include folders needed for compilation:
INCLUDES = -I../../src

# Source files
SOURCES = $(UCODE).c

# C compiler:
CC = /usr/local/MATLAB/R2018b/bin/mex

# Compiler flags:
CFLAGS = -lm

# The build target executable:
TARGET = $(UCODE).mexa64

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -output $(TARGET) $(SOURCES)

clean:
	rm $(TARGET)