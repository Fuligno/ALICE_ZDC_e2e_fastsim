# Targets:
#   make           -> run_cfm_onnx (requires ONNXRuntime)
#   make selftest  -> run_cfm_e2e_selftest (no ONNXRuntime)
#
# Point the build at an ONNXRuntime tree with `make ORT=/path/to/onnxruntime`,
# or set ORT_INC and ORT_LIB separately when include/ and lib/ are not siblings.

ORT      ?= ./onnxruntime
ORT_INC  ?= $(ORT)/include
ORT_LIB  ?= $(ORT)/lib

# Absolute rpath, so the binary finds libonnxruntime.so from any working directory.
ORT_LIB_ABS := $(abspath $(ORT_LIB))

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
LDFLAGS   = -L$(ORT_LIB) -lonnxruntime -Wl,-rpath,$(ORT_LIB_ABS)

.PHONY: all selftest clean

all: run_cfm_onnx

run_cfm_onnx: run_cfm_onnx.cpp
	$(CXX) $(CXXFLAGS) -I$(ORT_INC) $< -o $@ $(LDFLAGS)

selftest: run_cfm_e2e_selftest

run_cfm_e2e_selftest: run_cfm_onnx.cpp
	$(CXX) $(CXXFLAGS) -DSELFTEST $< -o $@

clean:
	rm -f run_cfm_onnx run_cfm_e2e_selftest