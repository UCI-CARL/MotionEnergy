#------------------------------------------------------------------------------
# MotionEnergy Engine Makefile
#
# Note: This file depends on variables set in configure.mk, thus must be run
# after importing those others.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# MotionEnergy Kernel
#------------------------------------------------------------------------------

# core has all source code
core_dir       := cppME
core_inc_dir   := $(core_dir)/inc
core_src_dir   := $(core_dir)/src
core_obj_dir   := $(core_dir)
COREINCFLAGS   := $(addprefix -I,$(core_inc_dir))

core_inc_files := $(addprefix $(core_inc_dir)/,cuda_version_control.h motion_energy.h)
core_src_files := $(addprefix $(core_src_dir)/,motion_energy.cu)
core_obj_files := $(addprefix $(core_obj_dir)/,motion_energy_cu.o)

core_tgt_file  := motion_energy


#------------------------------------------------------------------------------
# CARLsim Common
#------------------------------------------------------------------------------
targets += $(core_tgt_file)
objects += $(core_obj_files)

.PHONY: release debug $(core_tgt_file)

# release build
release: CXXFLAGS += -O3 -ffast-math
release: NVCCFLAGS += --compiler-options "-O3 -ffast-math"
release: $(targets)

# debug build
debug: CXXFLAGS += -g -Wall
debug: NVCCFLAGS += -g -G
debug: $(targets)

# ME target
$(core_tgt_file): $(core_obj_files) $(core_src_files) $(core_inc_files)

# CUDA files
$(core_obj_dir)/%_cu.o: $(core_src_dir)/%.cu $(core_inc_files)
	$(NVCC) -c $(NVCCSHRFLAGS) $(NVCCINCFLAGS) $(COREINCFLAGS) $(NVCCFLAGS) $< -o $@

# CPP files
$(core_obj_dir)/%.o: $(core_src_dir)/%.cpp $(core_inc_files)
	$(NVCC) -c $(NVCCSHRFLAGS) $(NVCCINCFLAGS) $(COREINCFLAGS) $(NVCCFLAGS) $< -o $@
