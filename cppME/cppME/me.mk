me_inc_dir := cppME/include
me_src_dir := cppME/src
me_obj_dir := cppME

me_inc_files := $(addprefix $(me_inc_dir)/,motion_energy.h cuda_version_control.h)
me_src_files := $(addprefix $(me_src_dir)/,motion_energy.cu)
me_obj_files := $(addprefix $(me_obj_dir)/,motion_energy_cu.o)
me_tgt_file  := ME

targets += $(me_tgt_file)
objects += $(me_obj_files)


.PHONY: $(me_tgt_file)

$(me_tgt_file): $(me_src_files) $(me_inc_files) $(me_obj_files)

$(me_obj_dir)/%_cu.o: $(me_src_dir)/%.cu $(me_inc_dir)/%.h
	nvcc -c -I/usr/local/cuda/samples/common/inc -I$(me_inc_dir) -D__CUDA6__ -arch sm_30 -use_fast_math $< -o $@