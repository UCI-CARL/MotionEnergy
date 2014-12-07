lib_ver := $(me_major_num).$(me_minor_num).$(me_build_num)
lib_name := libME.a

ME_LIB_DIR ?= /opt/CARL/MotionEnergy

targets += liBME
libraries += $(lib_name).$(lib_ver)

.PHONY: libME install
libME.a: $(me_src_files) $(me_inc_files) $(me_obj_files)
	ar rcs $@.$(lib_ver) $(me_obj_files)

install: $(lib_name)
	rm -rf $(ME_LIB_DIR)
	@test -d $(ME_LIB_DIR) || \
		mkdir -p $(ME_LIB_DIR)
	@test -d $(ME_LIB_DIR)/lib || mkdir \
		$(ME_LIB_DIR)/lib
	@test -d $(ME_LIB_DIR)/include || mkdir \
		$(ME_LIB_DIR)/include
	@install -m 0755 $(lib_name).$(lib_ver) $(ME_LIB_DIR)/lib
	@ln -Tfs $(ME_LIB_DIR)/lib/$(lib_name).$(lib_ver) \
		$(ME_LIB_DIR)/lib/$(lib_name).$(num_ver)
	@ln -Tfs $(ME_LIB_DIR)/lib/$(lib_name).$(num_ver) \
		$(ME_LIB_DIR)/lib/$(lib_name)
	@install -m 0644 $(me_inc_dir)/cuda_version_control.h \
		$(me_inc_dir)/motion_energy.h \
		$(ME_LIB_DIR)/include