#------------------------------------------------------------------------------
# liBME Makefile
#
# Note: This file depends on variables set in configure.mk and me.mk, thus
# must be run after importing those others.
#------------------------------------------------------------------------------

ME_MAJOR_NUM := 0
ME_MINOR_NUM := 3
ME_BUILD_NUM := 0

lib_ver := $(ME_MAJOR_NUM).$(ME_MINOR_NUM).$(ME_BUILD_NUM)
lib_name := libME.a

targets += $(lib_name)
libraries += $(lib_name).$(lib_ver)

.PHONY: $(lib_name) install uninstall

install: $(lib_name)
	ar rcs $(lib_name).$(lib_ver) $(core_obj_files)
	@test -d $(ME_LIB_DIR) || mkdir -p $(ME_LIB_DIR)
	@test -d $(ME_LIB_DIR)/lib || mkdir $(ME_LIB_DIR)/lib
	@test -d $(ME_LIB_DIR)/inc || mkdir $(ME_LIB_DIR)/inc
	@install -m 0755 $(lib_name).$(lib_ver) $(ME_LIB_DIR)/lib
	@ln -Tfs $(ME_LIB_DIR)/lib/$(lib_name).$(lib_ver) \
		$(ME_LIB_DIR)/lib/$(lib_name).$(ME_MAJOR_NUM).$(ME_MINOR_NUM)
	@ln -Tfs $(ME_LIB_DIR)/lib/$(lib_name).$(ME_MAJOR_NUM).$(ME_MINOR_NUM) \
		$(ME_LIB_DIR)/lib/$(lib_name)
	@install -m 0644 $(core_inc_files) $(ME_LIB_DIR)/inc

# uninstall lib folder, which by default is under /opt/CARL
uninstall:
	@test -d $(ME_LIB_DIR) && $(RM) -rf $(ME_LIB_DIR)
