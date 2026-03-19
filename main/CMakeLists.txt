


# idf_component_register(SRCS "main.c" "rl_tools_adapter.cpp" INCLUDE_DIRS "." "../../rl-tools/include")
idf_component_register(SRCS "main.cpp" "hvac_controler.cpp" INCLUDE_DIRS "." "../../rl-tools/include" REQUIRES esp-dsp)
idf_build_set_property(COMPILE_OPTIONS "-Wno-error=unused-local-typedefs" APPEND)
