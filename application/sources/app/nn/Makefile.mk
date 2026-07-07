include sources/app/nn/model/Makefile.mk
include sources/app/nn/inference/impact_detect/Makefile.mk
include sources/app/nn/inference/anomal_detect/Makefile.mk

CFLAGS		+= -I./sources/app/nn
CPPFLAGS	+= -I./sources/app/nn


VPATH += sources/app/nn

# CPP source files
SOURCES_CPP += sources/app/nn/nn_infer.cpp
