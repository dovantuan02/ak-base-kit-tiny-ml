include sources/libraries/QRCode/Makefile.mk
include sources/libraries/ArduinoJson/Makefile.mk
include sources/libraries/nlohmann/Makefile.mk

# Submodules ICM-20948
CFLAGS += -I./sources/libraries/SparkFun_ICM-20948_ArduinoLibrary/src
CPPFLAGS += -I./sources/libraries/SparkFun_ICM-20948_ArduinoLibrary/src

CFLAGS += -I./sources/libraries/SparkFun_ICM-20948_ArduinoLibrary/src/util
CPPFLAGS += -I./sources/libraries/SparkFun_ICM-20948_ArduinoLibrary/src/util

VPATH += sources/libraries/SparkFun_ICM-20948_ArduinoLibrary/src
VPATH += sources/libraries/SparkFun_ICM-20948_ArduinoLibrary/src/util

SOURCES 	+= sources/libraries/SparkFun_ICM-20948_ArduinoLibrary/src/util/ICM_20948_C.c
SOURCES_CPP += sources/libraries/SparkFun_ICM-20948_ArduinoLibrary/src/ICM_20948.cpp

#Submodules emlearn
CFLAGS += -I./sources/libraries/emlearn/emlearn
CPPFLAGS += -I./sources/libraries/emlearn/emlearn

