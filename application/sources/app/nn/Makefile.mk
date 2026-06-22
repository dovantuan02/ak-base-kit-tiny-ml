CFLAGS		+= -I./sources/app/nn -I./sources/app/screens -I./sources/driver/Adafruit_oled_drv
CPPFLAGS	+= -I./sources/app/nn -I./sources/app/screens -I./sources/driver/Adafruit_oled_drv

VPATH += sources/app/nn

# CPP source files
SOURCES += sources/app/nn/BitNetMCU.c
SOURCES += sources/app/nn/BitNetMCU_inference.c
