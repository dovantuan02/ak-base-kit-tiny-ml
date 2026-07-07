CFLAGS      += -I./sources/platform/stm32l/Libraries/CMSIS-DSP/Include
CFLAGS      += -I./sources/platform/stm32l/Libraries/CMSIS-DSP/PrivateInclude
CFLAGS      += -DARM_MATH_LOOPUNROLL

CPPFLAGS    += -I./sources/platform/stm32l/Libraries/CMSIS-DSP/Include
CPPFLAGS    += -I./sources/platform/stm32l/Libraries/CMSIS-DSP/PrivateInclude
CPPFLAGS    += -DARM_MATH_LOOPUNROLL

VPATH += sources/platform/stm32l/Libraries/CMSIS-DSP/Source/FilteringFunctions
VPATH += sources/platform/stm32l/Libraries/CMSIS-DSP/Source/StatisticsFunctions
VPATH += sources/platform/stm32l/Libraries/CMSIS-DSP/Source/TransformFunctions
VPATH += sources/platform/stm32l/Libraries/CMSIS-DSP/Source/BasicMathFunctions
VPATH += sources/platform/stm32l/Libraries/CMSIS-DSP/Source/CommonTables

SOURCES += sources/platform/stm32l/Libraries/CMSIS-DSP/Source/FilteringFunctions/arm_biquad_cascade_df2T_f32.c
SOURCES += sources/platform/stm32l/Libraries/CMSIS-DSP/Source/FilteringFunctions/arm_biquad_cascade_df2T_init_f32.c

SOURCES += sources/platform/stm32l/Libraries/CMSIS-DSP/Source/StatisticsFunctions/arm_rms_f32.c
SOURCES += sources/platform/stm32l/Libraries/CMSIS-DSP/Source/StatisticsFunctions/arm_mean_f32.c

SOURCES += sources/platform/stm32l/Libraries/CMSIS-DSP/Source/TransformFunctions/arm_cfft_f32.c
SOURCES += sources/platform/stm32l/Libraries/CMSIS-DSP/Source/TransformFunctions/arm_cfft_radix8_f32.c
SOURCES += sources/platform/stm32l/Libraries/CMSIS-DSP/Source/TransformFunctions/arm_bitreversal2.c

SOURCES += sources/platform/stm32l/Libraries/CMSIS-DSP/Source/BasicMathFunctions/arm_offset_f32.c

SOURCES += sources/platform/stm32l/Libraries/CMSIS-DSP/Source/CommonTables/arm_const_structs.c
SOURCES += sources/platform/stm32l/Libraries/CMSIS-DSP/Source/CommonTables/arm_common_tables.c
