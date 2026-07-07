#ifndef _TASK_ACCEL_SENSOR_H_
#define _TASK_ACCEL_SENSOR_H_

#include "ICM_20948.h"
#include "ring_buffer.h"

typedef struct {
    uint8_t ability;
    uint32_t time_irq; // timeIrq * 10ms 
    ICM_20948_I2C icm20948;
    ring_buffer_t sample_buff;
} Accel_t;

extern Accel_t accel_sensor;

extern void accel_timer_polling(Accel_t accel);
#endif
