#include "ak.h"
#include "message.h"
#include "timer.h"
#include "task_list.h"
#include "app.h"
#include "app_dbg.h"

#include "task_accel_sensor.h"
#include "utils.h"

#include "ICM_20948.h"
#include "Wire.h"
#include "sys_ctrl.h"

#define CONVERT_G_TO_MS2(x) ((x) * 9.80665f)

#define ACCEL_SAMPLE_RATE_HZ (58)
#define ACCEL_SAMPLE_DURATION_SECONDS (2)
/* acc: x, y, z (16bit * 3), gyro: x, y, z (16bit * 3) */ 
static constexpr size_t ACCEL_DATA_SIZE = (sizeof(int16_t) * 3 + sizeof(int16_t) * 3);
static constexpr size_t ACCEL_SAMPLE_BUFFER_SIZE = (ACCEL_SAMPLE_DURATION_SECONDS * ACCEL_SAMPLE_RATE_HZ * ACCEL_DATA_SIZE);

static uint8_t buffer[ACCEL_SAMPLE_BUFFER_SIZE];
Accel_t accel_sensor;

void task_accel(ak_msg_t *msg)
{
    switch (msg->sig)
    {
    case AC_ACCEL_INIT:
    {
        accel_sensor.ability = AK_DISABLE;
        accel_sensor.time_irq = 1;

        Wire.begin();
        Wire.setClock(50000);
        accel_sensor.icm20948.begin(Wire, true);
        APP_DBG_SIG("AC_ACCEL_INIT\n");
        APP_DBG("Sensor init: %s\n", accel_sensor.icm20948.statusString());
        if (accel_sensor.icm20948.status != ICM_20948_Stat_Ok)
        {
            timer_set(AC_TASK_ACCEL_ID, AC_ACCEL_INIT, 200, TIMER_ONE_SHOT);
            break;
        }
        task_post_pure_msg(AC_TASK_ACCEL_ID, AC_ACCEL_SET_CONFIG);
        APP_DBG("Init ring buffer: %d bytes\n", sizeof(buffer));
        ring_buffer_init(&accel_sensor.sample_buff, buffer, sizeof(buffer), ACCEL_DATA_SIZE);
    }
    break;

    case AC_ACCEL_SET_CONFIG:
    {
        APP_DBG_SIG("AC_ACCEL_SET_CONFIG\n");
        accel_sensor.ability = AK_DISABLE;
        bool success = true;
        success &= (accel_sensor.icm20948.initializeDMP() == ICM_20948_Stat_Ok);
        if (!success)
        {
            APP_DBG("Init DMP failed !\n");
            goto retry;
        }
        else
        {
            APP_DBG("Init DMP success !\n");
        }
        // DMP sensor options are defined in ICM_20948_DMP.h
        //    INV_ICM20948_SENSOR_ACCELEROMETER               (16-bit accel)
        //    INV_ICM20948_SENSOR_GYROSCOPE                   (16-bit gyro + 32-bit calibrated gyro)
        //    INV_ICM20948_SENSOR_RAW_ACCELEROMETER           (16-bit accel)
        //    INV_ICM20948_SENSOR_RAW_GYROSCOPE               (16-bit gyro + 32-bit calibrated gyro)
        //    INV_ICM20948_SENSOR_MAGNETIC_FIELD_UNCALIBRATED (16-bit compass)
        //    INV_ICM20948_SENSOR_GYROSCOPE_UNCALIBRATED      (16-bit gyro)
        //    INV_ICM20948_SENSOR_STEP_DETECTOR               (Pedometer Step Detector)
        //    INV_ICM20948_SENSOR_STEP_COUNTER                (Pedometer Step Detector)
        //    INV_ICM20948_SENSOR_GAME_ROTATION_VECTOR        (32-bit 6-axis quaternion)
        //    INV_ICM20948_SENSOR_ROTATION_VECTOR             (32-bit 9-axis quaternion + heading accuracy)
        //    INV_ICM20948_SENSOR_GEOMAGNETIC_ROTATION_VECTOR (32-bit Geomag RV + heading accuracy)
        //    INV_ICM20948_SENSOR_GEOMAGNETIC_FIELD           (32-bit calibrated compass)
        //    INV_ICM20948_SENSOR_GRAVITY                     (32-bit 6-axis quaternion)
        //    INV_ICM20948_SENSOR_LINEAR_ACCELERATION         (16-bit accel + 32-bit 6-axis quaternion)
        //    INV_ICM20948_SENSOR_ORIENTATION                 (32-bit 9-axis quaternion + heading accuracy)

        success &= (accel_sensor.icm20948.enableDMPSensor(INV_ICM20948_SENSOR_ACCELEROMETER) == ICM_20948_Stat_Ok);
        // Set to the maximum
        success &= (accel_sensor.icm20948.setDMPODRrate(DMP_ODR_Reg_Accel, 0) == ICM_20948_Stat_Ok);
        success &= (accel_sensor.icm20948.enableFIFO() == ICM_20948_Stat_Ok);
        success &= (accel_sensor.icm20948.enableDMP() == ICM_20948_Stat_Ok);
        success &= (accel_sensor.icm20948.resetDMP() == ICM_20948_Stat_Ok);
        success &= (accel_sensor.icm20948.resetFIFO() == ICM_20948_Stat_Ok);
        if (!success)
        {
            APP_DBG("Config DMP failed !\n");
            goto retry;
        }
        APP_DBG("Config DMP success !\n");
        accel_sensor.ability = AK_ENABLE;
        break;

    retry:
    {
        timer_set(AC_TASK_ACCEL_ID, AC_ACCEL_SET_CONFIG, 200, TIMER_ONE_SHOT);
    }
    }
    break;

    case AC_ACCEL_GET_CONFIG:
    {
        APP_DBG_SIG("AC_ACCEL_GET_CONFIG\n");
    }
    break;

    default:
        break;
    }
}

void accel_timer_polling(Accel_t accel)
{
    struct icm_data_internal_t {
        int16_t acc_x;
        int16_t acc_y;
        int16_t acc_z;
        int16_t gyro_x;
        int16_t gyro_y;
        int16_t gyro_z;

    };
    if (accel.ability == AK_ENABLE)
    {
        icm_20948_DMP_data_t data;
        accel_sensor.icm20948.readDMPdataFromFIFO(&data);

        float x, y, z;
        float gyro_x, gyro_y, gyro_z;
        x = y = z = 0.0f;
        gyro_x = gyro_y = gyro_z = 0.0f;
        if ((accel_sensor.icm20948.status == ICM_20948_Stat_Ok) || (accel_sensor.icm20948.status == ICM_20948_Stat_FIFOMoreDataAvail))
        {
            if ((data.header & DMP_header_bitmap_Accel) > 0)
            {
                x = CONVERT_G_TO_MS2((float)data.Raw_Accel.Data.X);
                y = CONVERT_G_TO_MS2((float)data.Raw_Accel.Data.Y);
                z = CONVERT_G_TO_MS2((float)data.Raw_Accel.Data.Z);

                #if 0
                xfprintf((void (*)(int))sys_ctrl_shell_put_char, "%.1f,%.1f,%.1f\n", x, y, z);
                #else
                struct icm_data_internal_t icm_data = {
                    .acc_x = data.Raw_Accel.Data.X,
                    .acc_y = data.Raw_Accel.Data.Y,
                    .acc_z = data.Raw_Accel.Data.Z,
                    .gyro_x = 0,
                    .gyro_y = 0,
                    .gyro_z = 0
                };
                if (!ring_buffer_is_full(&accel_sensor.sample_buff)) {
                    ring_buffer_put(&accel_sensor.sample_buff, &icm_data);
                    // APP_DBG("Put data to ring buffer: %d bytes, ringAvail: %d\n", sizeof(icm_data), ring_buffer_availble(&accel_sensor.sample_buff));
                }
                #endif
            }

            
        }
        else
        {
            // APP_DBG("Status: %s\n", accel_sensor.icm20948.statusString());
            if (accel_sensor.icm20948.status == ICM_20948_Stat_FIFOIncompleteData)
            {
                task_post_pure_msg(AC_TASK_ACCEL_ID, AC_ACCEL_SET_CONFIG);
            }
        }
    }
}
