#include <stdio.h>
#include <stdint.h>

#include "app_dbg.h"
#include "mic.h"

// #include "BitNetMCU_data.h"
#include "BitNetMCU_inference.h"

/* forward decl — avoids deep include chain of screens.h */
extern void scr_mic_wave_handle(void);

/**************************************
 *             MODEL SELECT
 **************************************/
// #include "BitNetMCU_model_test.h"
// #include "BitNetMCU_model_1k.h"
// #include "BitNetMCU_model_12k.h"
// #include "BitNetMCU_model_12k_FP130.h"
// #include "BitNetMCU_model_cnn_48.h"
// #include "BitNetMCU_model_cnn_32.h"
// #include "BitNetMCU_model_cnn_16.h"
// #include "BitNetMCU_model_cnn_16small.h"
// #include "BitNetMCU_model_cnn_64.h"
// #include "BitNetMCU_model_cnn_letters.h"
#include "mnist_model.h"

#ifdef MODEL_CNNMNIST

uint32_t BitMnistInference(const int8_t *input)
{
    static int32_t layer_out[256]; // has to hold 16x16 image
    static int8_t layer_in[MAX_N_ACTIVATIONS * 4];
    /*
        Layer: L2 Conv2d bpw: 8 1 -> 64 groups:1 Kernel: 3x3 Incoming: 16x16 Outgoing: 14x14
        Layer: L4 Conv2d bpw: 8 64 -> 64 groups:64 Kernel: 3x3 Incoming: 14x14 Outgoing: 12x12
        Layer: L6 MaxPool2d Pool Size: 2 Incoming: 12x12 Outgoing: 6x6
        Layer: L7 Conv2d bpw: 8 64 -> 64 groups:64 Kernel: 3x3 Incoming: 6x6 Outgoing: 4x4
        Layer: L9 MaxPool2d Pool Size: 2 Incoming: 4x4 Outgoing: 2x2
        Layer: L11 Quantization type: <2bitsym>, Bits per weight: 2, Num. incoming: 256,  Num outgoing: 96
        Layer: L13 Quantization type: <4bitsym>, Bits per weight: 4, Num. incoming: 96,  Num outgoing: 64
        Layer: L15 Quantization type: <4bitsym>, Bits per weight: 4, Num. incoming: 64,  Num outgoing: 10
    */

    memset(layer_out, 0, sizeof(layer_out));
    memset(layer_in, 0, sizeof(layer_in));

    APP_DBG("\n=== BitMnistInference START ===\n");

    // --- Step 1: Copy raw int8 input into int32 working buffer ---
    int32_t *buf = (int32_t *)layer_out;
    APP_DBG("  Copying input (%d int8 samples)\n", 16 * 16);
    for (uint32_t i = 0; i < 16 * 16; i++)
    {
        buf[i] = input[i];
    }

    // --- Step 2: Depthwise separable conv (per channel) ---
    // Each channel: L2 conv → L4 conv → L6 maxpool → L7 conv → L9 maxpool
    int32_t *tmpbuf = (int32_t *)layer_out;
    int32_t *outputptr = (int32_t *)layer_in;
    APP_DBG("  Channel loop: %u channels total\n", L7_out_channels);
    for (uint32_t channel = 0; channel < L7_out_channels; channel++)
    {
        // Re-init from input each channel
        for (uint32_t i = 0; i < 16 * 16; i++)
            tmpbuf[i] = buf[i];

        processconv33ReLU(tmpbuf, L2_weights + 9 * channel, L2_incoming_x, 4, tmpbuf);
        processconv33ReLU(tmpbuf, L4_weights + 9 * channel, L4_incoming_x, 4, tmpbuf);
        processmaxpool22(tmpbuf, L6_incoming_x, tmpbuf);
        processconv33ReLU(tmpbuf, L7_weights + 9 * channel, L7_incoming_x, 4, tmpbuf);
        outputptr = processmaxpool22(tmpbuf, L9_incoming_x, outputptr);
    }

    // Combined conv output

    // --- Step 3: Normalize conv output to int8 ---
    APP_DBG("  ReLUNorm (conv→FC): %u inputs\n", L7_out_channels * L9_outgoing_x * L9_outgoing_y);
    ReLUNorm((int32_t *)layer_in, layer_in, L7_out_channels * L9_outgoing_x * L9_outgoing_y);

    // --- Step 4: Fully connected layers ---
    // L11: 256→96  (2-bit weights)
    APP_DBG("  FC L11: %u→%u (bpw=%lu)\n", L11_incoming_weights, L11_outgoing_weights, L11_bitperweight);
    processfclayer(layer_in, L11_weights, L11_bitperweight, L11_incoming_weights, L11_outgoing_weights, layer_out);

    ReLUNorm(layer_out, layer_in, L11_outgoing_weights);

    // L13: 96→64  (4-bit weights)
    APP_DBG("  FC L13: %u→%u (bpw=%lu)\n", L13_incoming_weights, L13_outgoing_weights, L13_bitperweight);
    processfclayer(layer_in, L13_weights, L13_bitperweight, L13_incoming_weights, L13_outgoing_weights, layer_out);

    ReLUNorm(layer_out, layer_in, L13_outgoing_weights);

    // L15: 64→10  (4-bit weights) — final
    APP_DBG("  FC L15: %u→%u (bpw=%lu)\n", L15_incoming_weights, L15_outgoing_weights, L15_bitperweight);
    processfclayer(layer_in, L15_weights, L15_bitperweight, L15_incoming_weights, L15_outgoing_weights, layer_out);

    uint32_t prediction = ReLUNorm(layer_out, layer_in, L15_outgoing_weights);

    APP_DBG("=== BitMnistInference DONE → prediction=%lu ===\n", prediction);
    return prediction;
}

#elif defined(MODEL_FCMNIST)

uint32_t BitMnistInference(const int8_t *input)
{
    int32_t layer_out[MAX_N_ACTIVATIONS];
    int8_t layer_in[MAX_N_ACTIVATIONS];

    APP_DBG("\n=== BitMnistInference (FC) START ===\n");

    processfclayer((int8_t *)input, L1_weights, L1_bitperweight, L1_incoming_weights, L1_outgoing_weights, layer_out);
    ReLUNorm(layer_out, layer_in, L1_outgoing_weights);

    processfclayer(layer_in, L2_weights, L2_bitperweight, L2_incoming_weights, L2_outgoing_weights, layer_out);
    ReLUNorm(layer_out, layer_in, L2_outgoing_weights);

    processfclayer(layer_in, L3_weights, L3_bitperweight, L3_incoming_weights, L3_outgoing_weights, layer_out);
    uint32_t prediction = ReLUNorm(layer_out, layer_in, L3_outgoing_weights);

#if NUM_LAYERS == 4
    processfclayer(layer_in, L4_weights, L4_bitperweight, L4_incoming_weights, L4_outgoing_weights, layer_out);
    prediction = ReLUNorm(layer_out, layer_in, L4_outgoing_weights);
#endif

    APP_DBG("=== BitMnistInference DONE → prediction=%lu ===\n", prediction);
    return prediction;
}

#endif

void TestSample(const int8_t *input, const uint8_t label, const uint8_t sample)
{
    volatile uint32_t startticks, endticks;
    int32_t prediction;
    int16_t features[256];

    APP_DBG("--- TestSample %u ---\n", sample);
    // prediction = BitMnistInference(input);
    startticks = sys_ctrl_millis();

    for(int i=0;i<256;i++)
    {
        features[i] = input[i];
    }
    prediction = mnist_model_predict(features, 256);
    endticks = sys_ctrl_millis();

    APP_DBG("Sample %u: prediction=%ld  expected=%u -> timing inference: (%d)ms %s\n",
            sample, prediction, label, (endticks - startticks),
            (prediction == label) ? "OK" : "FAIL");
}

void task_polling_ml()
{
    // APP_DBG("Starting MNIST inference...\n");

    // TestSample(input_data_0, label_0, 1);
    // TestSample(input_data_1, label_1, 2);
    // TestSample(input_data_2, label_2, 3);
    // scr_mic_wave_handle();
}
