#ifndef __NN_INFERENCE_H_
#define __NN_INFERENCE_H_
#include <stdint.h>

enum eModelName
{
    ImpactDetect,
    MotionDirectClassify,
};

class NNInfer
{
private:
    void *infer;
    enum eModelName modelName;

public:
    NNInfer(enum eModelName);
    ~NNInfer();

    const void *getInfer();

    int inference(void *data, uint32_t len);
    int inference(void *data, uint32_t len, float *output, uint32_t output_len);

    int getMaxPredictClass();
};

#endif