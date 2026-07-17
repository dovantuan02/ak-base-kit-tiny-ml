#include "app_dbg.h"

#include "nn_infer.h"
#include "motion_direct_classify.h"

NNInfer::NNInfer(enum eModelName model)
{
    modelName = model;
    switch (model)
    {
    case MotionDirectClassify:
        infer = new MotionDirectInfer();
        break;
    default:
        break;
    }
    APP_DBG("Init NN for model %d, %08X\n", model, (unsigned int)infer);
}

NNInfer::~NNInfer()
{
    APP_DBG("Free NN: %08X\n", (unsigned int)infer);
    if (infer)
    {
        switch (modelName)
        {
        case MotionDirectClassify:
        {
            MotionDirectInfer *p = static_cast<MotionDirectInfer *>(infer);
            delete p;
            break;
        }
        default:
            break;
        }
    }
}

const void *NNInfer::getInfer()
{
    return infer;
}

int NNInfer::inference(void *data, uint32_t len)
{
    switch (modelName)
    {
    case MotionDirectClassify:
        return ((MotionDirectInfer *)infer)->inference(data, len);
    default:
        break;
    }
    return -1;
}

int NNInfer::inference(void *data, uint32_t len, float *output, uint32_t output_len) {
    switch (modelName)
    {
    case MotionDirectClassify:
        return ((MotionDirectInfer *)infer)->inference(data, len, output, output_len);
    default:
        break;
    }
    return -1;
}

int NNInfer::getMaxPredictClass() {
    switch (modelName)
    {
    case MotionDirectClassify:
        return MotionDirectInfer::getMaxPredictClass();
    default:
        break;
    }
    return -1;
}