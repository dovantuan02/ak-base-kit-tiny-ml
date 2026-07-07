#ifndef __NN_INFERENCE_H_
#define __NN_INFERENCE_H_
#include <stdint.h>

enum eModelName {
    ImpactDetect,
    AnomalyDetect,
};

class NNInfer
{
private:
    void *infer;
    enum eModelName modelName;
public:
    NNInfer(enum eModelName);
    NNInfer(enum eModelName, void *model);
    ~NNInfer();

    const void* getInfer();

    int inference(void *data, uint32_t len);
};



#endif