#ifndef _IMPACT_DETECT_H_
#define _IMPACT_DETECT_H_

#include "nn_infer.h"

#define FEATURE_LEN (8) /* peak, max, min, energy, mean, std, max slope, mean abs slope */

class ImpactInfer
{
private:
    int16_t features[FEATURE_LEN];
    int extract_feature(void* data, uint32_t len);
public:
    ImpactInfer(/* args */);
    ~ImpactInfer();

    int inference(void *data, uint32_t len);
};

#endif