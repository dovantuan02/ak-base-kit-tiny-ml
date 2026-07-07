#ifndef _ANOMAL_DETECT_H_
#define _ANOMAL_DETECT_H_

#include "nn_infer.h"
#include <stdint.h>

#define FEATURE_LEN (18)

class AnomalyInfer {
private:
    float features[FEATURE_LEN];
    float filter_state[3][3][2];

    int extract_feature(void* data, uint32_t len);

public:
    AnomalyInfer();
    ~AnomalyInfer();
    int inference(void *data, uint32_t len);
};

#endif
