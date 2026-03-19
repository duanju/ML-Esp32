#include <stdint.h>
#include "hvac_controler.h"

namespace hvac
{
    // HVACControler constructor
    HVACControler::HVACControler()
    {
        rlt::malloc(device, layer);
        rlt::init(device, rng, seed);
        rlt::init_weights(device, layer, rng);
    }

    float HVACControler::request(float env_status)
    {
        return 0.0f;
    }
} // namespace hvac
