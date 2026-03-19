#include <stdio.h>
#include <rl_tools/operations/esp32.h>
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
        printf("Received environment status: %f\n", env_status);

        for (size_t i = 0; i < INPUT_DIM; i++)
        {
            /* code */
            for (size_t j = 0; j < OUTPUT_DIM; j++)
            {
                /* code */
                printf("Weight at [%d,%d]: %f\n", i, j, rlt::get(device, layer.weights.parameters, i, j));
            }
            printf("Bias at [%d]: %f\n", i, rlt::get(device, layer.biases.parameters, i));
        }


        return 0.0f;
    }
} // namespace hvac
