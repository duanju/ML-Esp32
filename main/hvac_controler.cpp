#include <stdio.h>
// #include <rl_tools/operations/esp32.h>
// #include <rl_tools/nn_models/sequential/operations_generic.h>
#include "hvac_controler.h"

namespace hvac
{
    // HVACControler constructor
    HVACControler::HVACControler()
    {
        // rlt::malloc(device, sequential);
        // rlt::malloc(device, sequential_buffer);
        // rlt::init_weights(device, sequential, rng);
    }

    float HVACControler::request(float env_status)
    {
        // rlt::Tensor<rlt::tensor::Specification<T, TI, typename SEQUENTIAL::INPUT_SHAPE, false>> input;
        // rlt::Tensor<rlt::tensor::Specification<T, TI, typename SEQUENTIAL::OUTPUT_SHAPE, false>> output;

        // Manually set values at each index
        // rlt::set(device, input, 0, 1.5f);
        // rlt::set(device, input, 1, 2.0f);
        // rlt::set(device, input, 2, -0.5f);
        // rlt::set(device, input, 3, 3.14f);
        // rlt::set(device, input, 4, 0.0f);

        // rlt::forward(device, sequential, input, sequential_buffer, rng);
        // print output for debugging
        // printf("Output: ");
        // for (TI i = 0; i < output.shape().size(); ++i)        {
        //     printf("%f ", rlt::get(device, output, i));
        // }
        printf("\n");
        return 0.0f;
    }
} // namespace hvac
