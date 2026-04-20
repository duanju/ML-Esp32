#include <stdio.h>
#include <rl_tools/rl_tools.h>
#include <rl_tools/utils/generic/typing.h>
#include <rl_tools/devices/esp32.h>
#include <rl_tools/utils/assert/operations_esp32.h>
#include <rl_tools/math/operations_esp32.h>
#include <rl_tools/random/operations_generic.h>
#include <rl_tools/containers/matrix/operations_generic.h>
#include <rl_tools/containers/tensor/operations_generic.h>

#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

// #include <rl_tools/containers/tensor/tensor.h>
#include "hvac_controler.h"
#include "data/test_backprop_tools_nn_models_mlp_training.h"

namespace hvac
{
    // HVACControler constructor
    HVACControler::HVACControler()
    {
        rlt::malloc(device, model);
        rlt::malloc(device, optimizer);
        rlt::init_weights(device, model, rng); // recursively initializes all layers using kaiming initialization
        rlt::init(device, optimizer);
        rlt::zero_gradient(device, model); // recursively zeros all gradients in the layers
        // rlt::reset_optimizer_state(device, optimizer, model);
        rlt::malloc(device, buffer);
        rlt::malloc(device, input_mlp);
        rlt::malloc(device, d_input_mlp);
        rlt::malloc(device, d_output_mlp);
    }

    float HVACControler::request(float env_status)
    {
        rlt::randn(device, input_mlp, rng);
        rlt::forward(device, model, input_mlp, buffer, rng);
        T output_value = rlt::get(model.output_layer.output, 0, 0);
        return output_value;
    }

    T HVACControler::update()
    {
        // compute loss and gradients, then update model parameters using the optimizer
        // train for 100 iterations
        T loss_avg = 0;
        for (size_t batch = 0; batch < 10; batch++)
        {
            T loss_total = 0;
            rlt::zero_gradient(device, model);
            for (size_t i = 0; i < 100; i++)
            {
                T x = dataset::ln_inputs[i];
                rlt::set(d_input_mlp, 0, 0, x);
                T target = dataset::ln_targets[i];
                rlt::forward(device, model, d_input_mlp, buffer, rng);
                T output_value = get(model.output_layer.output, 0, 0);
                T loss = (output_value - target) * (output_value - target); // simple MSE loss
                loss_total += loss;
                T loss_gradient = 2.0f * (output_value - target);
                rlt::set(d_output_mlp, 0, 0, loss_gradient);
                rlt::backward(device, model, d_input_mlp, d_output_mlp, buffer);
            }

            rlt::step(device, optimizer, model);
            loss_avg = loss_total / 100;

            printf("Batch %d, Loss: %f\n", batch, loss_avg);
        }
        return loss_avg;
    }
} // namespace hvac
