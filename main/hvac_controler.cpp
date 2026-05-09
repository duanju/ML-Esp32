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

#include <algorithm>
#include <rl_tools/random/operations_generic.h>
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
        rlt::reset_optimizer_state(device, optimizer, model);
        rlt::malloc(device, buffer);
        rlt::malloc(device, input_mlp);
        rlt::malloc(device, d_input_mlp);
        rlt::malloc(device, d_output_mlp);
        for(TI i = 0; i < DATASET_SIZE; i++) indices[i] = i;
        compute_normalization_stats();
    }

    float HVACControler::request(float env_status)
    {
        // Normalize input
        T normalized_input = normalize(env_status, input_min, input_max);
        rlt::set(input_mlp, 0, 0, normalized_input);
        
        // Forward pass
        rlt::forward(device, model, input_mlp, buffer, rng);
        T normalized_output = rlt::get(model.output_layer.output, 0, 0);
        
        // Denormalize output back to original scale
        T output_value = denormalize(normalized_output, output_min, output_max);
        return output_value;
    }

    T HVACControler::update()
    {
        // Compute normalization statistics on first call
        compute_normalization_stats();
        
        // compute loss and gradients, then update model parameters using the optimizer
        // train until loss is less than 0.0001 or convergence (gap < 0.001)
        T loss_avg = 1.0f;
        T prev_loss_avg = 1.0f;
        size_t epoch = 0;
        
        while (epoch < MAX_EPOCHS)
        {
            prev_loss_avg = loss_avg;
            T loss_total = 0;
            rlt::zero_gradient(device, model);
            shuffle();
            for (size_t i = 0; i < DATASET_SIZE; i++)
            {
                size_t idx = indices[i];
                T x_raw = dataset::ln_inputs[idx];
                T x = normalize(x_raw, input_min, input_max);
                rlt::set(d_input_mlp, 0, 0, x);
                T target_raw = dataset::ln_targets[idx];
                T target = normalize(target_raw, output_min, output_max);
                rlt::forward(device, model, d_input_mlp, buffer, rng);
                T output_value = get(model.output_layer.output, 0, 0);
                T loss = (output_value - target) * (output_value - target); // simple MSE loss
                loss_total += loss;
                T loss_gradient = 2.0f * (output_value - target) / DATASET_SIZE; // gradient of MSE loss w.r.t. output, averaged over the batch
                rlt::set(d_output_mlp, 0, 0, loss_gradient);
                rlt::backward(device, model, d_input_mlp, d_output_mlp, buffer);
                if (i % 100 == 0)
                {
                    printf("Sample %d, Input (raw/norm): %f/%f, Target (raw/norm): %f/%f, Output: %f, Loss: %f\n", i, x_raw, x, target_raw, target, output_value, loss);
                }
            }

            rlt::step(device, optimizer, model);
            loss_avg = loss_total / DATASET_SIZE;

            printf("Epoch %d, Loss: %f, Gap: %f\n", epoch, loss_avg, (prev_loss_avg - loss_avg));
            epoch++;
            
            // Check stopping criteria
            if (epoch > 0 && (prev_loss_avg - loss_avg) < CONVERGENCE_THRESHOLD) {
                printf("Training converged! Loss gap: %f (< %f) after %d epochs\n", (prev_loss_avg - loss_avg), CONVERGENCE_THRESHOLD, epoch);
                break;
            }
        }
        
        if (epoch >= MAX_EPOCHS) {
            printf("Training stopped! Reached max epochs (%d) with loss: %f\n", MAX_EPOCHS, loss_avg);
        }
        
        return loss_avg;
    }

    void HVACControler::shuffle() {
        for (TI i = DATASET_SIZE - 1; i > 0; --i) {
            TI j = rlt::random::uniform_int_distribution(device.random, (TI)0, i, rng);
            std::swap(indices[i], indices[j]);
        }
    }

    void HVACControler::compute_normalization_stats() {
        // Compute min and max for inputs
        input_min = dataset::ln_inputs[0];
        input_max = dataset::ln_inputs[0];
        output_min = dataset::ln_targets[0];
        output_max = dataset::ln_targets[0];
        
        for (size_t i = 1; i < DATASET_SIZE; i++) {
            if (dataset::ln_inputs[i] < input_min) input_min = dataset::ln_inputs[i];
            if (dataset::ln_inputs[i] > input_max) input_max = dataset::ln_inputs[i];
            if (dataset::ln_targets[i] < output_min) output_min = dataset::ln_targets[i];
            if (dataset::ln_targets[i] > output_max) output_max = dataset::ln_targets[i];
        }
        
        printf("Input range: [%f, %f]\n", input_min, input_max);
        printf("Output range: [%f, %f]\n", output_min, output_max);
    }

    T HVACControler::normalize(T value, T min_val, T max_val) {
        if (max_val == min_val) return 0.0f;
        return (value - min_val) / (max_val - min_val);
    }

    T HVACControler::denormalize(T value, T min_val, T max_val) {
        return value * (max_val - min_val) + min_val;
    }
} // namespace hvac
