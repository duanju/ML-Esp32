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
#include "dataset_loader.h"

namespace hvac
{
    // HVACControler constructor
    HVACControler::HVACControler()
    {
        indices = new TI[DATASET_SIZE];
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
        for (TI i = 0; i < DATASET_SIZE; i++)
            indices[i] = i;
        compute_normalization_stats();
    }

    HVACControler::~HVACControler()
    {
        delete[] indices;
    }

    float HVACControler::request(float env_status[INPUT_DIM_MLP])
    {
        // Normalize 4D input
        for (TI i = 0; i < INPUT_DIM_MLP; i++)
        {
            T normalized_value = normalize(env_status[i], input_min[i], input_max[i]);
            rlt::set(input_mlp, 0, i, normalized_value);
        }

        // Forward pass
        rlt::forward(device, model, input_mlp, buffer, rng);
        T normalized_output = rlt::get(model.output_layer.output, 0, 0);

        // Denormalize output back to original scale
        T output_value = denormalize(normalized_output, output_min, output_max);
        return output_value;
    }

    T HVACControler::update()
    {

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
                // Set all 4 input features with normalization
                for (TI j = 0; j < INPUT_DIM_MLP; j++)
                {
                    T x_raw = dataset::get_input(idx, j);
                    T x_normalized = normalize(x_raw, input_min[j], input_max[j]);
                    rlt::set(d_input_mlp, 0, j, x_normalized);
                }
                T target_raw = dataset::get_target(idx);
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
                    printf("Sample %d, Target (raw/norm): %f/%f, Output: %f, Loss: %f\n", i, target_raw, target, output_value, loss);
                }
            }

            rlt::step(device, optimizer, model);
            loss_avg = loss_total / DATASET_SIZE;

            printf("Epoch %d, Loss: %f, Gap: %f\n", epoch, loss_avg, (prev_loss_avg - loss_avg));
            epoch++;

            // Check stopping criteria
            if (epoch > 0 && (prev_loss_avg - loss_avg) < CONVERGENCE_THRESHOLD)
            {
                printf("Training converged! Loss gap: %f (< %f) after %d epochs\n", (prev_loss_avg - loss_avg), CONVERGENCE_THRESHOLD, epoch);
                break;
            }
        }

        if (epoch >= MAX_EPOCHS)
        {
            printf("Training stopped! Reached max epochs (%d) with loss: %f\n", MAX_EPOCHS, loss_avg);
        }

        return loss_avg;
    }

    void HVACControler::shuffle()
    {
        for (TI i = DATASET_SIZE - 1; i > 0; --i)
        {
            TI j = rlt::random::uniform_int_distribution(device.random, (TI)0, i, rng);
            std::swap(indices[i], indices[j]);
        }
    }

    void HVACControler::compute_normalization_stats()
    {
        // Verify dataset constants match before processing
        if (DATASET_SIZE == 0 || dataset::NUM_FEATURES == 0)
        {
            printf("ERROR: Invalid dataset size (NUM_SAMPLES=%d, NUM_FEATURES=%d)\n", DATASET_SIZE, dataset::NUM_FEATURES);
            return;
        }

        // Initialize min and max for each feature using first sample
        for (size_t j = 0; j < dataset::NUM_FEATURES; j++)
        {
            input_min[j] = dataset::get_input(0, j);
            input_max[j] = dataset::get_input(0, j);
        }
        output_min = dataset::get_target(0);
        output_max = dataset::get_target(0);

        // printf("Starting normalization stats computation for %d samples...\n", DATASET_SIZE);

        // Compute min and max for each feature by iterating through all samples
        for (size_t i = 1; i < DATASET_SIZE; i++)
        {
            // Bounds check for safety
            if (i >= dataset::NUM_SAMPLES)
            {
                printf("WARNING: Index i=%zu >= NUM_SAMPLES=%d, stopping\n", i, dataset::NUM_SAMPLES);
                break;
            }

            for (size_t j = 0; j < dataset::NUM_FEATURES; j++)
            {
                float val = dataset::get_input(i, j);
                if (val < input_min[j])
                    input_min[j] = val;
                if (val > input_max[j])
                    input_max[j] = val;
            }

            // Compute output min/max
            float target = dataset::get_target(i);
            if (target < output_min)
                output_min = target;
            if (target > output_max)
                output_max = target;

            // Print progress every 1000 samples
            if (i % 1000 == 0)
            {
                printf("Processing sample %zu/%d...\n", i, DATASET_SIZE);
            }
        }

        // Print normalization stats for each input feature
        for (size_t j = 0; j < dataset::NUM_FEATURES; j++)
        {
            printf("Input feature %zu range: [%f, %f]\n", j, input_min[j], input_max[j]);
        }
        printf("Output range: [%f, %f]\n", output_min, output_max);
    }

    T HVACControler::normalize(T value, T min_val, T max_val)
    {
        if (max_val == min_val)
            return 0.0f;
        return (value - min_val) / (max_val - min_val);
    }

    T HVACControler::denormalize(T value, T min_val, T max_val)
    {
        return value * (max_val - min_val) + min_val;
    }
} // namespace hvac
