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
#include <cmath>
#include <rl_tools/random/operations_generic.h>
#include "esp_timer.h"
#include "hvac_controler.h"
#include "dataset_loader.h"

namespace hvac
{
    // HVACControler constructor
    HVACControler::HVACControler()
    {
        rlt::malloc(device, model);
        rlt::malloc(device, optimizer);
        rlt::init_weights(device, model, rng);
        rlt::init(device, optimizer);
        rlt::zero_gradient(device, model);
        rlt::reset_optimizer_state(device, optimizer, model);
        rlt::malloc(device, buffer);
        rlt::malloc(device, input_mlp);
        rlt::malloc(device, d_input_mlp);
        rlt::malloc(device, d_output_mlp);
        compute_normalization_stats();
        build_balanced_indices();
    }

    HVACControler::~HVACControler()
    {
        delete[] indices;
    }

    void HVACControler::build_balanced_indices()
    {
        TI minority_count = 0;
        TI majority_count = 0;
        for (TI i = 0; i < TRAIN_SIZE; i++)
        {
            if (dataset::get_target(i) > 0.5f)
                minority_count++;
            else
                majority_count++;
        }
        TI repeat_factor = majority_count / minority_count;
        TI remainder = majority_count % minority_count;
        balanced_size = majority_count + minority_count * repeat_factor;
        indices = new TI[balanced_size];
        TI w = 0;
        for (TI i = 0; i < TRAIN_SIZE; i++)
        {
            TI extra = (dataset::get_target(i) > 0.5f) ? repeat_factor - 1 + (remainder > 0 ? 1 : 0) : 0;
            if (dataset::get_target(i) > 0.5f && remainder > 0)
                remainder--;
            for (TI r = 0; r < extra + 1; r++)
                indices[w++] = i;
        }
        printf("Training set: majority=%d, minority=%d, repeat_factor=%d, balanced_size=%d\n",
               (int)majority_count, (int)minority_count, (int)repeat_factor, (int)balanced_size);
    }

    float HVACControler::request(float env_status[INPUT_DIM_MLP])
    {
        // Z-score normalize 4D input
        for (TI i = 0; i < INPUT_DIM_MLP; i++)
        {
            T normalized_value = normalize(env_status[i], input_mean[i], input_std[i]);
            rlt::set(input_mlp, 0, i, normalized_value);
        }

        // Forward pass — returns probability of class 1 (sigmoid output in [0,1])
        rlt::forward(device, model, input_mlp, buffer, rng);
        return rlt::get(model.output_layer.output, 0, 0);
    }

    T HVACControler::update()
    {
        int64_t start_time = esp_timer_get_time();

        T loss_avg = 1.0f;
        T prev_loss_avg = 1.0f;
        size_t epoch = 0;

        while (epoch < MAX_EPOCHS)
        {
            prev_loss_avg = loss_avg;
            T loss_total = 0;
            rlt::zero_gradient(device, model);
            shuffle();
            T batch_loss = 0;

            for (size_t i = 0; i < balanced_size; i++)
            {
                size_t idx = indices[i];
                // Set all 4 input features with Z-score normalization
                for (TI j = 0; j < INPUT_DIM_MLP; j++)
                {
                    T x_raw = dataset::get_input(idx, j);
                    T x_normalized = normalize(x_raw, input_mean[j], input_std[j]);
                    rlt::set(d_input_mlp, 0, j, x_normalized);
                }
                T target = dataset::get_target(idx); // Keep target as raw 0/1 for BCE
                rlt::forward(device, model, d_input_mlp, buffer, rng);
                T output_value = get(model.output_layer.output, 0, 0);

                // BCE loss with numerical stability (target is raw 0/1, output is sigmoid)
                T y_clamped = std::max(std::min(output_value, 1.0f - BCE_EPSILON), BCE_EPSILON);
                T loss = -(target * std::log(y_clamped) + (1.0f - target) * std::log(1.0f - y_clamped));
                loss_total += loss;
                batch_loss += loss;

                // Gradient of BCE w.r.t. post-sigmoid output, averaged over accumulation steps
                // dL/dy = (y - t) / (y*(1-y) + eps), scaled by 1/N for mean loss over accumulation window
                T dL_dy = (output_value - target) / (output_value * (1.0f - output_value) + BCE_EPSILON) / (T)GRADIENT_ACCUMULATION_STEPS;
                rlt::set(d_output_mlp, 0, 0, dL_dy);
                rlt::backward(device, model, d_input_mlp, d_output_mlp, buffer);

                // Step optimizer every GRADIENT_ACCUMULATION_STEPS samples
                if ((i + 1) % GRADIENT_ACCUMULATION_STEPS == 0)
                {
                    rlt::step(device, optimizer, model);
                    rlt::zero_gradient(device, model);
                    printf("Step %d, avg batch loss: %f\n", (int)(i + 1), batch_loss / GRADIENT_ACCUMULATION_STEPS);
                    batch_loss = 0;
                }
            }

            // Handle remaining gradients if balanced_size is not divisible by GRADIENT_ACCUMULATION_STEPS
            if (balanced_size % GRADIENT_ACCUMULATION_STEPS != 0)
            {
                rlt::step(device, optimizer, model);
            }

            loss_avg = loss_total / balanced_size;

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

        int64_t elapsed_us = esp_timer_get_time() - start_time;
        int64_t elapsed_s = elapsed_us / 1000000;
        int h = elapsed_s / 3600;
        int m = (elapsed_s % 3600) / 60;
        int s = elapsed_s % 60;
        printf("Training time: %02d:%02d:%02d\n", h, m, s);

        return loss_avg;
    }

    void HVACControler::shuffle()
    {
        for (TI i = balanced_size - 1; i > 0; --i)
        {
            TI j = rlt::random::uniform_int_distribution(device.random, (TI)0, i, rng);
            std::swap(indices[i], indices[j]);
        }
    }

    void HVACControler::compute_normalization_stats()
    {
        // Verify dataset constants match before processing
        if (dataset::NUM_SAMPLES == 0 || dataset::NUM_FEATURES == 0)
        {
            printf("ERROR: Invalid dataset size (NUM_SAMPLES=%d, NUM_FEATURES=%d)\n", dataset::NUM_SAMPLES, dataset::NUM_FEATURES);
            return;
        }

        // Compute mean for each feature using TRAIN_SIZE samples
        for (size_t j = 0; j < dataset::NUM_FEATURES; j++)
        {
            input_mean[j] = 0;
        }
        output_mean = 0;

        for (size_t i = 0; i < TRAIN_SIZE; i++)
        {
            for (size_t j = 0; j < dataset::NUM_FEATURES; j++)
            {
                input_mean[j] += dataset::get_input(i, j);
            }
            output_mean += dataset::get_target(i);
        }
        for (size_t j = 0; j < dataset::NUM_FEATURES; j++)
        {
            input_mean[j] /= TRAIN_SIZE;
        }
        output_mean /= TRAIN_SIZE;

        // Compute standard deviation
        for (size_t j = 0; j < dataset::NUM_FEATURES; j++)
        {
            input_std[j] = 0;
        }
        output_std = 0;

        for (size_t i = 0; i < TRAIN_SIZE; i++)
        {
            for (size_t j = 0; j < dataset::NUM_FEATURES; j++)
            {
                T diff = dataset::get_input(i, j) - input_mean[j];
                input_std[j] += diff * diff;
            }
            T diff = dataset::get_target(i) - output_mean;
            output_std += diff * diff;
        }
        for (size_t j = 0; j < dataset::NUM_FEATURES; j++)
        {
            input_std[j] = std::sqrt(input_std[j] / (TRAIN_SIZE - 1));
        }
        output_std = std::sqrt(output_std / (TRAIN_SIZE - 1));

        // Print normalization stats
        for (size_t j = 0; j < dataset::NUM_FEATURES; j++)
        {
            printf("Input feature %zu: mean=%f, std=%f\n", j, input_mean[j], input_std[j]);
        }
        printf("Output: mean=%f, std=%f\n", output_mean, output_std);
    }

    T HVACControler::normalize(T value, T mean, T std)
    {
        if (std == 0.0f)
            return 0.0f;
        return (value - mean) / std;
    }

    T HVACControler::denormalize(T value, T mean, T std)
    {
        return value * std + mean;
    }

    void HVACControler::evaluate_test()
    {
        int64_t start_time = esp_timer_get_time();
        printf("Evaluating on test set (%d samples)...\n", TEST_SIZE);
        T total_loss = 0;
        T total_abs_error = 0;
        TI correct = 0;
        for (TI i = 0; i < TEST_SIZE; i++)
        {
            TI idx = dataset::TEST_START + i;
            for (TI j = 0; j < INPUT_DIM_MLP; j++)
            {
                T x_raw = dataset::get_input(idx, j);
                T x_normalized = normalize(x_raw, input_mean[j], input_std[j]);
                rlt::set(input_mlp, 0, j, x_normalized);
            }
            T target = dataset::get_target(idx); // raw 0/1, no normalization needed
            rlt::forward(device, model, input_mlp, buffer, rng);
            T output_value = get(model.output_layer.output, 0, 0);

            // BCE loss for evaluation consistency
            T y_clamped = std::max(std::min(output_value, 1.0f - BCE_EPSILON), BCE_EPSILON);
            T loss = -(target * std::log(y_clamped) + (1.0f - target) * std::log(1.0f - y_clamped));
            total_loss += loss;

            total_abs_error += std::abs(output_value - target);
            TI pred = output_value > 0.5f ? 1 : 0;
            TI label = target > 0.5f ? 1 : 0;
            if (pred == label)
                correct++;
        }
        int64_t elapsed_us = esp_timer_get_time() - start_time;
        T avg_loss = total_loss / TEST_SIZE;
        T mae = total_abs_error / TEST_SIZE;
        T accuracy = (T)correct / TEST_SIZE * 100.0f;
        printf("Test results: BCE=%f, MAE=%f, Accuracy=%d/%d (%.1f%%), Time=%lld us (%lld ms)\n",
               avg_loss, mae, (int)correct, (int)TEST_SIZE, accuracy, (long long)elapsed_us, (long long)elapsed_us / 1000);
    }
} // namespace hvac
