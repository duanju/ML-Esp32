#ifndef HVAC_CONTROLER_H
#define HVAC_CONTROLER_H

// First include the device operations (ESP32 in your case)
// #include <rl_tools/operations/esp32.h>
#include <rl_tools/rl_tools.h>
#include <rl_tools/devices/esp32.h>
// Then include other RLtools components
#include <rl_tools/utils/assert/operations_esp32.h>
#include <rl_tools/math/operations_esp32.h>
#include <rl_tools/random/operations_generic.h>
#include <rl_tools/containers/matrix/operations_generic.h>
#include <rl_tools/containers/tensor/tensor.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/numeric_types/policy.h>
#include <rl_tools/nn_models/mlp/network.h>
#include "dataset_loader.h"

namespace rlt = rl_tools;
// add HVACControler in namespace hvac
namespace hvac
{
    using DEV_SPEC_S3 = rlt::devices::DefaultESP32Specification<rlt::devices::esp32::Hardware::S3>;
    using DEVICE = rlt::devices::ESP32<DEV_SPEC_S3>;
    using TI = typename DEVICE::index_t;
    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;

    constexpr TI INPUT_DIM_MLP = dataset::NUM_FEATURES;
    constexpr TI OUTPUT_DIM_MLP = 1;
    constexpr TI NUM_LAYERS = 6;
    constexpr TI HIDDEN_DIM = 32;
    constexpr TI BATCH_SIZE = 1;       // Since the controler is used in an online setting, the batch size is set to 1. However, if you want to use the controler in an offline setting, you can increase the batch size and modify the request function accordingly.
    constexpr TI TRAIN_SIZE = dataset::TRAIN_SIZE;
    constexpr TI TEST_SIZE = dataset::TEST_SIZE;
    constexpr size_t MAX_EPOCHS = 10000; // Safety limit to prevent infinite loops
    constexpr TI GRADIENT_ACCUMULATION_STEPS = 64; // Number of samples to accumulate gradients over before calling step()
    constexpr T BCE_EPSILON = 1e-7f;
    constexpr T CONVERGENCE_THRESHOLD = 0.00001f;
    constexpr T POS_WEIGHT = 6.0f; // Majority/minority ratio for class-weighted BCE

    constexpr auto ACTIVATION_FUNCTION_MLP = rlt::nn::activation_functions::GELU;
    constexpr auto OUTPUT_ACTIVATION_FUNCTION_MLP = rlt::nn::activation_functions::SIGMOID;
    using MODEL_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, OUTPUT_DIM_MLP, NUM_LAYERS, HIDDEN_DIM, ACTIVATION_FUNCTION_MLP, OUTPUT_ACTIVATION_FUNCTION_MLP>;

    using PARAMETER_TYPE = rlt::nn::parameters::Adam;
    using CAPABILITY = rlt::nn::capability::Gradient<PARAMETER_TYPE>;
    using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, INPUT_DIM_MLP>;
    using MODEL_TYPE = rlt::nn_models::mlp::NeuralNetwork<MODEL_CONFIG, CAPABILITY, INPUT_SHAPE>;

    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;

    class HVACControler
    {
    public:
        HVACControler();
        ~HVACControler();
        // Request control action based on 4-dimensional environment status
        // env_status: pointer to array of 4 input features
        float request(float env_status[INPUT_DIM_MLP]); // request control action based on 4D status
        T update();
        void evaluate_test();

    private:
        // internal state and methods for the controler
        DEVICE device;
        RNG rng;
        TI seed = 1;

        OPTIMIZER optimizer;
        MODEL_TYPE model;
        typename MODEL_TYPE::Buffer<> buffer;

        rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, INPUT_DIM_MLP>> input_mlp, d_input_mlp;
        rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, OUTPUT_DIM_MLP>> d_output_mlp;

        TI* indices;
        void shuffle();

        // Z-score normalization parameters
        T input_mean[INPUT_DIM_MLP], input_std[INPUT_DIM_MLP];
        T output_mean, output_std;
        void compute_normalization_stats();
        T normalize(T value, T mean, T std);
        T denormalize(T value, T mean, T std);
    };
} // namespace hvac

#endif