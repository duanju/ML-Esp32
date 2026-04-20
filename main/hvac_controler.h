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

namespace rlt = rl_tools;
// add HVACControler in namespace hvac
namespace hvac
{
    using DEV_SPEC_S3 = rlt::devices::DefaultESP32Specification<rlt::devices::esp32::Hardware::C3>;
    using DEVICE = rlt::devices::ESP32<DEV_SPEC_S3>;
    using TI = typename DEVICE::index_t;
    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;

    constexpr TI INPUT_DIM_MLP = 1;
    constexpr TI OUTPUT_DIM_MLP = 1;
    constexpr TI NUM_LAYERS = 3;
    constexpr TI HIDDEN_DIM = 32;
    constexpr TI BATCH_SIZE = 1; // Since the controler is used in an online setting, the batch size is set to 1. However, if you want to use the controler in an offline setting, you can increase the batch size and modify the request function accordingly.
    constexpr TI TRAINING_EPOCHS = 100; // Since the controler is used in an online setting, the batch size is set to 1. However, if you want to use the controler in an offline setting, you can increase the batch size and modify the request function accordingly.

    constexpr auto ACTIVATION_FUNCTION_MLP = rlt::nn::activation_functions::RELU;
    constexpr auto OUTPUT_ACTIVATION_FUNCTION_MLP = rlt::nn::activation_functions::IDENTITY;
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
        // todo: change the signature of the request function to fit your needs,
        // e.g., you might want to pass in more information about the environment status
        float request(float env_status); // request control action based on status
        T update();
    private :
        // internal state and methods for the controler
        DEVICE device;
        RNG rng;
        TI seed = 1;

        OPTIMIZER optimizer;
        MODEL_TYPE model;
        typename MODEL_TYPE::Buffer<> buffer;

        rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, INPUT_DIM_MLP>> input_mlp, d_input_mlp;
        rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, OUTPUT_DIM_MLP>> d_output_mlp;
    };
} // namespace hvac

#endif