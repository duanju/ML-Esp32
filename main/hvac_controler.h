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
namespace rlt = rl_tools;
// add HVACControler in namespace hvac
namespace hvac
{
    using DEV_SPEC_S3 = rlt::devices::DefaultESP32Specification<rlt::devices::esp32::Hardware::C3>;
    using DEVICE = rlt::devices::ESP32<DEV_SPEC_S3>;
    using T = float;
    // using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
    // using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    // using TI = typename DEVICE::index_t;

    // constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::RELU;

    // using LAYER_1_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 32,
    //                                                              ACTIVATION_FUNCTION>;
    // using Layer_1 = rlt::nn::layers::dense::BindConfiguration<LAYER_1_CONFIG>;
    // using LAYER_2_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 16,
    //                                                              ACTIVATION_FUNCTION>;
    // using Layer_2 = rlt::nn::layers::dense::BindConfiguration<LAYER_2_CONFIG>;
    // using LAYER_3_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4,
    //                                                              rlt::nn::activation_functions::IDENTITY>;
    // using Layer_3 = rlt::nn::layers::dense::BindConfiguration<LAYER_3_CONFIG>;

    // using CAPABILITY = rlt::nn::capability::Backward<>;
    
    // using namespace rlt::nn_models::sequential;
    // using MUDLE_CHAIN = Module<Layer_1, Module<Layer_2, Module<Layer_3>>>;

    // constexpr TI SEQUENCE_LENGTH = 1;
    // constexpr TI BATCH_SIZE = 1;
    // constexpr TI INPUT_DIM = 5;
    // using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, INPUT_DIM>;

    // using SEQUENTIAL = Build<CAPABILITY, MUDLE_CHAIN, INPUT_SHAPE>;

    class HVACControler
    {
    public:
        HVACControler();
        // todo: change the signature of the request function to fit your needs,
        // e.g., you might want to pass in more information about the environment status
        float request(float env_status); // request control action based on status
    private:
        // internal state and methods for the controler
        DEVICE device;
        // RNG rng;
        // TI seed = 0;
        // SEQUENTIAL sequential;
        // SEQUENTIAL::Buffer<> sequential_buffer;
    };
} // namespace hvac

#endif