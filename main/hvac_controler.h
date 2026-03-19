#ifndef HVAC_CONTROLER_H
#define HVAC_CONTROLER_H

// First include the device operations (ESP32 in your case)
#include <rl_tools/operations/esp32.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/capability/capability.h>
// Then include other RLtools components
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/devices/esp32.h>
#include <rl_tools/rl_tools.h>
namespace rlt = rl_tools;
// add HVACControler in namespace hvac
namespace hvac
{
    using DEV_SPEC_S3 = rlt::devices::DefaultESP32Specification<rlt::devices::esp32::Hardware::S3>;
    using DEVICE = rlt::devices::ESP32<DEV_SPEC_S3>;
    using T = float;
    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 1;
    constexpr TI INPUT_DIM = 5;
    constexpr TI OUTPUT_DIM = 5;
    constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::RELU;

    using LAYER_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, OUTPUT_DIM, ACTIVATION_FUNCTION>;
    using CAPABILITY = rlt::nn::capability::Forward<>;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, INPUT_DIM>;
    using Layer = rlt::nn::layers::dense::Layer<LAYER_CONFIG, CAPABILITY, INPUT_SHAPE>;

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
        RNG rng;
        TI seed = 0;
        // OPTIMIZER optimizer;
        Layer layer;
    };
} // namespace hvac

#endif