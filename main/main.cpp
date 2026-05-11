#include <stdio.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <driver/gpio.h>
#include <esp_system.h>
#include <esp_random.h>
#include "esp_timer.h"
#include <inttypes.h>
#include "hvac_controler.h"
#include "dataset_loader.h"

static constexpr gpio_num_t BUTTON1_GPIO = GPIO_NUM_10; // replace this with your actual button 1 pin
static constexpr gpio_num_t BUTTON2_GPIO = GPIO_NUM_0; // Boot button on many ESP32 boards, replace if needed
static constexpr TickType_t BUTTON_DEBOUNCE_MS = 50;

static bool is_button_pressed(gpio_num_t gpio)
{
    return gpio_get_level(gpio) == 0; // assume buttons pull pin to ground when pressed
}

static int wait_for_button_press()
{
    bool last1 = false;
    bool last2 = false;

    while (true)
    {
        bool current1 = is_button_pressed(BUTTON1_GPIO);
        bool current2 = is_button_pressed(BUTTON2_GPIO);

        if (current1 && !last1)
        {
            vTaskDelay(pdMS_TO_TICKS(BUTTON_DEBOUNCE_MS));
            if (is_button_pressed(BUTTON1_GPIO))
            {
                return 1;
            }
        }

        if (current2 && !last2)
        {
            vTaskDelay(pdMS_TO_TICKS(BUTTON_DEBOUNCE_MS));
            if (is_button_pressed(BUTTON2_GPIO))
            {
                return 2;
            }
        }

        last1 = current1;
        last2 = current2;
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

extern "C" void app_main(void)
{
    gpio_config_t io_conf = {};
    io_conf.intr_type = GPIO_INTR_DISABLE;
    io_conf.mode = GPIO_MODE_INPUT;
    io_conf.pin_bit_mask = (1ULL << BUTTON1_GPIO) | (1ULL << BUTTON2_GPIO);
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_ENABLE;
    gpio_config(&io_conf);

    printf("Button1 GPIO=%d -> train model\n", BUTTON1_GPIO);
    printf("Button2 GPIO=%d -> random inference\n", BUTTON2_GPIO);
    printf("Press a button to begin.\n");

    hvac::HVACControler controler;


    while (true)
    {
        int button = wait_for_button_press();

        if (button == 1)
        {
            printf("Button1 pressed: starting training...\n");
            float loss = controler.update();
            printf("Training completed. Final Loss: %f\n", loss);
        }
        else if (button == 2)
        {
            uint32_t idx = esp_random() % hvac::DATASET_SIZE;
            float random_input[4] = {
                dataset::get_input(idx, 0),
                dataset::get_input(idx, 1),
                dataset::get_input(idx, 2),
                dataset::get_input(idx, 3)
            };
            // record the total time taken for inference
            int64_t start_time = esp_timer_get_time();
            float output = controler.request(random_input);
            int64_t end_time = esp_timer_get_time();
            float target = dataset::get_target(idx);
            printf("Button2 pressed: random input idx=%" PRIu32 " input=[%f, %f, %f, %f] inferred output=%f target=%f (Time: %" PRId64 " us)\n",
                   idx, random_input[0], random_input[1], random_input[2], random_input[3], output, target, end_time - start_time);
        }

        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
