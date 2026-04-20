#include <stdio.h>
#include "hvac_controler.h"

extern "C" void app_main(void)
{
    hvac::HVACControler controler;
    float output = controler.request(0.1f);
    printf("Initial OUTPUT: %f\n", output);

    printf("Train model...\n");
    float loss = controler.update();
    printf("Final Loss: %f\n", loss);

    output = controler.request(0.1f);
    printf("OUTPUT: %f\n", output);
}