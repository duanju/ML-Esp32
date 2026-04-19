#include <stdio.h>
// #include <stdbool.h>
// #include "esp_timer.h"
#include "hvac_controler.h"


#define RL_TOOLS_OUTPUT_DIM 4
#define RL_TOOLS_BATCH_SIZE 1
float output_mem[RL_TOOLS_BATCH_SIZE * RL_TOOLS_OUTPUT_DIM];
extern "C" void app_main(void)
{
    // for(int batch_i = 0; batch_i < RL_TOOLS_BATCH_SIZE; batch_i++){
    //     for(int col_i = 0; col_i < RL_TOOLS_OUTPUT_DIM; col_i++){
    //         printf("%+5.5f ", output_mem[batch_i * RL_TOOLS_OUTPUT_DIM + col_i]);
    //     }
    //     printf("\n");
    // }

    hvac::HVACControler controler;
    controler.request(0.5f);
  
    printf("OUTPUT_DIM: %d\n", RL_TOOLS_OUTPUT_DIM);
}