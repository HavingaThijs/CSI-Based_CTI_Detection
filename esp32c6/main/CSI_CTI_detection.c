#include <stdio.h>
#include "esp_wifi.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "tvmgen_default.h"
#include "nvs_flash.h"
#include "cmd_decl.h"
#include "input_data.h"
#include "output_data.h"

#define NUM_SUBCARRIERS 242
#define NUM_CLASSES 14
static const char *TAG = "CSI";

// Callback function for received CSI data
void rx_csi_callback(void *ctx, wifi_csi_info_t *data)
{ 
  struct tvmgen_default_inputs inputs = {.input = input_data};
  struct tvmgen_default_outputs outputs = {.output = output_data};
  float *input = (float *)inputs.input;
  uint16_t cnt = 0;

  // Fill input data with active subcarriers only
  for (int i = 0; i < data->len; i+=2) {
    if (i != 0 && i != 2 && i != 4 && i != 6 && i != 8 && i != 10 
        && i != 254 && i != 256 && i != 258 && 
        i != 502 && i != 504 && i != 506 && i != 508 && i != 510) {
      input[cnt] = data->buf[i+1]; // Imaginary part is first!
      input[NUM_SUBCARRIERS+cnt] = data->buf[i];
      cnt++;
    }
  }

#if CONFIG_CSI_CTI_DETECT_ENABLE_MODEL
  uint64_t elapsed_time = 0;
  uint64_t time1, time2;
  time1 = esp_timer_get_time();
  int32_t err = tvmgen_default_run(&inputs, &outputs); // Run inference
  time2 = esp_timer_get_time();
  elapsed_time = time2 - time1;

  if (err != 0) {
    printf("Error running model: %ld", err);
    return;
  }

  // Print scores and calculate argmax
  float *res = outputs.output;
  float max_score = -__FLT_MAX__;
  uint8_t max_index = 0;
  printf("Scores: ");
  for (int i = 0; i < NUM_CLASSES; i++) {
    printf("%f ", res[i]);
    if (res[i] > max_score) {
        max_score = res[i];
        max_index = i;
    }
  }

  printf("\nPrediction: %d time: %lld us\n", max_index, elapsed_time);
#endif

  // Print input data
  for (int i = 0; i < (NUM_SUBCARRIERS * 2); i++) {
    printf("%d,", (int)input[i]);
  }
  printf("\n");
  printf("%d %d\n", data->rx_ctrl.rssi, data->rx_ctrl.data_rssi);
}

// Initialize CSI data acquisition
void init_csi(void) 
{
  // Only get HE data
  wifi_csi_config_t csi_config = {
      .enable = true,
      .acquire_csi_legacy = false,
      .acquire_csi_ht20 = false,
      .acquire_csi_mu = true,
      .acquire_csi_su = true,
  };

  ESP_LOGI(TAG, "Setting CSI callback"); 
  ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true)); // Also record for other clients
  ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
  ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(rx_csi_callback, NULL));
  ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK( ret );

    initialise_wifi();
    register_system();
    register_wifi();
    init_csi();
}
