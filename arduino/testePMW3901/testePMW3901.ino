#include <SPI.h>
#include "Padrao_PMW3901.h"

SPIClassRP2040 SPI_FLOW(
    spi1,
    12,  // RX (MISO)
    13,  // CS
    14,  // SCK
    15   // TX (MOSI)
);

Padrao_PMW3901 flow(
    &SPI_FLOW,
    13);

int16_t dx;
int16_t dy;

void setup()
{
    Serial.begin(115200);
    delay(2000);

    Serial.println("Inicializando PMW3901...");

    if (!flow.begin())
    {
        Serial.println("ERRO PMW3901");
        while (1);
    }

    Serial.println("PMW3901 OK");
}

void loop()
{
    flow.readMotionCount(
        &dx,
        &dy
    );

    Serial.print("dX=");
    Serial.print(dx);

    Serial.print(" dY=");
    Serial.print(dy);
}