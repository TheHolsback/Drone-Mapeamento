#include <Arduino.h>

#define LIDAR_RX 5
#define LIDAR_TX 4
#define LIDAR_PWM 6

#define LD06_BAUD 230400
#define PACKET_SIZE 47
#define POINTS_PER_PACKET 12

uint8_t packet[PACKET_SIZE];

uint16_t readUint16LE(int index) {
  return packet[index] | (packet[index + 1] << 8);
}

bool readLD06Packet() {

  while (Serial1.available()) {

    uint8_t b = Serial1.read();

    if (b == 0x54) {

      packet[0] = b;

      unsigned long startTime = millis();

      while (Serial1.available() < (PACKET_SIZE - 1)) {

        if (millis() - startTime > 100) {
          return false;
        }
      }

      for (int i = 1; i < PACKET_SIZE; i++) {
        packet[i] = Serial1.read();
      }

      if (packet[1] == 0x2C) {
        return true;
      }
    }
  }

  return false;
}

void printLD06Packet() {

  uint16_t speed = readUint16LE(2);

  uint16_t startAngleRaw = readUint16LE(4);
  uint16_t endAngleRaw   = readUint16LE(42);

  uint16_t timestamp = readUint16LE(44);

  float startAngle = startAngleRaw / 100.0f;
  float endAngle   = endAngleRaw / 100.0f;

  float angleStep;

  if (endAngle >= startAngle) {
    angleStep = (endAngle - startAngle) / (POINTS_PER_PACKET - 1);
  } else {
    angleStep = ((endAngle + 360.0f) - startAngle) /
                (POINTS_PER_PACKET - 1);
  }

  Serial.println();
  Serial.println("========== LD06 ==========");

  Serial.print("Velocidade: ");
  Serial.println(speed);

  Serial.print("Angulo inicial: ");
  Serial.println(startAngle, 2);

  Serial.print("Angulo final: ");
  Serial.println(endAngle, 2);

  Serial.print("Timestamp: ");
  Serial.println(timestamp);

  for (int i = 0; i < POINTS_PER_PACKET; i++) {

    int base = 6 + i * 3;

    uint16_t distance = readUint16LE(base);
    uint8_t intensity = packet[base + 2];

    float angle = startAngle + angleStep * i;

    if (angle >= 360.0f) {
      angle -= 360.0f;
    }

    Serial.print("Ponto ");
    Serial.print(i);

    Serial.print(" | Angulo: ");
    Serial.print(angle, 2);

    Serial.print("° | Distancia: ");
    Serial.print(distance);

    Serial.print(" mm | Intensidade: ");
    Serial.println(intensity);
  }

  Serial.println("==========================");
}

void setup() {

  Serial.begin(115200);

  delay(2000);

  pinMode(LIDAR_PWM, OUTPUT);

  // Liga o motor do LD06
  analogWrite(LIDAR_PWM, 180);

  // Configuração UART
  Serial1.setRX(LIDAR_RX);
  Serial1.setTX(LIDAR_TX);

  Serial1.begin(LD06_BAUD);

  Serial.println();
  Serial.println("Iniciando LD06...");
  Serial.println("RX = GP5");
  Serial.println("TX = GP4");
  Serial.println("Baud = 230400");
}

void loop() {
  while (Serial1.available()) {
    Serial.print((uint8_t)Serial1.read(), HEX);
    Serial.print(" ");
  }
  // if (readLD06Packet()) {
  //   printLD06Packet();
  // }
}