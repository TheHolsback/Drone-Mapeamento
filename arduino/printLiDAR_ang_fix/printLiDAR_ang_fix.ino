#include <Arduino.h>
#include <SPI.h>
#include <SD.h>

#define LIDAR_RX 5
#define LIDAR_TX_UNUSED 4
#define LIDAR_PWM 6

#define SD_CS 17

#define LD06_BAUD 230400
#define PACKET_SIZE 47
#define POINTS_PER_PACKET 12

UART lidarSerial(LIDAR_TX_UNUSED, LIDAR_RX, NC, NC);

uint8_t packet[PACKET_SIZE];
File logFile;

uint16_t readUint16LE(int index) {
  return packet[index] | (packet[index + 1] << 8);
}

bool readLD06Packet() {
  while (lidarSerial.available()) {
    uint8_t b = lidarSerial.read();

    if (b == 0x54) {
      packet[0] = b;

      unsigned long startTime = millis();

      while (lidarSerial.available() < PACKET_SIZE - 1) {
        if (millis() - startTime > 100) {
          return false;
        }
      }

      for (int i = 1; i < PACKET_SIZE; i++) {
        packet[i] = lidarSerial.read();
      }

      if (packet[1] == 0x2C) {
        return true;
      }
    }
  }

  return false;
}

void processAndSaveLD06() {
  uint16_t startAngleRaw = readUint16LE(4);
  uint16_t endAngleRaw   = readUint16LE(42);

  float startAngle = startAngleRaw / 100.0;
  float endAngle   = endAngleRaw / 100.0;

  float angleStep;

  if (endAngle >= startAngle) {
    angleStep = (endAngle - startAngle) / (POINTS_PER_PACKET - 1);
  } else {
    angleStep = ((endAngle + 360.0) - startAngle) / (POINTS_PER_PACKET - 1);
  }

  float angulosAlvo[] = {0, 90, 180, 270};
  int qtdAngulos = 4;

  float tolerancia = 2.0;

  for (int i = 0; i < POINTS_PER_PACKET; i++) {
    int base = 6 + i * 3;

    uint16_t distance = readUint16LE(base);
    uint8_t intensity = packet[base + 2];

    float angle = startAngle + angleStep * i;

    if (angle >= 360.0) {
      angle -= 360.0;
    }

    for (int j = 0; j < qtdAngulos; j++) {
      float alvo = angulosAlvo[j];

      float diff = abs(angle - alvo);

      if (diff > 180.0) {
        diff = 360.0 - diff;
      }

      if (diff <= tolerancia) {
        logFile.print(alvo, 2);
        logFile.print(",");

        logFile.print(angle, 2);
        logFile.print(",");

        logFile.print(distance);
        logFile.print(",");

        logFile.println(intensity);

        Serial.print("Salvo -> Alvo: ");
        Serial.print(alvo, 2);
        Serial.print(" | Angle: ");
        Serial.print(angle, 2);
        Serial.print(" | Distance: ");
        Serial.print(distance);
        Serial.print(" | Intensity: ");
        Serial.println(intensity);
      }
    }
  }

  static int contador = 0;
  contador++;

  if (contador >= 20) {
    logFile.flush();
    contador = 0;
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("Inicializando LD06 + SD...");

  pinMode(LIDAR_PWM, OUTPUT);
  analogWrite(LIDAR_PWM, 180);

  lidarSerial.begin(LD06_BAUD);

  if (!SD.begin(SD_CS)) {
    Serial.println("ERRO: SD falhou");
    while (true);
  }

  Serial.println("SD OK");

  logFile = SD.open("lidar.csv", FILE_WRITE);

  if (!logFile) {
    Serial.println("ERRO: não foi possível abrir lidar.csv");
    while (true);
  }

  if (logFile.size() == 0) {
    logFile.println("alvo,angle,distance,intensity");
    logFile.flush();
  }

  Serial.println("Arquivo lidar.csv pronto.");
}

void loop() {
  if (readLD06Packet()) {
    processAndSaveLD06();
  }
}