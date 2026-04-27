#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>

#define MPU_ADDR 0x68

// IMUs
TwoWire imu1(8, 9);
TwoWire imu2(10, 11);

// SD
#define SD_CS 17

// LD06
#define LIDAR_RX 5
#define LIDAR_TX_UNUSED 4
#define LIDAR_PWM 6
#define LD06_BAUD 230400
#define INTENSITY_FILTER 223

#define PACKET_SIZE 47
#define POINTS_PER_PACKET 12

UART lidarSerial(LIDAR_TX_UNUSED, LIDAR_RX, NC, NC);

File logFile;
uint8_t packet[PACKET_SIZE];

struct IMUData {
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
};

IMUData data1, data2;

uint16_t readUint16LE(int index) {
  return packet[index] | (packet[index + 1] << 8);
}

void initMPU(TwoWire &i2c) {
  i2c.begin();

  i2c.beginTransmission(MPU_ADDR);
  i2c.write(0x6B);
  i2c.write(0x00);
  i2c.endTransmission(true);

  delay(100);
}

bool readMPU(TwoWire &i2c, IMUData &data) {
  i2c.beginTransmission(MPU_ADDR);
  i2c.write(0x3B);

  if (i2c.endTransmission(false) != 0) {
    return false;
  }

  int bytes = i2c.requestFrom(MPU_ADDR, 14);

  if (bytes < 14) {
    return false;
  }

  data.ax = (i2c.read() << 8) | i2c.read();
  data.ay = (i2c.read() << 8) | i2c.read();
  data.az = (i2c.read() << 8) | i2c.read();

  i2c.read();
  i2c.read();

  data.gx = (i2c.read() << 8) | i2c.read();
  data.gy = (i2c.read() << 8) | i2c.read();
  data.gz = (i2c.read() << 8) | i2c.read();

  return true;
}

bool readLD06Packet() {
  while (lidarSerial.available()) {
    uint8_t b = lidarSerial.read();

    if (b == 0x54) {
      packet[0] = b;

      unsigned long startTime = millis();

      while (lidarSerial.available() < PACKET_SIZE - 1) {
        if (millis() - startTime > 50) {
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

void saveLD06PacketWithIMU() {
  uint32_t tempo = micros();

  bool ok1 = readMPU(imu1, data1);
  bool ok2 = readMPU(imu2, data2);

  if (!ok1 || !ok2) {
    return;
  }

  uint16_t startAngle = readUint16LE(4);
  uint16_t endAngle = readUint16LE(42);

  int32_t angleDiff;

  if (endAngle >= startAngle) {
    angleDiff = endAngle - startAngle;
  } else {
    angleDiff = (36000 + endAngle) - startAngle;
  }

  static uint32_t contadorFlush = 0;
  char line[180];

  for (int i = 0; i < POINTS_PER_PACKET; i++) {
    int base = 6 + i * 3;

    uint16_t distance = readUint16LE(base);
    uint8_t intensity = packet[base + 2];
    if (intensity < INTENSITY_FILTER) {
      return;
    }

    uint16_t angle = startAngle + ((angleDiff * i) / (POINTS_PER_PACKET - 1));

    if (angle >= 36000) {
      angle -= 36000;
    }

    int n = snprintf(
      line,
      sizeof(line),
      "%lu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%u,%u,%u\n",
      tempo,
      data1.ax, data1.ay, data1.az, data1.gx, data1.gy, data1.gz,
      data2.ax, data2.ay, data2.az, data2.gx, data2.gy, data2.gz,
      angle,
      distance,
      intensity
    );

    logFile.write((uint8_t *)line, n);
    contadorFlush++;
  }

  if (contadorFlush >= 240) {
    logFile.flush();
    contadorFlush = 0;
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("Iniciando sistema...");

  initMPU(imu1);
  initMPU(imu2);

  Serial.println("IMUs OK");

  pinMode(LIDAR_PWM, OUTPUT);
  analogWrite(LIDAR_PWM, 180);

  lidarSerial.begin(LD06_BAUD);

  Serial.println("LD06 OK");

  if (!SD.begin(SD_CS)) {
    Serial.println("ERRO: SD falhou");
    while (true);
  }

  Serial.println("SD OK");

  logFile = SD.open("dados.csv", FILE_WRITE);

  if (!logFile) {
    Serial.println("ERRO: nao foi possivel abrir dados.csv");
    while (true);
  }

  if (logFile.size() == 0) {
    logFile.println("tempo_us,ax1,ay1,az1,gx1,gy1,gz1,ax2,ay2,az2,gx2,gy2,gz2,angle,distance,intensity");
    logFile.flush();
  }

  Serial.println("Gravando dados...");
}

void loop() {
  if (readLD06Packet()) {
    saveLD06PacketWithIMU();
  }
}