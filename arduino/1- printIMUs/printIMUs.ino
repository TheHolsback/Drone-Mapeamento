#include <Wire.h>

#define MPU_ADDR 0x68

// Não usar nomes I2C_0 ou I2C_1, pois são reservados internamente
TwoWire imu1(8, 9);     // SDA GP8, SCL GP9
TwoWire imu2(10, 11);   // SDA GP10, SCL GP11

void initMPU(TwoWire &i2c) {
  i2c.begin();

  i2c.beginTransmission(MPU_ADDR);
  i2c.write(0x6B);      // PWR_MGMT_1
  i2c.write(0x00);      // acorda o MPU
  i2c.endTransmission(true);

  delay(100);
}

bool readMPU(TwoWire &i2c,
             int16_t &ax, int16_t &ay, int16_t &az,
             int16_t &gx, int16_t &gy, int16_t &gz) {

  i2c.beginTransmission(MPU_ADDR);
  i2c.write(0x3B); // ACCEL_XOUT_H

  if (i2c.endTransmission(false) != 0) {
    return false;
  }

  int bytes = i2c.requestFrom(MPU_ADDR, 14);

  if (bytes < 14) {
    return false;
  }

  ax = (i2c.read() << 8) | i2c.read();
  ay = (i2c.read() << 8) | i2c.read();
  az = (i2c.read() << 8) | i2c.read();

  i2c.read();
  i2c.read(); // ignora temperatura

  gx = (i2c.read() << 8) | i2c.read();
  gy = (i2c.read() << 8) | i2c.read();
  gz = (i2c.read() << 8) | i2c.read();

  return true;
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("Iniciando MPU-6500...");

  initMPU(imu1);
  initMPU(imu2);

  Serial.println("Inicialização concluída.");
}

void loop() {
  int16_t ax1, ay1, az1, gx1, gy1, gz1;
  int16_t ax2, ay2, az2, gx2, gy2, gz2;

  bool ok1 = readMPU(imu1, ax1, ay1, az1, gx1, gy1, gz1);
  bool ok2 = readMPU(imu2, ax2, ay2, az2, gx2, gy2, gz2);

  Serial.println("========== LEITURA ==========");

  if (ok1) {
    Serial.println("IMU 1:");
    Serial.print("ACC X: "); Serial.print(ax1);
    Serial.print(" | Y: "); Serial.print(ay1);
    Serial.print(" | Z: "); Serial.println(az1);

    Serial.print("GYRO X: "); Serial.print(gx1);
    Serial.print(" | Y: "); Serial.print(gy1);
    Serial.print(" | Z: "); Serial.println(gz1);
  } else {
    Serial.println("ERRO ao ler IMU 1");
  }

  if (ok2) {
    Serial.println("IMU 2:");
    Serial.print("ACC X: "); Serial.print(ax2);
    Serial.print(" | Y: "); Serial.print(ay2);
    Serial.print(" | Z: "); Serial.println(az2);

    Serial.print("GYRO X: "); Serial.print(gx2);
    Serial.print(" | Y: "); Serial.print(gy2);
    Serial.print(" | Z: "); Serial.println(gz2);
  } else {
    Serial.println("ERRO ao ler IMU 2");
  }

  Serial.println("=============================");
  delay(200);
}