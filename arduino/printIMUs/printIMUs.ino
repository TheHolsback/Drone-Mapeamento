#include <Wire.h>
#include <VL53L1X.h>

#define MPU_ADDR 0x68

TwoWire imu1(8, 9);      // GP8 SDA, GP9 SCL
TwoWire imu2(10, 11);    // GP10 SDA, GP11 SCL

VL53L1X tof;

// ======================================================
// MPU6500
// ======================================================

void initMPU(TwoWire &i2c)
{
  i2c.begin();

  i2c.beginTransmission(MPU_ADDR);
  i2c.write(0x6B);
  i2c.write(0x00);
  i2c.endTransmission(true);

  delay(100);
}

bool readMPU(TwoWire &i2c,
             int16_t &ax, int16_t &ay, int16_t &az,
             int16_t &gx, int16_t &gy, int16_t &gz)
{
  i2c.beginTransmission(MPU_ADDR);
  i2c.write(0x3B);

  if (i2c.endTransmission(false) != 0)
    return false;

  if (i2c.requestFrom(MPU_ADDR, 14) < 14)
    return false;

  ax = (i2c.read() << 8) | i2c.read();
  ay = (i2c.read() << 8) | i2c.read();
  az = (i2c.read() << 8) | i2c.read();

  i2c.read();
  i2c.read(); // temperatura

  gx = (i2c.read() << 8) | i2c.read();
  gy = (i2c.read() << 8) | i2c.read();
  gz = (i2c.read() << 8) | i2c.read();

  return true;
}

// ======================================================
// Setup
// ======================================================

void setup()
{
  Serial.begin(115200);
  delay(2000);

  Serial.println("Inicializando sensores...");

  initMPU(imu1);
  initMPU(imu2);

  // VL53L1X está no mesmo barramento do MPU #1
  if (!tof.init(&imu1))
  {
    Serial.println("ERRO: VL53L1X nao encontrado!");
    while (1);
  }

  tof.setDistanceMode(VL53L1X::Long);
  tof.setMeasurementTimingBudget(50000);
  tof.startContinuous(50);

  Serial.println("Sensores inicializados.");
}

// ======================================================
// Loop
// ======================================================

void loop()
{
  int16_t ax1, ay1, az1, gx1, gy1, gz1;
  int16_t ax2, ay2, az2, gx2, gy2, gz2;

  bool ok1 = readMPU(imu1, ax1, ay1, az1, gx1, gy1, gz1);
  bool ok2 = readMPU(imu2, ax2, ay2, az2, gx2, gy2, gz2);

  uint16_t distancia = tof.read();

  Serial.println("\n========== LEITURA ==========");

  // ---------------- IMU 1 ----------------

  if (ok1)
  {
    Serial.println("IMU 1");

    Serial.print("ACC: ");
    Serial.print(ax1);
    Serial.print(", ");
    Serial.print(ay1);
    Serial.print(", ");
    Serial.println(az1);

    Serial.print("GYRO: ");
    Serial.print(gx1);
    Serial.print(", ");
    Serial.print(gy1);
    Serial.print(", ");
    Serial.println(gz1);
  }
  else
  {
    Serial.println("ERRO IMU 1");
  }

  // ---------------- IMU 2 ----------------

  if (ok2)
  {
    Serial.println("IMU 2");

    Serial.print("ACC: ");
    Serial.print(ax2);
    Serial.print(", ");
    Serial.print(ay2);
    Serial.print(", ");
    Serial.println(az2);

    Serial.print("GYRO: ");
    Serial.print(gx2);
    Serial.print(", ");
    Serial.print(gy2);
    Serial.print(", ");
    Serial.println(gz2);
  }
  else
  {
    Serial.println("ERRO IMU 2");
  }

  // ---------------- VL53L1X ----------------

  if (tof.timeoutOccurred())
  {
    Serial.println("VL53L1X TIMEOUT");
  }
  else
  {
    Serial.print("Distancia VL53L1X: ");
    Serial.print(distancia);
    Serial.println(" mm");
  }

  Serial.println("=============================");

  delay(250);
}