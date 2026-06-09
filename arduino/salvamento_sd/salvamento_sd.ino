#include <Wire.h>
#include <SPI.h>
#include <SD.h>

#define MPU_ADDR 0x68

#define SD_CS 17

TwoWire imu1(8, 9);
TwoWire imu2(10, 11);

File logFile;

void initMPU(TwoWire &i2c) {
  i2c.begin();

  i2c.beginTransmission(MPU_ADDR);
  i2c.write(0x6B);
  i2c.write(0x00);
  i2c.endTransmission(true);

  delay(100);
}

bool readMPU(TwoWire &i2c,
             int16_t &ax, int16_t &ay, int16_t &az,
             int16_t &gx, int16_t &gy, int16_t &gz) {

  i2c.beginTransmission(MPU_ADDR);
  i2c.write(0x3B);

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
  i2c.read();

  gx = (i2c.read() << 8) | i2c.read();
  gy = (i2c.read() << 8) | i2c.read();
  gz = (i2c.read() << 8) | i2c.read();

  return true;
}

String getNextFileName() {
  if (!SD.exists("dados.csv")) {
    return "dados.csv";
  }

  int index = 1;
  String fileName;

  while (true) {
    fileName = "dados_" + String(index) + ".csv";

    if (!SD.exists(fileName)) {
      return fileName;
    }

    index++;
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("Iniciando...");

  initMPU(imu1);
  initMPU(imu2);

  Serial.println("IMUs OK");

  if (!SD.begin(SD_CS)) {
    Serial.println("ERRO: SD falhou");
    while (true);
  }

  Serial.println("SD OK");

  String nomeArquivo = getNextFileName();
  Serial.print("Criando arquivo: ");
  Serial.println(nomeArquivo);

  logFile = SD.open(nomeArquivo, FILE_WRITE);

  if (!logFile) {
    Serial.println("Erro ao abrir dados.csv");
    while (true);
  }

  if (logFile.size() == 0) {
    logFile.println("tempo_us,ax1,ay1,az1,gx1,gy1,gz1,ax2,ay2,az2,gx2,gy2,gz2");
    logFile.flush();
  }

  Serial.println("Arquivo pronto.");
}

void loop() {
  int16_t ax1, ay1, az1, gx1, gy1, gz1;
  int16_t ax2, ay2, az2, gx2, gy2, gz2;

  unsigned long tempo = micros();

  bool ok1 = readMPU(imu1, ax1, ay1, az1, gx1, gy1, gz1);
  bool ok2 = readMPU(imu2, ax2, ay2, az2, gx2, gy2, gz2);

  if (ok1 && ok2) {
    logFile.print(tempo); logFile.print(",");

    logFile.print(ax1); logFile.print(",");
    logFile.print(ay1); logFile.print(",");
    logFile.print(az1); logFile.print(",");
    logFile.print(gx1); logFile.print(",");
    logFile.print(gy1); logFile.print(",");
    logFile.print(gz1); logFile.print(",");

    logFile.print(ax2); logFile.print(",");
    logFile.print(ay2); logFile.print(",");
    logFile.print(az2); logFile.print(",");
    logFile.print(gx2); logFile.print(",");
    logFile.print(gy2); logFile.print(",");
    logFile.println(gz2);

    static int contador = 0;
    contador++;

    if (contador >= 20) {
      logFile.flush();
      contador = 0;
    }

    Serial.println("Linha gravada.");
  } else {
    Serial.println("Erro ao ler uma das IMUs.");
  }

  delay(10);
}