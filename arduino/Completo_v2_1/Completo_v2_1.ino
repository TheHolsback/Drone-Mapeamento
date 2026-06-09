/*
 * Completo_v2.ino
 *
 * Sensores:
 *   • IMU1  – MPU6500 (I²C bus GP8/GP9  → i2c0)
 *   • IMU2  – MPU6500 (I²C bus GP10/GP11 → i2c1)
 *   • ToF   – VL53L1X (mesmo bus da IMU1) → modo Short, budget 20 ms → 50 Hz
 *   • Flow  – PMW3901 (SPI1: MISO=GP12, CS=GP13, SCK=GP14, MOSI=GP15) → ~100–150 Hz
 *   • LiDAR – LD06  (uart1, RX=GP5) → ~490 pacotes/s, ~5880 pontos/s
 *   • SD    – cartão (SPI0, CS=GP17)
 *
 * CSV: tempo_us, ax1..gz1, ax2..gz2, angle, distance, intensity, tof_mm, flow_dx, flow_dy
 *
 * Arquitetura de leitura:
 *   - VL53L1X: não bloqueante via dataReady(); atualiza cache a cada nova medição (50 Hz)
 *   - PMW3901: polled a cada iteração do loop; retorna delta acumulado desde última leitura
 *   - LD06: driver principal da gravação; cada pacote dispara uma linha no CSV
 *
 * Notas de compatibilidade (core RP2040 5.x):
 *   - TwoWire agora exige i2c_inst_t* como 1º argumento
 *   - UART foi renomeado para SerialUART; construtor sem args CTS/RTS
 *   - logFile.size() trava (deadlock FAT em modo O_APPEND) → removido
 */

#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <VL53L1X.h>
#include "Padrao_PMW3901.h"

// ── Endereço I²C comum dos MPU6500 ───────────────────────────────────────────
#define MPU_ADDR 0x68

// ── Buses I²C ────────────────────────────────────────────────────────────────
// Core 5.x: construtor exige periférico como 1º arg.
//   GP8 (SDA) + GP9 (SCL)   → i2c0
//   GP10(SDA) + GP11(SCL)   → i2c1
TwoWire imu1(i2c0, 8, 9);    // GP8 SDA, GP9 SCL  — IMU1 + VL53L1X
TwoWire imu2(i2c1, 10, 11);  // GP10 SDA, GP11 SCL — IMU2

// ── VL53L1X (ToF) ────────────────────────────────────────────────────────────
VL53L1X tof;

// ── PMW3901 (fluxo óptico) ───────────────────────────────────────────────────
// SPI1: MISO=GP12, CS=GP13, SCK=GP14, MOSI=GP15
SPIClassRP2040 SPI_FLOW(spi1, 12, 13, 14, 15);
Padrao_PMW3901 flow(&SPI_FLOW, 13);

// ── SD ───────────────────────────────────────────────────────────────────────
#define SD_CS 17

// ── LD06 ─────────────────────────────────────────────────────────────────────
#define LIDAR_RX          5
#define LIDAR_TX_UNUSED   4
#define LIDAR_PWM         6
#define LD06_BAUD         230400
#define INTENSITY_FILTER  223
#define PACKET_SIZE       47
#define POINTS_PER_PACKET 12

// Core 5.x: UART renomeado para SerialUART; GP4=TX uart1, GP5=RX uart1
SerialUART lidarSerial(uart1, LIDAR_TX_UNUSED, LIDAR_RX);

// ── Estado global ─────────────────────────────────────────────────────────────
File    logFile;
uint8_t packet[PACKET_SIZE];

unsigned long inicioGravacao          = 0;
bool          calibFinalizadaIndicada = false;

struct IMUData {
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
};
IMUData data1, data2;

// Valores cacheados — atualizados de forma assíncrona no loop
uint16_t cachedTof    = 0;
int16_t  cachedFlowDx = 0;
int16_t  cachedFlowDy = 0;

// ── Utilitários ───────────────────────────────────────────────────────────────

uint16_t readUint16LE(int index) {
  return packet[index] | (packet[index + 1] << 8);
}

// ── MPU6500 ──────────────────────────────────────────────────────────────────

void writeMPU(TwoWire &i2c, uint8_t reg, uint8_t value) {
  i2c.beginTransmission(MPU_ADDR);
  i2c.write(reg);
  i2c.write(value);
  i2c.endTransmission(true);
}

void initMPU(TwoWire &i2c) {
  i2c.begin();
  writeMPU(i2c, 0x6B, 0x00); // PWR_MGMT_1: acordar
  delay(100);

  //             |   ACCELEROMETER    |           GYROSCOPE
  //  DLPF_CFG | Bandwidth | Delay  | Bandwidth | Delay  | Sample Rate
  //  ---------+-----------+--------+-----------+--------+-------------
  //  0        | 260Hz     | 0ms    | 256Hz     | 0.98ms | 8kHz
  //  1        | 184Hz     | 2.0ms  | 188Hz     | 1.9ms  | 1kHz
  //  2        | 94Hz      | 3.0ms  | 98Hz      | 2.8ms  | 1kHz  ← atual
  //  3        | 44Hz      | 4.9ms  | 42Hz      | 4.8ms  | 1kHz

  writeMPU(i2c, 0x1A, 0x02); // CONFIG: DLPF_CFG=2 → giroscópio 98 Hz BW
  writeMPU(i2c, 0x1D, 0x02); // ACCEL_CONFIG2: A_DLPF_CFG=2 → acelerômetro 94 Hz BW
}

bool readMPU(TwoWire &i2c, IMUData &data) {
  i2c.beginTransmission(MPU_ADDR);
  i2c.write(0x3B);
  if (i2c.endTransmission(false) != 0) return false;
  if (i2c.requestFrom(MPU_ADDR, 14) < 14) return false;

  data.ax = (i2c.read() << 8) | i2c.read();
  data.ay = (i2c.read() << 8) | i2c.read();
  data.az = (i2c.read() << 8) | i2c.read();
  i2c.read(); i2c.read(); // temperatura (descartada)
  data.gx = (i2c.read() << 8) | i2c.read();
  data.gy = (i2c.read() << 8) | i2c.read();
  data.gz = (i2c.read() << 8) | i2c.read();
  return true;
}

// ── LD06 ─────────────────────────────────────────────────────────────────────

bool readLD06Packet() {
  // Non-blocking stateful parser to find and assemble LD06 packets.
  // Resilient to bytes lost and does not block the main loop.
  static uint8_t buf[PACKET_SIZE];
  static int idx = 0;
  static bool synced = false;

  while (lidarSerial.available()) {
    uint8_t b = lidarSerial.read();

    if (!synced) {
      // Busca sequência de início: 0x54 seguido por 0x2C
      if (idx == 0 && b == 0x54) {
        buf[0] = b;
        idx = 1;
      } else if (idx == 1 && b == 0x2C) {
        buf[1] = b;
        idx = 2;
        synced = true;
      } else {
        // reinicia busca se byte inesperado
        idx = 0;
      }
      continue;
    }

    buf[idx++] = b;

    if (idx == PACKET_SIZE) {
      // pacote completo — copie para buffer público e re-sincronize
      idx = 0;
      synced = false; // forçar redetecção no próximo pacote
      memcpy(packet, buf, PACKET_SIZE);
      return true;
    }
  }
  return false;
}

void saveLD06PacketWithIMU() {
  uint32_t tempo = micros();

  bool ok1 = readMPU(imu1, data1);
  bool ok2 = readMPU(imu2, data2);
  if (!ok1 || !ok2) {
    Serial.println("Aviso: falha na leitura da IMU; pacote LD06 ignorado.");  
    // return
  };

  uint16_t startAngle = readUint16LE(4);
  uint16_t endAngle   = readUint16LE(42);

  int32_t angleDiff = (endAngle >= startAngle)
    ? (int32_t)(endAngle - startAngle)
    : (int32_t)(36000 + endAngle) - startAngle;

  static uint32_t contadorFlush = 0;
  char line[256];

  for (int i = 0; i < POINTS_PER_PACKET; i++) {
    int      base      = 6 + i * 3;
    uint16_t distance  = readUint16LE(base);
    uint8_t  intensity = packet[base + 2];

    uint16_t angle = startAngle + (uint16_t)((angleDiff * i) / (POINTS_PER_PACKET - 1));
    if (angle >= 36000) angle -= 36000;

    int n = snprintf(
      line, sizeof(line),
      "%lu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%u,%u,%u,%u,%d,%d\n",
      tempo,
      data1.ax, data1.ay, data1.az, data1.gx, data1.gy, data1.gz,
      data2.ax, data2.ay, data2.az, data2.gx, data2.gy, data2.gz,
      angle, distance, intensity,
      cachedTof,     // VL53L1X — última leitura válida (50 Hz)
      cachedFlowDx,  // PMW3901 — delta X desde última leitura
      cachedFlowDy   // PMW3901 — delta Y desde última leitura
    );

    logFile.write((uint8_t *)line, n);
    contadorFlush++;
  }

  // Flush a cada ~240 pontos (~0.5 s a 490 pacotes/s)
  if (contadorFlush >= 240) {
    logFile.flush();
    contadorFlush = 0;
  }
}

// ── SD helpers ───────────────────────────────────────────────────────────────

String getNextFileName() {
  if (!SD.exists("dados.csv")) return "dados.csv";
  for (int i = 1; ; i++) {
    String name = "dados_" + String(i) + ".csv";
    if (!SD.exists(name)) return name;
  }
}

void piscarLed(int n, float dur) {
  for (int i = 0; i < n; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(dur);
    digitalWrite(LED_BUILTIN, LOW);
    delay(dur);
  }
}

// ── Setup ────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  piscarLed(3, 750);

  Serial.println("Iniciando sistema...");

  // ── IMUs ─────────────────────────────────────────────────────────────────
  initMPU(imu1);
  initMPU(imu2);
  Serial.println("IMUs OK");

  // ── VL53L1X ──────────────────────────────────────────────────────────────
  // Compartilha o bus I²C com a IMU1 (endereço 0x29, sem conflito com 0x68)
  tof.setBus(&imu1);
  if (!tof.init()) {
    Serial.println("ERRO: VL53L1X nao encontrado!");
    while (true);
  }
  // Modo Short: range até ~1.3 m, budget mínimo de 20 ms → 50 Hz (máximo)
  // Para range maior use Long (33 ms mínimo → ~30 Hz):
  //   tof.setDistanceMode(VL53L1X::Long);
  //   tof.setMeasurementTimingBudget(33000);
  //   tof.startContinuous(33);
  tof.setDistanceMode(VL53L1X::Short);
  tof.setMeasurementTimingBudget(20000); // 20 ms → 50 Hz
  tof.startContinuous(20);
  Serial.println("VL53L1X OK  [modo Short | 20 ms budget | 50 Hz]");

  // ── PMW3901 ──────────────────────────────────────────────────────────────
  if (!flow.begin()) {
    Serial.println("ERRO: PMW3901 nao encontrado!");
    while (true);
  }
  Serial.println("PMW3901 OK  [~100-150 Hz, polled por loop]");

  // ── LD06 ─────────────────────────────────────────────────────────────────
  pinMode(LIDAR_PWM, OUTPUT);
  analogWrite(LIDAR_PWM, 180);
  lidarSerial.begin(LD06_BAUD);
  Serial.println("LD06 OK     [230400 baud | ~490 pacotes/s]");

  // ── SD ───────────────────────────────────────────────────────────────────
  if (!SD.begin(SD_CS)) {
    Serial.println("ERRO: SD falhou");
    while (true);
  }
  Serial.println("SD OK");

  String nome = getNextFileName();
  Serial.print("Criando arquivo: ");
  Serial.println(nome);

  logFile = SD.open(nome, FILE_WRITE);
  if (!logFile) {
    Serial.println("ERRO: nao foi possivel abrir arquivo");
    while (true);
  }

  // Escreve cabeçalho diretamente — sem verificar logFile.size().
  // logFile.size() trava no core RP2040 5.x: ao abrir em O_APPEND o FAT
  // entra em deadlock ao tentar sincronizar metadados ainda não commitados.
  // A verificação é desnecessária porque getNextFileName() garante que o
  // arquivo é sempre novo (não existia antes de ser aberto).
  logFile.println("tempo_us,ax1,ay1,az1,gx1,gy1,gz1,"
                  "ax2,ay2,az2,gx2,gy2,gz2,"
                  "angle,distance,intensity,"
                  "tof_mm,flow_dx,flow_dy"); 
  logFile.flush();

  Serial.println("Gravando dados...");
  piscarLed(5, 250);

  inicioGravacao          = millis();
  calibFinalizadaIndicada = false;
}

// ── Loop ─────────────────────────────────────────────────────────────────────

void loop() {

  // Indicação visual fim da calibração (5 s)
  if (!calibFinalizadaIndicada && millis() - inicioGravacao >= 5000) {
    logFile.println("1,1,1,1,1,1,1,"
                  "1,1,1,1,1,1,"
                  "1,1,1,"
                  "1,1,1"); 
    logFile.flush();
    piscarLed(2, 250);
    calibFinalizadaIndicada = true;
    Serial.println("Calibracao finalizada.");
  }

  // ── VL53L1X: leitura não bloqueante ──────────────────────────────────────
  // dataReady() consulta apenas um registrador I²C (~100 µs);
  // read(false) lê o resultado sem esperar — seguro pois já verificamos ready.
  if (tof.dataReady()) {
    cachedTof = tof.read(false);
  }

  // ── PMW3901: leitura não bloqueante ──────────────────────────────────────
  // readMotionCount faz uma transação SPI rápida e retorna o delta acumulado
  // desde a última chamada. Polling frequente = melhor resolução temporal.
  flow.readMotionCount(&cachedFlowDx, &cachedFlowDy);

  // ── LD06: driver principal da gravação ───────────────────────────────────
  if (readLD06Packet()) {
    saveLD06PacketWithIMU();
  }
}