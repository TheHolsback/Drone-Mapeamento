# Lista de Componentes - Sistema de Mapeamento 3D com LiDAR

## 📋 Visão Geral
Sistema de mapeamento tridimensional dedicado à aquisição de dados espaciais de alta precisão para geração de nuvens de pontos em ambientes confinados (cavernas, inspeção, operações em ambientes de difícil acesso).

---

## 🔧 Componentes Eletrônicos

### 1. **Microcontrolador**
- **Componente:** Raspberry Pi Pico (RP2040)
- **Função:** Controlar sensores, processar dados e gerenciar armazenamento
- **Especificações:**
  - Processador: ARM Cortex-M0+ dual-core @ 125 MHz (overclock até 270 MHz)
  - Memória RAM: 264 KB
  - Memória Flash: 2 MB
  - Periféricos: 2x UART, 2x I2C, 2x SPI, 16x PWM, ADC
- **Recursos Utilizados:**
  - 2x UART (para LiDAR e comunicação)
  - 2x I2C (para 2x IMU)
  - 1x SPI (para cartão SD)
  - Pinos GPIO customizáveis

### 2. **Unidade de Medição Inercial (IMU) - 2x**
- **Componente:** MPU-6050
- **Quantidade:** 2 unidades
- **Função:** Medir aceleração (ax, ay, az) e velocidade angular (gx, gy, gz) para estimativa de atitude e posição inercial
- **Especificações:**
  - Endereço I2C: 0x68
  - Acelerómetro: ±2g a ±16g
  - Giroscópio: ±250°/s a ±2000°/s
  - Taxa de amostragem: Configurável
- **Saídas de Dados:**
  - Aceleração em 3 eixos (X, Y, Z)
  - Velocidade angular em 3 eixos (X, Y, Z)

### 3. **Sensor LiDAR**
- **Componente:** LD06 (LiDAR 2D)
- **Função:** Medir distância e ângulo para geração de nuvem de pontos 2D
- **Especificações:**
  - Protocolo de Comunicação: Serial UART
  - Baud Rate: 230400 bps
  - Formato de Pacote: 47 bytes
  - Pontos por Pacote: 12 medições
  - Pino PWM: Para controle de operação (ligação/desligação)
- **Saídas de Dados:**
  - Ângulo de varredura (0° a 360°)
  - Distância medida (em mm)
  - Intensidade do retorno (0-255, com filtro em 223 para qualidade)
- **Filtragem Aplicada:** Intensidade mínima = 223 (baseado em análise de bancada)

### 4. **Armazenamento de Dados**
- **Componente:** Módulo Cartão SD
- **Função:** Registrar dados de sensores em tempo real
- **Especificações:**
  - Interface: SPI
  - Pino CS (Chip Select): 17
  - Capacidade: Recomendado mínimo 16GB

---

## 📊 Dados Coletados

### Estrutura de Dados por Ciclo
O sistema coleta **15 variáveis de dados** por ciclo de leitura:

| Variável | Fonte | Descrição |
|----------|-------|-----------|
| `tempo_us` | Arduino | Timestamp em microsegundos |
| `ax1, ay1, az1` | IMU 1 | Aceleração nos 3 eixos (IMU 1) |
| `gx1, gy1, gz1` | IMU 1 | Velocidade angular nos 3 eixos (IMU 1) |
| `ax2, ay2, az2` | IMU 2 | Aceleração nos 3 eixos (IMU 2) |
| `gx2, gy2, gz2` | IMU 2 | Velocidade angular nos 3 eixos (IMU 2) |
| `angle` | LD06 | Ângulo da varredura LiDAR (0° a 360°) |
| `distance` | LD06 | Distância medida (em unidades de 0.25mm) |
| `intensity` | LD06 | Intensidade do retorno laser (0-255) |

### Formato de Armazenamento
- **Arquivo:** CSV (Comma-Separated Values)
- **Exemplo de linha:**
  ```
  14497338,-472,-280,16760,-26,-116,-160,-312,156,16272,-139,-549,-464,32291,923,224
  ```

---

## 🔌 Conexões de Hardware

### Pinos Raspberry Pi Pico Utilizados

#### I2C (Comunicação com IMUs)
- **IMU 1:** I2C0 (SDA: Pino 8, SCL: Pino 9)
- **IMU 2:** I2C1 (SDA: Pino 10, SCL: Pino 11)

#### UART (Comunicação LiDAR)
- **UART 1:**
  - RX (Recebimento): Pino 5
  - TX (Transmissão): Pino 4 (não utilizado)
- **PWM (Controle do LiDAR):** Pino 6

#### SPI (Cartão SD)
- **SPI 0:**
  - MOSI (Transmissão): Pino 19
  - MISO (Recepção): Pino 20
  - SCK (Relógio): Pino 18
- **CS (Chip Select):** Pino 17

---

## 🏗️ Estrutura Física

### Plataforma
- **Tipo Inicial:** Drone terrestre (Carro de controle remoto modificado)
- **Evolução Planejada:** Drone aéreo
- **Suporte:** Placa customizada (arquivo: `Placa suporte.SLDPRT` em SolidWorks)

### Alimentação
- **Fonte:** Bateria (tipo e voltagem conforme drone)
- **Reguladores:** Necessários para fornecer voltagens adequadas aos componentes

---

## 📈 Processamento de Dados

### Transformação de Coordenadas
```
LiDAR (sensor) → Corpo (drone) → Referencial Global
```

### Metodologia de Cálculo
1. **Leitura de Sensores:** Coleta simultânea de IMU e LiDAR
2. **Compensação Inercial:** Transformação de acelerações do corpo para referencial global
3. **Estimativa de Posição:** Integração dupla de aceleração (v = ∫a, s = ∫v)
4. **Geração de Nuvem de Pontos:** Transformação de coordenadas polares (ângulo, distância) para cartesianas com posição do drone

### Tratamento de Dados
- **Filtro de Intensidade LiDAR:** Apenas pontos com intensidade ≥ 223
- **Integração Numérica:** Métodos de Euler ou Trapézio para transformação aceleração → velocidade → posição

---

## 🧪 Ambiente de Testes

### Coleta de Dados Inicial
- **Local:** Ambiente controlado (laboratório/área interna)
- **Método:** Teste com medidas conhecidas em 4 pontos de referência
- **Ângulos Testados:** 0°, 90°, 180°, 270°
- **Duração:** 1 minuto de amostragem por ângulo
- **Dados Coletados:** 6.584 aferições

### Valores de Referência
| Ângulo | Distância Esperada |
|-------:|-------------------:|
| 0°     | 1700 mm            |
| 90°    | 2000 mm            |
| 180°   | 1300 mm            |
| 270°   | 1000 mm            |

---

## 📝 Arquivos Relacionados

- **Código Firmware:** `arduino/Completo/Completo.ino` - Firmware completo do sistema (SDK Pico/C)
- **Análise LiDAR:** `drafts/dados_lidar.md` - Análise do filtro de intensidade
- **Processamento de Dados:** `lidar_analysis.py` - Script Python para análise
- **Dados Coletados:** 
  - `DADOS.CSV` - Leitura com IMU e LiDAR
  - `LIDAR.CSV` - Dados apenas do LiDAR em teste controlado
- **Documentação Técnica:** `main.tex` - Relatório LaTeX completo
- **Placa de Suporte:** `Placa suporte.SLDPRT`, `Placa suporte.STL` - Projeto mecânico

---

## ✅ Checklist de Funcionalidades

- [x] Leitura de 2 IMU via I2C
- [x] Leitura de LiDAR LD06 via UART
- [x] Gravação de dados em cartão SD
- [x] Sincronização temporal de sensores
- [x] Filtragem de qualidade do LiDAR
- [x] Análise de precisão em ambiente controlado
- [ ] Processamento completo de nuvem de pontos
- [ ] Validação em drone aéreo

---

## 📚 Referências Teóricas

- **Microcontroladores:** Processamento embarcado e aquisição de dados em tempo real
- **IMU:** Medição de aceleração e velocidade angular; derivação de atitude
- **LiDAR:** Medição por tempo de voo (ToF); varredura angular
- **Integração Numérica:** Métodos de Euler e Trapézio para cinemática
- **Transformação de Coordenadas:** Matrizes de rotação SO(3); ângulos de Euler (roll, pitch, yaw)
