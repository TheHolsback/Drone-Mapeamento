# Teoria

Obs: adicionar na seção de 

Para a teoria iremos utilizar as equações de movimento newtoniano, mais especificamente a aceleração para conseguir determinar a posição do drone e assim obter com a maior precisão possível a nuvem de pontos.

## Dados

O microcontrolador obterá 11 informações por ciclo:

- Tempo
- Aceleração X
- Aceleração Y
- Aceleração Z
- Atitude X
- Atitude Y
- Atitude Z
- Distância LiDar (Eixo) X
- Angulo LiDar X
- Distância LiDar Y
- Angulo LiDar Y

Com todas essas informações poderemos estimar com precisão a nuvem de pontos

## Cálculos Posição (por eixo)

Para obter a posição pelo tempo, primeiro é necesário fazer duas integrações:

$$
a(t)=\frac{dv}{dt}⇒v(t)=v0+∫a(t)dt\\
v(t)=\frac{ds}{dt}⇒s(t)=s0+∫v(t) dt\\
$$

Mas antes de fazermos essa integração é importante lembrar que as acelerações não correspondem as posições reais pois quando o drone acelera, ele acelera inclinado. Logo, para obtermos as posições corretas primeiro precisamos obter as acelerações ortogonais relativas ao mundo.

## Dados obtidos

| Coluna | Significado |
| --- | --- |
| aceleracao_x | aceleração no eixo X do drone |
| aceleracao_y | aceleração no eixo Y do drone |
| aceleracao_z | aceleração no eixo Z do drone |
| inclinacao_x | roll (ϕ) |
| inclinacao_y | pitch (θ) |
| inclinacao_z | yaw (ψ) |

### 1. Vetor aceleração no corpo

$$

\mathbf{a}_{corpo} =
\begin{bmatrix}
a_x \\ a_y \\ a_z
\end{bmatrix}
$$

### 2. Vetor gravidade no mundo

$$
\mathbf{g}_{mundo} =
\begin{bmatrix}
0 \\ 0 \\ -g
\end{bmatrix}
\quad \text{com } g = 9.811
$$

*(Z do mundo aponta para cima)*

### 3. Matriz de rotação Corpo → Mundo

Temos as inclinações do drone:

- **roll** ( $\phi$ ) → rotação em X
- **pitch** ( $\theta$ ) → rotação em Y
- **yaw** ( $\psi$ ) → rotação em Z

A matriz de rotação típica (ordem ZYX):

$$
\mathbf{R}_{mundo}^{drone}=
R_z(\psi)
R_y(\theta)
R_x(\phi)
$$

Onde:

$$
R_x(\phi)=
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos\phi & -\sin\phi \\
0 & \sin\phi & \cos\phi
\end{bmatrix}
$$

$$
R_y(\theta)=
\begin{bmatrix}
\cos\theta & 0 & \sin\theta \\
0 & 1 & 0 \\
-\sin\theta & 0 & \cos\theta
\end{bmatrix}
$$

$$
R_z(\psi)=
\begin{bmatrix}
\cos\psi & -\sin\psi & 0 \\
\sin\psi & \cos\psi & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

### 4. Removendo a gravidade corretamente

$$
\mathbf{a}_{mundo}
=
\mathbf{R}_{mundo}^{corpo}
\mathbf{a}_{corpo}
-
\mathbf{g}_{mundo}

$$

Essa forma é válida pois a IMU mede aceleração específica.

### Código para transformar as acelerações:

```python
import numpy as np

def rot_x(phi):
    return np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi),  np.cos(phi)]
    ])

def rot_y(theta):
    return np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rot_z(psi):
    return np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi),  np.cos(psi), 0],
        [0, 0, 1]
    ])

# dados da tabela
a_body = np.array([ax, ay, az])

roll  = phi
pitch = theta
yaw   = psi

R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

g_world = np.array([0, 0, -9.81])

a_world = R @ a_body + g_world
```

### Dados Discretos

Como nossos dados são discretos podemos calcular levando em conta que entre as aferições a acelereção foi constante, a tabela base será:

| i | tempo (t_i) | aceleração (a_i) |
| --- | --- | --- |

Com passo de tempo:

$$
\Delta t_i = t_i - t_{i-1}
$$

Com a consideração de aceleração constante entre as aferições a área a baixo do gráfico se aproxima a de um trapézio. 

Primeiro o cálculo da velocidade:

$$
v_i=v_{i-1}+\frac{a_{i−1}+a_i}{2}\Delta t_i
$$

Agora podemos calcular a posição em relação ao tempo com:

$$
s_i = s_{i-1} + \frac{v_{i-1}+v_i}{2}\,\Delta t_i
$$

sabendo então das formulas, podemos criar a seguinte função em python para obter a lista de posição pelo tempo:

```python
import numpy as np

t = np.array([...])   # tempo
a = np.array([...])   # aceleração

def posicao_tempo(a,t):
	v = np.zeros_like(t)
	s = np.zeros_like(t)
	
	for i in range(1, len(t)):
	    dt = t[i] - t[i-1]
	    v[i] = v[i-1] + 0.5*(a[i-1] + a[i])*dt
	    s[i] = s[i-1] + 0.5*(v[i-1] + v[i])*dt

	return s
```

## Calculo do ponto

Vamos tomar de exemplo o LiDAR no eixo Z como exemplo

Para cada ponto do LiDAR, vamos precisar fazer **3 transformações**:

1. **Ponto no referencial do LiDAR**
2. **Transformação LiDAR → corpo do drone**
3. **Transformação corpo do drone → referencial do solo (mundo)**
4. Somar a posição global do drone, obtida anteriormente

## 1. Referenciais envolvidos

### Referencial do LiDAR (L)

- Origem: centro do LiDAR
- Eixos: definidos pelo fabricante
- Medidas:
    - distância $r$
    - ângulo horizontal $\theta$ (360° no plano X–Y do LiDAR)
    - *(se houver)* ângulo vertical $\phi$

### Referencial do Drone (D)

- Origem: centro de massa ou ponto definido
- Eixos típicos:
    - X: frente
    - Y: direita
    - Z: para cima

### Referencial do Mundo / Solo (W)

- Fixo no ambiente
- X, Y no plano horizontal
- Z vertical (altura)

## 2. Coordenadas do ponto no referencial do LiDAR

Como o LiDAR é **2D (360° no plano X–Y)**:

$$
\mathbf{p}_{lidar} =
\begin{bmatrix}
x_L \\
y_L \\
z_L
\end{bmatrix}
=
\begin{bmatrix}
r\cos\theta \\
r\sin\theta \\
0
\end{bmatrix}
$$

Se houver **inclinação da aferição** (ângulo vertical ϕ\phiϕ):

$$
\begin{aligned}
x_L &= r\cos\phi\cos\theta \\
y_L &= r\cos\phi\sin\theta \\
z_L &= r\sin\phi
\end{aligned}
$$

## 3. Transformação LiDAR → Drone

Como o LiDAR não está perfeitamente alinhado com o drone, vamos precisar de:

- Rotação fixa Rdronelidar $\mathbf{R}_{drone}^{lidar}$
- Translação fixa tdronelidar $\mathbf{t}_{drone}^{lidar}$

Então:

$$
\mathbf{p}_{drone}
=
\mathbf{R}_{drone}^{lidar}\mathbf{p}_{lidar}
+
\mathbf{t}_{drone}^{lidar}
$$

## 4. Atitude do drone: Rotação Drone → Mundo

Usando **roll–pitch–yaw** (ordem ZYX):

$$
\mathbf{R}_{mundo}^{corpo}
=
R_z(\psi)\,R_y(\theta)\,R_x(\phi)
$$

(com as mesmas matrizes explicadas anteriormente)

## 5. Transformação final para o referencial do solo

Como calculamos anteriormente a posição do drone na caverna:

$$
\mathbf{p}_{drone}^{mundo}
=
\begin{bmatrix}
x_d \\ y_d \\ z_d
\end{bmatrix}
$$

Então:

$$
\boxed{
\mathbf{p}_{mundo}
=
\mathbf{R}_{mundo}^{drone}
\mathbf{R}_{drone}^{lidar}
\mathbf{p}_{lidar}
+
\mathbf{R}_{mundo}^{drone}
\mathbf{t}_{drone}^{lidar}
+
\mathbf{p}_{drone}^{mundo}
}
$$

## Código para a transformação do ponto:

```python
import numpy as np

# funções rot_x, rot_y e rot_z já declaradas anteriormente

# ponto no lidar
r = 5.0
theta = np.deg2rad(30)
phi = np.deg2rad(0)

p_lidar = np.array([
    r*np.cos(phi)*np.cos(theta),
    r*np.cos(phi)*np.sin(theta),
    r*np.sin(phi)
])

# atitude do drone
roll = np.deg2rad(5)
pitch = np.deg2rad(3)
yaw = np.deg2rad(45)

R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

p_world = R @ p_lidar
```