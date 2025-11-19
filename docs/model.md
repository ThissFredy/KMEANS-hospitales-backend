# Implementación de K-Means para la Optimización de Ubicación Hospitalaria

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Backend-FastAPI%20%7C%20NumPy-blue)
![Docker](https://img.shields.io/badge/Deploy-Docker%20%7C%20Render-cyan)

<div align="center" style="margin:16px 0;">
  <div style="display:inline-flex;align-items:center;gap:12px;padding:12px 16px;border-radius:8px;
              background:#f6f8fa;border:1px solid #e1e4e8;box-shadow:0 1px 2px rgba(0,0,0,0.03);">
    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
         alt="GitHub" width="40" height="40" style="flex:0 0 40px;">
    <div style="text-align:left;">
      <div style="font-weight:700;font-size:1.05rem;color:#24292f;">
        <a href="https://github.com/ThissFredy/KMEANS-hospitales-backend.git" target="_blank" rel="noopener" style="color:inherit;text-decoration:none;">
          KMEANS-hospitales-backend
        </a>
      </div>
      <div style="color:#57606a;font-size:0.9rem;">
        Repositorio · API K-Means de ubicación hospitalaria
      </div>
    </div>
  </div>
</div>
<div align="center" style="margin:16px 0;">
  <div style="display:inline-flex;align-items:center;gap:12px;padding:12px 16px;border-radius:8px;
              background:#f6f8fa;border:1px solid #e1e4e8;box-shadow:0 1px 2px rgba(0,0,0,0.03);">
    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
         alt="GitHub" width="40" height="40" style="flex:0 0 40px;">
    <div style="text-align:left;">
      <div style="font-weight:700;font-size:1.05rem;color:#24292f;">
        <a href="https://github.com/ThissFredy/kmeans-hospitales-frontend.git" target="_blank" rel="noopener" style="color:inherit;text-decoration:none;">
          kmeans-hospitales-frontend
        </a>
      </div>
      <div style="color:#57606a;font-size:0.9rem;">
        Repositorio · UI K-Means de ubicación hospitalaria
      </div>
    </div>
  </div>
</div>

<div align="center" style="margin:16px 0;">
  <div style="display:inline-flex;align-items:center;gap:12px;padding:12px 16px;border-radius:8px;
              background:#f6f8fa;border:1px solid #e1e4e8;box-shadow:0 1px 2px rgba(0,0,0,0.03);">
    <img src="https://colab.research.google.com/img/colab_favicon_256px.png"
         alt="GitHub" width="40" height="40" style="flex:0 0 40px;">
    <div style="text-align:left;">
      <div style="font-weight:700;font-size:1.05rem;color:#24292f;">
        <a href="https://colab.research.google.com/drive/1HeS-y8HrfYeNbaTmXbrkJq6UfNb650ry?usp=sharing" target="_blank" rel="noopener" style="color:inherit;text-decoration:none;">
          MODELO-KMEANS-HOSPITALES.ipynb
        </a>
      </div>
      <div style="color:#57606a;font-size:0.9rem;">
        Notebook · Implementación Manual de K-Means
      </div>
    </div>
  </div>
</div>

Este proyecto es una implementación interactiva del algoritmo de **Machine Learning no supervisado K-Means**, diseñado para resolver un problema de infraestructura urbana: encontrar la ubicación óptima de $A$ hospitales dada una distribución geográfica de $N$ residencias en un plano $M \times M$.

## Introducción y Objetivo

El problema de la ubicación de instalaciones busca minimizar el costo de transporte o distancia entre los proveedores de servicios (hospitales) y los consumidores (casas).

En este simulador:

-   **El Universo:** Una cuadrícula plana de $M \times M$ coordenadas.
-   **Los Datos (Puntos):** $N$ casas distribuidas aleatoriamente.
-   **Los Centroides:** $A$ hospitales cuya posición se optimiza iterativamente.

El objetivo del algoritmo es encontrar las coordenadas $(x, y)$ para los hospitales tal que la distancia promedio que cualquier ciudadano debe recorrer para llegar a su hospital más cercano sea mínima.

---

## Marco Teórico

### 1. Algoritmo K-Means

K-Means es un algoritmo de agrupamiento (clustering) que particiona un conjunto de datos en $K$ grupos predefinidos (en nuestro caso, $A$ hospitales). Funciona iterando dos pasos principales hasta la convergencia:

1.  **Asignación (Expectation):** Cada punto de datos se asigna al centroide más cercano.
2.  **Actualización (Maximization):** Se recalculan los centroides tomando el promedio de todos los puntos asignados a ese grupo.

### 2. Distancia Euclidiana

Para calcular la "cercanía", utilizamos la distancia euclidiana, que representa la distancia lineal física en un plano 2D. La fórmula entre una casa $P(x_1, y_1)$ y un hospital $Q(x_2, y_2)$ es:

$$d(P, Q) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

### 3. Métricas de Rendimiento

El sistema calcula en tiempo real dos métricas clave:

-   **Inercia (WCSS - Within-Cluster Sum of Squares):**
    Es la suma de las distancias al cuadrado de cada punto a su centroide asignado. Matemáticamente, buscamos minimizar este valor. Una inercia baja indica agrupamientos compactos.
    $$\sum_{i=0}^{n} \min_{\mu_j \in C} (||x_i - \mu_j||^2)$$

-   **Distancia Promedio:**
    Una métrica más interpretable para el usuario final. Representa, en promedio, cuántas unidades de distancia debe recorrer una persona para llegar al hospital.

## Arquitectura del Sistema

El proyecto sigue una arquitectura desacoplada **Cliente-Servidor** contenerizada con Docker.

### Backend (API & Modelo)

-   **Lenguaje:** Python 3.10
-   **Framework:** FastAPI
-   **Cálculo Numérico:** NumPy
-   **Persistencia:** Memoria volátil

### Frontend (UI & Visualización)

-   **Framework:** Next.js 14 (React).
-   **Estilos:** Tailwind CSS para un diseño responsivo y moderno.
-   **Visualización:** SVG nativo para renderizar la cuadrícula, casas y hospitales con alto rendimiento.
-   **Estado:** React Hooks (`useState`, `useEffect`) manejando una máquina de estados finita.

---

## Lógica del Negocio

La aplicación implementa una **Máquina de Estados Estricta** para guiar al usuario a través del proceso algorítmico correcto. No se permite saltar pasos ilógicos.

| Paso (Step) | Estado del Sistema | Acción Permitida                                                   |
| :---------- | :----------------- | :----------------------------------------------------------------- |
| **0**       | Inicio             | Configurar tamaño del Grid ($M$).                                  |
| **1**       | Grid Configurado   | Generar población aleatoria ($N$).                                 |
| **2**       | Población Lista    | Definir e inicializar Hospitales ($A$).                            |
| **3**       | Listo para Iterar  | **Asignar Clusters:** Calcular distancias y asignar casas.         |
| **4**       | Clusterizado       | **Actualizar Centroides:** Mover hospitales a la media aritmética. |
| **5**       | Optimizado         | Volver al Paso 3 (Iterar nuevamente) hasta convergencia.           |

## Despliegue en Producción

La aplicación está desplegada en **Render** utilizando contenedores Docker orquestados.

### Estrategia de Conexión

Dado que el Frontend y el Backend están en repositorios y servicios separados:

1.  **Backend:** Se despliega primero. Render le asigna una URL pública (ej. `https://kmeans-hospitales-backend.onrender.com`).
2.  **Frontend:** Se despliega después. Durante la construcción (build), se le inyecta la URL del backend mediante una variable de entorno.

### Configuración de Variables (Environment Variables)

En el servicio del Frontend en Render, se configuró:

-   `NEXT_PUBLIC_API_URL`: `https://kmeans-hospitales-backend.onrender.com`

Esto permite que el cliente React sepa exactamente a dónde enviar las peticiones HTTP para ejecutar los cálculos matemáticos.
