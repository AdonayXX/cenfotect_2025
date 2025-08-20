# Avance Preliminar del Proyecto

## 1. Información del Proyecto
**Nombre del Proyecto:**  
Sistema de Control de Acceso Universitario por Reconocimiento Facial

**Equipo:**  
- Kevin Juárez  
- Emerson Adonay  
- Jean Paladino  
- Maria Artavia  

**Roles:**  
- **Emerson Adonay:** Líder de Proyecto y Desarrollador  
- **Kevin Juárez:** Desarrollador  
- **Jean Paladino:** Desarrollador  
- **Maria Artavia:** Desarrollador  

## 2. Descripción y Justificación

**Problema que se aborda:**  
La falta de un sistema de control de acceso automatizado, seguro y eficiente en las instalaciones universitarias, lo que permite el ingreso de personas no autorizadas, la suplantación de identidad para acceder a servicios (bibliotecas, laboratorios) y dificulta una respuesta rápida ante incidentes de seguridad.

**Importancia y contexto:**  
En un entorno universitario con alto flujo de personas, garantizar la seguridad de estudiantes, docentes y personal es fundamental. Este proyecto plantea una solución que aumenta la seguridad restringiendo el acceso no autorizado, optimiza el uso de recursos y agiliza procesos en espacios como bibliotecas, comedores y laboratorios.

**Usuarios/beneficiarios:**  
- **Directos:** Estudiantes, docentes y personal administrativo.  
- **Indirectos:** Administración universitaria y departamento de seguridad.

## 3. Objetivos del Proyecto

**Objetivo General:**  
Desarrollar un prototipo de sistema de control de acceso basado en reconocimiento facial, operado por un microcontrolador, para mejorar la seguridad y gestión de servicios en el campus.

**Objetivos Específicos:**  
1. Diseñar y ensamblar el hardware del sistema, integrando microcontrolador, cámara y actuador.  
2. Implementar un algoritmo de reconocimiento facial para identificar usuarios registrados.  
3. Crear una base de datos segura y eficiente para almacenar perfiles y datos faciales autorizados.  

## 4. Requisitos Iniciales

**Lista breve de lo que el sistema debe lograr:**  
- **Requisito 1:** Capturar el rostro en tiempo real a una distancia de 30 a 80 cm.  
- **Requisito 2:** Procesar la imagen y compararla con la base de datos, emitiendo acceso concedido o denegado.  
- **Requisito 3:** Registrar cada intento de acceso, incluyendo ID (si se reconoce), fecha y hora.


## 5. Diseño Preliminar del Sistema

**Arquitectura inicial (diagrama):**  

```text
[Usuario] → [Cámara OV2640] → [ESP32-CAM] → [Base de Datos de Usuarios]
                                     ↓
                            [LED Verde / LED Rojo + Zumbador]
                                     ↓
                               [Registro de Evento]
```

*Ilustración 1: Diagrama conceptual del flujo del sistema*  

![Diagrama del Sistema](https://res.cloudinary.com/ddpyc9gjq/image/upload/DiagramaCenfotec.jpg)

**Componentes previstos:**  
- **Microcontrolador:** ESP32-CAM (procesador doble núcleo, Wi-Fi/Bluetooth)  
- **Sensores/actuadores:**  
  - Cámara OV2640 (integrada en ESP32-CAM)  
  - LED verde para acceso concedido  
  - LED rojo + zumbador para acceso denegado  
- **LLM/API:** Integración con API de IA (DeepSeek) para análisis/clasificación de eventos (a evaluar en fase de integración).  
- **Librerías y herramientas:**  
  - CircuitPython (por confirmar compatibilidad con cámara en ESP32-CAM)  
  - Librería `esp32-camera`, GPIO y HTTP/REST para comunicación con APIs  

**Bocetos o esquemas:**  
https://res.cloudinary.com/ddpyc9gjq/image/upload/DiagramaCenfotec.jpg

## 6. Plan de Trabajo

**Cronograma preliminar:**  

| Hito                                            | Fecha Estimada |
|------------------------------------------------|----------------|
| Diseño de hardware y selección de componentes  | Semana 1       |
| Configuración inicial del ESP32-CAM            | Semana 2       |
| Implementación de reconocimiento facial        | Semana 3       |
| Integración con base de datos                  | Semana 4       |
| Pruebas y ajustes finales                      | Semana 5       |

**Riesgos identificados y mitigaciones:**  
- **Riesgo 1:** Compatibilidad de cámara/stack (CircuitPython + ESP32-CAM/OV2640).  
  - *Mitigación:* Evaluar alternativas (MicroPython o Arduino/ESP-IDF) y/o procesar reconocimiento en servidor.  
- **Riesgo 2:** Precisión del reconocimiento en condiciones de baja iluminación o ángulos variables.  
  - *Mitigación:* Mejorar condiciones de luz, capturar dataset por usuario, ajustar umbrales y pruebas en campo.
- **Riesgo 3:** Dificultad con adquisición de componenetes necesarios en la implementación.

## 7. Prototipos conceptuales (si aplica)

**Código mínimo de prueba:**  
_Pendiente: aún no se han realizado prototipos conceptuales._

**Evidencia visual:**  
_Pendiente._
** Link del repositorio del frontend **
[Este es un enlace externo hacía el frontend](https://github.com/EkarCortes/SeguridadFront)
