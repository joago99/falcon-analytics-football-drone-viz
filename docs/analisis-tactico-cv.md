# **Deconstrucción del Análisis Táctico Mediante Visión Computacional: Desmitificación de Modelos Generativos y Pipeline de Ejecución Local de Código Abierto**

La convergencia entre la inteligencia artificial generativa y el análisis cinemático en el deporte profesional ha suscitado un notable interés técnico y mediático a raíz de demostraciones virales difundidas en redes sociales1. En estas secuencias, se atribuye a modelos fundacionales avanzados, específicamente GPT-6 Astra en articulación con plataformas como Higgsfield AI, la capacidad de procesar secuencias de vídeo televisivo y reconstruir automáticamente la física de un partido de fútbol dentro del software de modelado 3D Blender mediante simples instrucciones textuales1. Este documento descompone la base técnica real de dicho fenómeno, delimita las fronteras operativas entre los modelos de lenguaje y la visión computacional determinista, y expone una arquitectura reproducible de código abierto para ejecutar localmente y a coste cero un sistema integral de seguimiento táctico y visualización tridimensional5.

## **Desmitificación Técnica del Fenómeno Viral**

El material analizado tiene su origen en una demostración divulgada en septiembre de 2026 por Axultan Alimkulov, líder de producto en Higgsfield AI1. El clip presenta una retransmisión deportiva sobre la que se proyectan cajas delimitadoras de jugadores, vectores direccionales de pases, estimaciones de posesión y un radar táctico bidimensional en tiempo real1. Su posterior viralización en canales secundarios alteró sustancialmente la naturaleza técnica de la herramienta al etiquetarla como un simulador físico de Blender accionado por agentes generativos multimodales1.  
La realidad computacional subyacente difiere drásticamente de esta narrativa comercial7. Los modelos de lenguaje y razonamiento como GPT-6 Astra carecen de motores internos de cálculo cinemático de cuerpos rígidos o transformaciones proyectivas de vóxeles a partir de vídeo sin procesar3. La intervención de estos sistemas en tales flujos de trabajo se restringe a la orquestación lógica mediante protocolos de interoperabilidad como el Model Context Protocol (MCP)4. En este marco, el modelo generativo asume la función de un agente de ingeniería de software capaz de escribir código intermediario, invocar librerías externas de visión artificial, configurar esquemas de inferencia y diseñar la interfaz gráfica del usuario3.  
El rastreo cinemático de atletas y el cálculo posicional no emanan de algoritmos generativos de difusión o redes autorregresivas de texto a vídeo, sino de arquitecturas clásicas y discriminativas de visión artificial entrenadas específicamente para el seguimiento de múltiples objetos y la estimación de homografías9. Del mismo modo, la vinculación con Blender no se deriva de una reconstrucción neuronal del espacio tridimensional generada de manera espontánea4. El proceso responde a una transferencia de datos tabulares donde las trayectorias calculadas en dos dimensiones en el plano del campo se exportan en formatos estructurados para ser leídas por la interfaz de programación de aplicaciones de Blender (bpy), asignando fotogramas clave de traslación a primitivas geométricas sobre un escenario tridimensional previamente configurado12.

## **Arquitectura de Visión Computacional para el Rastreo Táctico**

La extracción de métricas espaciales y la proyección tridimensional de un partido de fútbol a partir de una única señal de emisión televisiva exige una secuencia algorítmica modular rigurosa15. Cada etapa del sistema responde a principios matemáticos bien definidos y resuelve problemas cinemáticos específicos10.

| Etapa del Pipeline | Enfoque Algorítmico y Modelos | Entrada de Datos | Salida Estructurada | Restricción Técnica Crítica |
| :---- | :---- | :---- | :---- | :---- |
| **Detección de Entidades** | YOLOv8x, YOLO11, RT-DETR9 | Fotogramas de vídeo RGB (![][image1]) | Bounding boxes ![][image2], clase y confianza | Pérdida de detección del esférico por desenfoque y escala reducida18. |
| **Seguimiento Temporal** | ByteTrack, BoT-SORT, OC-SORT9 | Coordenadas de detección cuadro a cuadro | Identificadores persistentes (![][image3]) por objeto | Ruptura de trayectorias tras oclusiones prolongadas9. |
| **Clasificación de Equipos** | K-Means, SigLIP, UMAP, Espacio HSV15 | Recortes de imagen de cada silueta | Categoría (Equipo local, visitante o árbitros) | Sensibilidad a contrastes extremos de iluminación y sombras18. |
| **Detección de Puntos Clave** | HRNet, YOLOv8-pose, ResNet6 | Fotograma panorámico del terreno | Coordenadas ![][image4] de 29 a 32 marcas reglamentarias | Disminución de puntos visibles en planos cerrados o con zoom6. |
| **Calibración y Homografía** | DLT con estimación robusta RANSAC11 | Puntos clave detectados vs. modelo métrico FIFA | Matriz de proyección planar ![][image5] | Exigencia mínima de cuatro puntos no colineales para convergencia6. |
| **Reconstrucción 3D / Métricas** | Álgebra lineal cinemática y script bpy \[cite: 12, 15\] | Coordenadas métricas ![][image6] en el plano real | Ficheros tabulares, velocidades y keyframes en Blender | Propagación de ruido métrico en aceleraciones instantáneas6. |

El sistema inicia con la detección simultánea de jugadores, cuerpo arbitral y balón mediante redes neuronales convolucionales de etapa única9. La asignación temporal de identidades descansa en ByteTrack, cuyo algoritmo destaca por preservar las trayectorias incluso frente a detecciones con puntuaciones de baja confianza9. Mediante el uso de un Filtro de Kalman para predecir la posición futura y la posterior asociación en dos fases empleando la intersección sobre unión (IoU), ByteTrack mantiene la identidad de los futbolistas cuando ocurren cruces o agrupamientos dentro del área9. Para discriminar los conjuntos en pugna sin entrenamiento supervisado previo, se aíslan los parches visuales del torso y se extraen descriptores cromáticos o incrustaciones de modelos de visión-lenguaje como SigLIP, aplicando luego agrupamiento mediante ![][image7]\-Means sobre componentes principales15.  
El puente entre las coordenadas de la pantalla y el espacio físico real se fundamenta en la geometría proyectiva plana11. Una homografía define una transformación invertible que mapea un punto del plano del sensor ![][image8] al plano bidimensional del terreno de juego ![][image9] reglamentado por FIFA (![][image10])11:  
![][image11]  
La conversión a coordenadas métricas cartesianas desnormalizadas se efectúa a través de la proyección normalizada11:  
![][image12]  
Para resolver los 8 grados de libertad de la matriz ![][image13], es matemáticamente imperativo disponer de al menos cuatro pares de correspondencias no colineales6. Los modelos de estimación de pose aplicados al campo ubican marcas canónicas (cruces de líneas, puntos de penal, intersección de la línea media y circunferencias), permitiendo que el estimador RANSAC calcule dinámicamente la transformación que absorbe las oscilaciones de panorámica, inclinación y zoom de la cámara de retransmisión11.

## **Ecosistemas y Repositorios de Código Abierto**

La replicación de esta tecnología prescinde enteramente de servicios corporativos de suscripción. Existen múltiples entornos desarrollados por la comunidad de código abierto que integran detección, calibración de campos deportivos y extracción cinemática bajo licencias de libre utilización5.

| Ecosistema / Herramienta | Enfoque Tecnológico | Pesos de Redes Neuronales | Nivel de Madurez y Propósito | Licencia de Uso |
| :---- | :---- | :---- | :---- | :---- |
| **Roboflow Sports** | YOLOv8/11, Supervision, SigLIP, Homografía planar | Modelos accesibles en Roboflow Universe5 | Librería de utilidades tácticas y calibración de campo5 | MIT5 |
| **Eagle (Football Analytics)** | YOLOv8 (HD 960px), HRNet, BoT-SORT, Voronoi | Descarga automática vía scripts internos6 | Aplicación de terminal integrada para análisis cinemático6 | Código Abierto6 |
| **FootballVision** | YOLOv5/v8, ByteTrack, Autocalibración homográfica | Pesos preentrenados distribuidos en el repositorio15 | Sistema orientado a la extracción de distancias y posesión15 | MIT15 |
| **SoccerNet-Calibration / TVCalib** | Redes de puntos clave y conics fitting para TV | Modelos basados en el SoccerNet Challenge11 | Calibración analítica avanzada para tomas complejas11 | Académica / MIT25 |
| **Wunderscout** | Detección YOLO, proyección normalizada \[0,1\], CSV | Compatibilidad con cualquier punto de control YOLO23 | Exportación directa de métricas posicionales espaciales23 | MIT23 |

Tanto **Roboflow Sports** como **Eagle** constituyen las alternativas más avanzadas para procesamiento directo5. Mientras la primera aporta módulos versátiles para construir soluciones a medida con la biblioteca Supervision, Eagle optimiza la tasa de recuperación de objetos diminutos incorporando inferencia sobre recortes a 960 píxeles, aspecto determinante para mantener la trayectoria del balón en jugadas aéreas5.

## **Infraestructura y Requerimientos de Cómputo**

El procesamiento de flujos de vídeo de alta definición mediante múltiples redes neuronales concurrentes requiere dimensionar el hardware en función de la latencia tolerable y la resolución espacial6.

| Perfil de Ejecución | Hardware Mínimo / Plataforma | Capacidad de Memoria | Rendimiento Estimado | Coste Económico |
| :---- | :---- | :---- | :---- | :---- |
| **Nube Gratuita** | Google Colab / Kaggle (NVIDIA T4 o L4)10 | 15–16 GB VRAM / 12 GB RAM del sistema | 18–25 fotogramas por segundo (procesamiento diferido)6 | Gratuito bajo asignación por cuotas20 |
| **Local Mínimo** | Intel Core i5 (10.ª gen) o AMD Ryzen 5, NVIDIA GTX 1660 Super | 6 GB VRAM GDDR6 / 16 GB RAM DDR4 | 8–15 fotogramas por segundo (YOLOv8 medium)6 | Coste cero sobre hardware existente |
| **Estación de Trabajo Local** | AMD Ryzen 7 5800X / Core i7 13700, NVIDIA RTX 3060 / 4070 | 12 GB VRAM GDDR6X / 32 GB RAM DDR5 | 35–65 fotogramas por segundo en tiempo real continuo10 | Coste cero en software y licencias |

El despliegue en entornos gratuitos como Google Colab es completamente viable mediante la importación de repositorios basados en PyTorch y aceleradores CUDA estándar, ofreciendo la capacidad de procesar fragmentos tácticos de partidos sin incurrir en desembolsos económicos de ningún tipo10.

## **Implementación Técnica del Pipeline de Análisis y Animación**

Para replicar de forma fidedigna el flujo de seguimiento y su posterior traslado tridimensional, se ejecuta un proceso en dos fases: un script de visión artificial encargado del análisis geométrico y un script de automatización en Blender que materializa la escena visual12.

### **Fase 1: Configuración de Dependencias del Sistema**

En un entorno local con Python 3.10 o 3.11, se procede a inicializar el espacio virtual y compilar las dependencias matemáticas indispensables6:

Bash  
python3 \-m venv sports\_cv\_env  
source sports\_cv\_env/bin/activate

pip install torch torchvision \--index-url https://download.pytorch.org/whl/cu121  
pip install ultralytics supervision opencv-python pandas numpy  
pip install git+https://github.com/roboflow/sports.git

### **Fase 2: Pipeline de Visión Computacional, Calibración y Exportación Cinemática**

El siguiente programa procesa un archivo de vídeo monocámara (partido.mp4), identifica a los actores en el terreno de juego, calcula la matriz de homografía cuadro a cuadro a partir de las intersecciones de las líneas de cal y genera un archivo CSV con las coordenadas métricas relativas al campo de juego15:

Python  
import cv2  
import numpy as np  
import pandas as pd  
from ultralytics import YOLO  
import supervision as sv

\# Inicialización de modelos de detección y seguimiento  
detector\_entidades \= YOLO("yolov8x.pt")  
detector\_puntos\_campo \= YOLO("football-field-detection-f07vi.pt")  
rastreador \= sv.ByteTrack()

\# Correspondencias geométricas del campo según reglamento FIFA (105m x 68m)  
COORDENADAS\_CANONICAS\_CAMPO \= {  
    0: (0.0, 0.0),       \# Córner superior izquierdo  
    1: (0.0, 68.0),      \# Córner inferior izquierdo  
    2: (105.0, 0.0),     \# Córner superior derecho  
    3: (105.0, 68.0),    \# Córner inferior derecho  
    4: (52.5, 34.0),     \# Centro geométrico del terreno  
}

registro\_seguimiento \= \[\]  
indice\_cuadro \= 0

captura \= cv2.VideoCapture("partido.mp4")

while captura.isOpened():  
    estado, fotograma \= captura.read()  
    if not estado:  
        break

    \# 1\. Detección y seguimiento de jugadores  
    prediccion\_entidades \= detector\_entidades(fotograma, conf=0.4, verbose=False)\[0\]  
    detecciones \= sv.Detections.from\_ultralytics(prediccion\_entidades)  
    detecciones\_jugadores \= detecciones\[detecciones.class\_id \== 0\]  
    jugadores\_rastreados \= rastreador.update\_with\_detections(detecciones\_jugadores)

    \# 2\. Localización de puntos clave estructurales para calibración de cámara  
    prediccion\_campo \= detector\_puntos\_campo(fotograma, conf=0.5, verbose=False)\[0\]  
    puntos\_origen \= \[\]  
    puntos\_destino \= \[\]

    if prediccion\_campo.keypoints is not None:  
        for idx, coord in enumerate(prediccion\_campo.keypoints.xy\[0\]):  
            px, py \= float(coord\[0\]), float(coord\[1\])  
            fiabilidad \= float(prediccion\_campo.keypoints.conf\[0\]\[idx\])  
            if fiabilidad \> 0.5 and idx in COORDENADAS\_CANONICAS\_CAMPO:  
                puntos\_origen.append(\[px, py\])  
                puntos\_destino.append(COORDENADAS\_CANONICAS\_CAMPO\[idx\])

    \# 3\. Estimación robusta de la matriz de homografía  
    matriz\_H \= None  
    if len(puntos\_origen) \>= 4:  
        arr\_origen \= np.array(puntos\_origen, dtype=np.float32)  
        arr\_destino \= np.array(puntos\_destino, dtype=np.float32)  
        matriz\_H, \_ \= cv2.findHomography(arr\_origen, arr\_destino, cv2.RANSAC, 5.0)

    \# 4\. Proyección métrica y persistencia de trayectorias  
    for rectangulo, identificador in zip(jugadores\_rastreados.xyxy, jugadores\_rastreados.tracker\_id):  
        pos\_base\_x \= (rectangulo\[0\] \+ rectangulo\[2\]) / 2.0  
        pos\_base\_y \= rectangulo\[3\]

        pos\_campo\_x, pos\_campo\_y \= np.nan, np.nan  
        if matriz\_H is not None:  
            vector\_pixel \= np.array(\[\[\[pos\_base\_x, pos\_base\_y\]\]\], dtype=np.float32)  
            vector\_transformado \= cv2.perspectiveTransform(vector\_pixel, matriz\_H)  
            pos\_campo\_x \= vector\_transformado\[0\]\[0\]\[0\]  
            pos\_campo\_y \= vector\_transformado\[0\]\[0\]\[1\]

        registro\_seguimiento.append({  
            "cuadro": indice\_cuadro,  
            "id\_jugador": identificador,  
            "campo\_x\_metros": pos\_campo\_x,  
            "campo\_y\_metros": pos\_campo\_y  
        })

    indice\_cuadro \+= 1

captura.release()

dataframe\_metricas \= pd.DataFrame(registro\_seguimiento)  
dataframe\_metricas.to\_csv("datos\_tacticos\_partido.csv", index=False)

### **Fase 3: Automatización e Ingesta de Coordenadas en Blender**

Para reproducir visualmente la dinámica sin depender de capas cerradas de software, se recurre a la API interna de Blender12. Desde el editor de texto interno de Blender, se ejecuta el siguiente script, el cual genera el terreno reglamentario, crea entidades cilíndricas representativas de cada atleta y anima automáticamente su traslación sobre el plano tridimensional sincronizando sus posiciones fotograma a fotograma12:

Python  
import bpy  
import csv  
import os

ruta\_archivo\_csv \= bpy.path.abspath("//datos\_tacticos\_partido.csv")

if not os.path.exists(ruta\_archivo\_csv):  
    raise FileNotFoundError(f"Archivo de trayectoria no encontrado: {ruta\_archivo\_csv}")

nombre\_coleccion \= "Rastreo\_Táctico"  
if nombre\_coleccion not in bpy.data.collections:  
    coleccion\_rastreo \= bpy.data.collections.new(nombre\_coleccion)  
    bpy.context.scene.collection.children.link(coleccion\_rastreo)  
else:  
    coleccion\_rastreo \= bpy.data.collections\[nombre\_coleccion\]

\# Creación del plano métrico del campo (105 metros x 68 metros)  
if "Plano\_Terreno" not in bpy.data.objects:  
    bpy.ops.mesh.primitive\_plane\_add(size=1, location=(52.5, 34.0, 0.0))  
    campo \= bpy.context.active\_object  
    campo.name \= "Plano\_Terreno"  
    campo.scale \= (105.0, 68.0, 1.0)

instancias\_jugadores \= {}

with open(ruta\_archivo\_csv, mode="r") as descriptor\_archivo:  
    lector \= csv.DictReader(descriptor\_archivo)  
    for fila in lector:  
        if fila\["campo\_x\_metros"\] \== "" or fila\["campo\_x\_metros"\] \== "nan":  
            continue

        numero\_cuadro \= int(fila\["cuadro"\])  
        id\_jugador \= str(fila\["id\_jugador"\])  
        x\_metros \= float(fila\["campo\_x\_metros"\])  
        y\_metros \= float(fila\["campo\_y\_metros"\])  
        z\_metros \= 0.9  \# Centro de masa estándar para visualización

        \# Generación de la entidad 3D si es su primera aparición  
        if id\_jugador not in instancias\_jugadores:  
            malla \= bpy.data.meshes.new(f"Malla\_Jugador\_{id\_jugador}")  
            objeto\_jugador \= bpy.data.objects.new(f"Jugador\_{id\_jugador}", malla)  
            coleccion\_rastreo.objects.link(objeto\_jugador)

            bpy.context.view\_layer.objects.active \= objeto\_jugador  
            bpy.ops.mesh.primitive\_cylinder\_add(radius=0.45, depth=1.8, location=(x\_metros, y\_metros, z\_metros))  
            objeto\_jugador.data \= bpy.context.active\_object.data  
            bpy.data.objects.remove(bpy.context.active\_object)

            instancias\_jugadores\[id\_jugador\] \= objeto\_jugador

        \# Asignación de traslación e inserción de keyframe  
        nodo \= instancias\_jugadores\[id\_jugador\]  
        nodo.location \= (x\_metros, y\_metros, z\_metros)  
        nodo.keyframe\_insert(data\_path="location", frame=numero\_cuadro)

## **Limitaciones Operativas en Retransmisiones Reales**

La solidez de este pipeline se degrada cuando se traslada de fragmentos cortos y controlados de 15 a 30 segundos a partidos completos de 90 minutos7. Las emisiones deportivas comerciales introducen discontinuidades que quebrantan las asunciones básicas de la visión computacional monocámara7.  
La alternancia entre la toma táctica general y los planos cerrados o repeticiones a velocidad lenta interrumpe de forma irreversible la continuidad temporal del Filtro de Kalman7. Cada transición de plano anula la correspondencia de identidades, obligando al sistema a reidentificar a los 22 deportistas mediante redes neuronales de reidentificación de apariencia externa (Re-ID), un procedimiento altamente costoso que con frecuencia intercambia identificadores entre jugadores que visten equipaciones con patrones idénticos7.  
La variabilidad en la longitud focal y los movimientos de inclinación y barrido rápido provocan que los planos cerrados expongan menos de cuatro puntos de referencia sobre el césped6. En tales condiciones, la ecuación homográfica carece de solución matemática unívoca, interrumpiendo la traslación de píxeles a metros hasta que la cámara regresa a una perspectiva abierta6. Por último, el balón reglamentario sufre problemas severos de desenfoque de movimiento en desplazamientos rápidos, ocupando áreas no superiores a 20 o 25 píxeles en resolución estándar de 1080p, lo que suele originar pérdidas sistemáticas de la trayectoria balística en pases aéreos y disputas colectivas10. Por estas limitaciones, la élite profesional continúa empleando infraestructuras físicas de estadios inteligentes compuestas por múltiples cámaras fijas de alta frecuencia y sistemas de posicionamiento local por radiofrecuencia7.

## **Conclusiones**

La reconstrucción visual viralizada como un sistema autónomo basado en GPT-6 Astra y Blender obedece a una distorsión narrativa de técnicas formales de visión artificial1. Si bien la asistencia de modelos de lenguaje agiliza de forma drástica la integración modular del software, el motor funcional depende exclusivamente de detectores discriminativos de convolución o atención visual, filtros bayesianos de seguimiento y matrices proyectivas de homografía3.  
Cualquier investigador o analista técnico dispone de los recursos para reproducir de forma local, soberana y a coste cero el análisis cinemático de cualquier fragmento táctico utilizando herramientas consolidadas como YOLO, ByteTrack y Roboflow Sports o Eagle5. El volcado de las trayectorias normalizadas a Blender mediante la biblioteca estándar de Python permite erigir gemelos digitales tácticos con precisión métrica, prescindiendo por completo de dependencias propietarias o suscripciones a servicios en la nube12.

#### **Works cited**

> 1. GPT-6 Astra와 Higgsfield로 구현한 실시간 축구 분석 툴 화제 \- promppy, [https://www.promppy.com/item/1512892](https://www.promppy.com/item/1512892)  
> 2. Trove, [https://trove.sarmadsaleem.com/](https://trove.sarmadsaleem.com/)  
> 3. GPT-6 Astra: A new generation of intelligence \- OpenAI, [https://openai.com/index/gpt-6-astra/](https://openai.com/index/gpt-6-astra/)  
> 4. Higgsfield MCP on GPT-6 Astra: Introducing Higgsfield Games 2.0, [https://higgsfield.ai/blog/higgsfield-mcp-gpt6-astra-games-2](https://higgsfield.ai/blog/higgsfield-mcp-gpt6-astra-games-2)  
> 5. roboflow/sports: computer vision and sports \- GitHub, [https://github.com/roboflow/sports](https://github.com/roboflow/sports)  
> 6. nreHieW/Eagle: Convert broadcast data to tracking data ... \- GitHub, [https://github.com/nreHieW/Eagle](https://github.com/nreHieW/Eagle)  
> 7. [https://golero.pl/ciekawostki/ai-oglada-mecz-i-analizuje-go-na-zywo-to-moze-zmienic-futbol/](https://golero.pl/ciekawostki/ai-oglada-mecz-i-analizuje-go-na-zywo-to-moze-zmienic-futbol/)  
> 8. OpenAI Inc. (via Public) / Introducing ChatGPT Images 2.5, [https://www.publicnow.com/view/C16EA39F86D1DF509396A27112C0187FA41A1B60](https://www.publicnow.com/view/C16EA39F86D1DF509396A27112C0187FA41A1B60)  
> 9. Football players tracking with YOLOv8 and ByteTrack \- GitHub, [https://github.com/Darkmyter/Football-Players-Tracking](https://github.com/Darkmyter/Football-Players-Tracking)  
> 10. Tools for Sports Field and Player Detection with Computer Vision, [https://www.blog.brightcoding.dev/2025/07/15/tools-for-sports-field-and-player-detection-with-computer-vision](https://www.blog.brightcoding.dev/2025/07/15/tools-for-sports-field-and-player-detection-with-computer-vision)  
> 11. Enhancing Soccer Camera Calibration Through Keypoint Exploitation, [https://arxiv.org/html/2410.07401v1](https://arxiv.org/html/2410.07401v1)  
> 12. Importing CSV Data Using Blender Python \- YouTube, [https://www.youtube.com/watch?v=utNayZsIleA](https://www.youtube.com/watch?v=utNayZsIleA)  
> 13. Machine Learning & Computer Vision, A Better Match Tactician, [https://medium.com/@johncomonitski/machine-learning-computer-vision-a-better-match-tactician-than-pep-guardiola-1b44924968e9](https://medium.com/@johncomonitski/machine-learning-computer-vision-a-better-match-tactician-than-pep-guardiola-1b44924968e9)  
> 14. Automate Blender Animation with Python and CSV Data, [https://blog.cg-wire.com/blender-programmatic-animation/](https://blog.cg-wire.com/blender-programmatic-animation/)  
> 15. GitHub \- mserra0/FootballVision: A project that utilizes deep learning, [https://github.com/mserra0/FootballVision](https://github.com/mserra0/FootballVision)  
> 16. soccer data analysis and visualization using machine \- CHAPTER I, [https://scholars.csus.edu/view/pdfCoverPage?instCode=01CALS\_USL\&filePid=13295166840001671\&download=true](https://scholars.csus.edu/view/pdfCoverPage?instCode=01CALS_USL&filePid=13295166840001671&download=true)  
> 17. Calibrating Football Fields with Keypoints (Part 1/2) \- Medium, [https://medium.com/@akbarnezhad1380/calibrating-football-fields-with-keypoints-part-1-3-88fa4aad4d6e](https://medium.com/@akbarnezhad1380/calibrating-football-fields-with-keypoints-part-1-3-88fa4aad4d6e)  
> 18. Ball Tracking in Sports with Computer Vision \- Roboflow Blog, [https://blog.roboflow.com/tracking-ball-sports-computer-vision/](https://blog.roboflow.com/tracking-ball-sports-computer-vision/)  
> 19. player-tracking · GitHub Topics, [https://github.com/topics/player-tracking?l=python\&o=desc\&s=stars](https://github.com/topics/player-tracking?l=python&o=desc&s=stars)  
> 20. football-ai.ipynb \- Google Colab, [https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/football-ai.ipynb](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/football-ai.ipynb)  
> 21. Camera Calibration in Sports with Keypoints \- Roboflow Blog, [https://blog.roboflow.com/camera-calibration-sports-computer-vision/](https://blog.roboflow.com/camera-calibration-sports-computer-vision/)  
> 22. Simon9/football-field-detection-roboflow \- Hugging Face, [https://huggingface.co/Simon9/football-field-detection-roboflow](https://huggingface.co/Simon9/football-field-detection-roboflow)  
> 23. wunderscout \- PyPI, [https://pypi.org/project/wunderscout/0.1.17/](https://pypi.org/project/wunderscout/0.1.17/)  
> 24. Top-1 solution of SoccerNet Camera Calibration Challenge 2023, [https://nikolasent.github.io/deeplearning/computervision/calibration/competitions/2023/06/20/SoccerNet-Camera-Calibration-2023.html](https://nikolasent.github.io/deeplearning/computervision/calibration/competitions/2023/06/20/SoccerNet-Camera-Calibration-2023.html)  
> 25. MM4SPA/tvcalib \- GitHub, [https://github.com/mm4spa/tvcalib](https://github.com/mm4spa/tvcalib)  
> 26. No Bells, Just Whistles: Sports Field Registration by Leveraging, [https://github.com/mguti97/No-Bells-Just-Whistles](https://github.com/mguti97/No-Bells-Just-Whistles)  
> 27. MichiganDataScienceTeam/F24-SoccerNet \- GitHub, [https://github.com/MichiganDataScienceTeam/F24-SoccerNet](https://github.com/MichiganDataScienceTeam/F24-SoccerNet)  
> 28. How to export object data such as coordinates to csv? : r/blender, [https://www.reddit.com/r/blender/comments/azvds9/how\_to\_export\_object\_data\_such\_as\_coordinates\_to/](https://www.reddit.com/r/blender/comments/azvds9/how_to_export_object_data_such_as_coordinates_to/)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAF8AAAAVCAYAAAAgjzL/AAACXUlEQVR4Xu2YTYhOURjHH6GmKB8jspg0NSkrJlkosrGxoAnTbKeUj1gwQ8rGxs5GKBspycLHwgIlxbsahZLF1JSUZIOFlYWF+P86z9Pc9869lxo1bu/51a+m+9n8n3uec85rlslkMpleZZ08L0+WT4hFcsx9LB/Im3KFG2z08/hJfrB0D/djwHUP3VuWrt9WON8T9Mkz7kU5I892XZHYJ6fcfj92Ql5yCXatvCHXuHBAfrd0PwLFeip3ujAkX8hBt+dYLjs2N3wK9EjeduML3iM/ugS2Xf6QEy6skq8sjRKE3fKdXO9CvPuIW8dSOVA+WGK12yrqwo/jxQBhq/zs7pDL5ClLLQWhKnye37H0XIQl8r687tZB4RlxI26RDe41626FrSCHv4DUhV/XdvbKXy5/V0FP/2bdPf+czQ0fokDFAlexWE66+/1YhN7K4KEufCC4Ny6rInovX2hT+ITwxFJIFCyKdtnmFz5QgCjCaZsNvZXBQ1P4BMcEix15Tx6XX9ziMpHC4BU5bt1LTKhqO1zDqPrb8APa21s5Wj7RNprCJ8xYRUSYDPlpN5aW0ZOjLwNf42EXuO+9pRGEEO+mJeGfYGmKfPHMK+xNeG60odaRw19AmsKfkF9dhjrFuCOPukDwXHfVPejSkwknds6D8rXc5AJFeGlpr4BNROjR54F3xztaVYBD7jP509LqhGBxi1/Dxui5O2ZpsmVHHP0dWHqym41JuGj5qzwm77qbLc0PLFMJsTxHFFkpL1j15Br3MsKGS+daT/zDu2y2XcwHfqbAf/W8TCaTyWT+S34DeLeaptrpjhsAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGoAAAAXCAYAAADjndqIAAAEK0lEQVR4Xu2ZW4hWVRTHl5iXMq9JYzc1zSQSTbqMZtgMXrCHQqUHQfNJTdQMzKy8NmWQSiRdREVRA19iUEPDeRAKA7FeKvCG0IuvvvWc5vqx15qzZ8/5znzjN/B9A+cPP+bM3mfOOnuvtdfaZ49IqX6t4cpAo1T9NMDAH7kqHdUYKnTUYmWDZDeVqr/eTBueUU4qD6cdpeqqB/zCV8/XyqLO7lINp6eNDqUp6SvVQCod1U/0hvGjRPkw0gjlfeUL4ykJm45vlGeNvtA4Axs8P97MjFVeSdpq0VRlr7LG4PmfGMOMWsW7Mg5gTIzN24HxYLdqfWQw8anYWLRJMNJq3FbeUY4ph41aNVL5zHhN+UuZEPUzUOz1haNmSAi8Icoe428J9fmaZIFbq3DQW8YB5VNrn2Rg60Vrq0q+UnBWqrlKi13PN/5UxisfWz/Uqlcle/4y5ZIE53l0X1De7by7Nq1THrVrgtOZrOxWHjNq0WDlPeVJg/EwLuSBcEV6uaJKR/UTRx0x8hwVyx1aKQXRRu5/3eitqI/QLtm7eJq4obwsmQ23U4tGK38YpKdUg5TlEtIxaRHyxl0k0jhcVZ6wNp9Hvll53mxll7FeCuqj16g8R3GMxBHGUOVnwyObgRA5gJqVzVL5WT3Jow+nMDjkOd6jz224nfsRY4HnJEwgcI0ekuz4bImEOvOgcsZYYPdVq60GwUcQUvN/NZhHdtvbJcwlcCq0XyoEBH8AJ9IOCQXwpoT094/hkY33KYZxQeS4o5Kj2B3+osxJO0wUeWAjgcOw4cXeow9hw+2kYlIPKtuMeMCkUrhs9zDm3wyCkT4inckEnMKOEPkq2GS/81zABs/Cbp48W+EsNEXCfALzSIr9UjKbBOhZqbCqWHpwXrofAH4gIdXh5eMG1zuUVZK9sKvIUbOUf5VDaYfJX/a0sk/CTuk/I65PRY5isi9KWJXQFPX5809JqEWMo8NgF/it8kLn3ZkekTB58Ly1eZr+QcIumADLk7/r78pGCbtLX8VxffJ5bFPWRu1dVDqqnzjKU8I5ZVrSh+gjf7pGScjxeSpyFGLiPH2kmmg8LmFySA0+4fF7FTnK9aGRt6ui/rCRYGK8HmEv7986jJugxIF5TkQrJatxsUiH0w1SGe9C8H1npAEOq5P2XK1QdqaNvVRPjppnpMKBXgPJ+0wQuf8rI29QlWzwLFYI9DjoAvG3b0vYdbKdh9YudwRnbLGfqQjIWwafM5xEsKJwauxYVuNCA5tLJdug5YrJoXDPTDuqVLPyvfKT0SJdj6TGSPi2iFeni4EeNSiu7RIGyr3x/W7D7bRI92MvopJTjfhk437EbvOO8r9y1+A7Lxb3vJS0ubjX5+JzCRmLDVUsNk3XJdhw0hOi3HGQ5ogQ0huUqp/80yc3c5SOahwVOqpUA+oeTuvnGzUAGpYAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAWCAYAAADXYyzPAAABe0lEQVR4XuXWvytFYRgH8Ed+RJQfyc0gJQZKKROD5VIsSlgoq+IPMJiNShYWyUSZbbqJ0SDFaGBgsBkkpXy/nefNc9573nvu4DiDb32W+7z3vufc53lPR+Q/p14VoNvTYtYxtdCltSRtqqrktnGHWoAbeFArMGjWMa0wB9dwp5Yk+u4a3KoTXVtVOuEetlQo7RJtnLSOm9El7EBNvJycUXiDWRUK/4VXmFFJ2ZDoJngzqVmFF+hToSxL+jpu/ChR31Pz5xuzD3QI59CsQtmFM2hUNva3OKA8KcGwD3awKsUOVlLsbx1DXbwcD4fKDVZoWFw4WM8w6Rc00+oDprxaEUbsB+yt7W+l2P764TFiq2hPyu92SMyUu37QhZQ/qVxs71x/bfg0O4V91WRq/WpczLnOZWM+JErwpd7hAHqUC/vOdW7tJzx5rmBCfi7QpRfm1ZGkTPlvpgHGFE+CvajMs6kWYdirZRZO9rZah4F4ObvktjHDlwfiy0Z++QbUY1OpgHNEigAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAXCAYAAACxvufDAAACoElEQVR4XuWXy8uNURSHl1xyv0Q+QnJLopBbiqIMGLhEuWfgkokRIZREBkJi8H1CRJmIGaEMDEQMjJj4F/wT1nP2b52z33NzzvedgeP86um8Z7377LX3Wutd7z5mPawJznDRjWpp/S0N+of11/Xvck45w0Q3arVzUYysumcLnefO+OobXaiz4nAYImt3nK1h7HLNEm/CME+8c/rC2OWKxD0OQ09scpt44YwIo8RAmtF1Z4YI+1pnmuiEoiPusVp/2539um5HJ+PivLhXuVfWPme30+9cEWi+89NZLjohfOX+Yl0E/pnz0VJTbKcxkrySiBowYWisoEPNdj5bimREkx9/c6aIoSp8wRxLc+f+COpta//VtiouemKTj0S+yVwbLJVmtGVEUJ5Y0SkTvhdrZBuMtji/rOiPuQ9YKl045ryydLpppvImo/YbbZKTw0urOBjnfLD0UHMNnCzI6FMxlGZEAHN/6KCzrDwiBf5G9r2RysFmscDi6okss9HQIkuRXudcEGyKBvRQEOHNsofI+CXnvjNG1BP+bmbfpztnrNj5TzuHnJnOJlFzhLPUlUtaL95a/fQz8KulEoEB55OlDnhUIJxGReyw1K3zE1R0yd/WvCuTpe9WmZuTGH0BRXZZA6+anZZefdCnMbnKyemJTU4Sr61Y97lGO1MFZcdLm2cwbzwPnC+CUh6V3ct1xFkiGonSi4NG/peJjcAPS/4oZfxArngc+MNRECf2y9XGFhVNZ4HgmkzR+nPh+Jw+Gz2TzUSWo+nQVMjyUjExG7dR3M1sJRE9fryy+kYLIiu3rLL4q85xq+2ylDH/9warE4J5FjvXnL0iqoaqpJxhrmwFMYBITxatipLKu1uUdKeFD4i5+cwfGcSbYoWoq57Y5H+nP2X/e0+wCn57AAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAXCAYAAABDArJmAAABU0lEQVR4Xu3XTytFQRjH8Ucoukokf7IQb8HCxkoWLFhgp9TNRl6FjQ1lYSniFfAWrJQsvAJLWVgoSyV+v+YZZ5osZnZ35jzf+nS755zVNPPce0Qsy+qx+mESZiIjMAhTKrw3pFrXOOzAM7yoPZiDWeiqV3jQZydUKxuDJzhWYQvqDXaje61sET5hU4X5a7zP51LrgxV1C5dwAYdqA0b/ni4oW6yMeLx4zPyRCztXPKY8rqlxMQ4Uf0SqiDvgGt71k7gD6ErcIhKv89mU5uEIBlQ1hcM9zg/23OG+BqvxxRqyxcooHO5xfrDnDncu1p00x/k/p+L+DBdVONzjuNu463KH+zJsxxdLjsPaD/d7ca83YfzO637opw53Ngxn4t4CqOjW4VF9wxfcSPMeuC9uN/3Ah+Kx6Uh609IcuS0p+F3SFqsH459RWoITqWDAW5ZlWZX0C+Q1WQdCQwPCAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAXCAYAAACrggdNAAACZUlEQVR4XuWXu2tUQRTGR3w0KvFFfCDEiAiChUEQNbEyhRaSkCaBWCo2CiKoiCJpLFKJNkFJY0B8ENBCiYVgK6YO+QcigoVBwUJF9Pt2vnOdnb27d5bdoLn+4Mcmc+bO3nN35sxc5/4j1sOV8l9nhfP3y09al9IlNQjPu9pOTLBTbpdb4epA/m8xuqFyZTF519rYIRwv7LPZ+Xs8Bc/KmsT2wIdwXRwAm+CwnIM/4E21053wFfwIx+W+ypXF2Ngcl36FZ+DasBMYgJ/hlDyq9lXwjjymtuxXuQ1PWGMDRuBPV923C07CbUFbs5yTfGB9UYz3dxEORe3GQfnEGrolnzR/9iLYZx4+cn4K0AnnE2uF3fIDvBvFmAyTqplegrOLvrSGUiZ1Uj51fn6mMAa/wWcydf00gt9N+bC4trbAQ5JrNC4ceWQP46qMn04j9sNFeEu2E1uzN+B92VHVoz6X7A+7MSaWAqcAyz4r3VuZ+qUpcCkswNfOj9vM2CzvFUqZFEsxLUrKSr+VVpsmcXlvFe55TGo0DiSQJWVrqigpJhJWIauCVglTi0wR/fCT8/tOs2T7mG16D/7EqmACw/CKDM+EY5JFg8Ujj73yDeyNYnlwKczCjXEggeyHOSJnnD8cEiuv1+F7+Au+kzvUh+dExizOz2twjTQOyy/wXtAecgA+l9+dn9KP4S5ZhC0NHp8qlDIpqzAvXP0p1A64BrN9pM1wbDodB047f/JeKo7LpYD3Ti/HAR5DeBzpiQMtwlcLesGlHXWahWfOCZm7p7GRFY4vZKkveX8Trl1WPO5tNJdSJrXs+Q0Y8ISDeGRtuAAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAAA80lEQVR4Xu3SLYsCURTG8SOLoKzFBZVFxJe22WQ0icWiQdiyZb+EX8EgiFX0QxgMBpMgBqPFIggmy4YNyuI+h/tcZlDBGZgizB9+OMyVy8y5I/IsvUAa3q8kuB6ljGstRjcFutkbtGANO/qCPNdTtIAtdCBLd0vCCvpki0CbvsW8xcPK8AMN0l6hC1Xy3Ccc4YMKMOSv7wLbTOcygrk4r6mHsBFzgr6ywz+IM+wanHntKzv8puteHKak18pTOi99qtLVfX2qX6iQpwLZTAdvhz8T8125K8IeBqT/vVsdlvQHJxiLOT2lM+px7UITyElYWJirf+7gMGi5BSbPAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE4AAAAbCAYAAADS6blZAAAC9ElEQVR4Xu2YS6hNYRTHlzzyzDtEueSRUpJ3GNzII5EYEDIgmZAi75KJwkSUZ+RRkseQroGSkrySCQOZGBgZKGXM/99a657dZ5999uOenFPfr36dffZ3zn6ss761vn1EIj3KENjbbFd6id4H5XbIAHgDvoBvzQ/wGvwIl5uFiIErEbj1cI/oydJO2E6sNXfJv/cyHa6w7d3mKXvfCeeZuZgC78LB4UCLwmB0iGbJGjNJH/M8XBqMzYQjRcfvmZ5hk2yMZuLZdQ6uCsZakWHmafOn1LIrjTnwvuj0DBkFu8wJwVhDGGH6FI4JxlqdcfCbZAeOM+gJXBQOgCXwodk/GGtIDFzJwK02H4jO+SScwmwYLJxjTd8/XzTVaV44HXgsHtNLxGh4RDQIRckTOHIB7g93Sq0peGMoxGGTBw/ZBDfAS/CkSSbDT3CW2QgvtmdFf+X3ojdNF8Lfkn3j9cgbOAbteuL9CLgRvoMXzQWJ8Vx4xBk8Z6B5UDRLXsHNJmGGcu0z3GxEp7kSboWv4VCTHJUC7T9B3sBx7Ha4syoxcCVhCtNk4JJwanFajjcJA31Tii+UWYBZqJPn4vcZuCK10vmvgfMaVy9wx+AjqS0oB8FnooWV27Rv96ezYW38KvpjOOzk+6TYD+DkDRzrdI8Hzh856h2Y2cjgOVPhF9FiykzxbJkGn8PFZhpsJJ9Fj+FsgbNt2zP4OLwiumhNW7g6eQPHpCjVObPg+oZ2iT4Uh/CC3sCd5mX4UrTT7jAJu+MveNVMg0G4JdrBD5j8vmebZ/Ud+EPSuzanO2X9ZVf8A7+bDHZH9ycVHpvHWxfsr0wMXEm8uz0WffhNgxfqazFeCP9yYjf1qeWwXnHNlLbYTMJnTa7os/5Q2A5nmFXgNbFG87UpbIMnwp0FWZawCszMQ/aaVePywPtidjYNdsYzUivUReFqfK/ocfJ22XpwWs0NdxZkosnS4uvFphADVwGegFPE//NqV/pJbW3KJ59IJBKJRJrLX1U3nm/bK1NGAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFsAAAAbCAYAAAATbvP1AAADS0lEQVR4Xu2Y3YsOURjAH7EiZH1kfZVoI7WFfIcLkbhQoqyQGxduKCnkIzblQi5ESYRysclHUWjlgnUnSm7EH0DKjbiTfDy/9zxn35nZd8zMOzPIzq9+7b5zznyc55x5zpkjUvHXGaUONgcSce0ebl5Rn6rP1ZfmJfWVurqvdkbibvq/E9fu0oK9Qd2tDjI9PMAEc5LZprYE5Lcvw9bamcmMlPB5OE4a3z9Yh/OKZIF62KQ9nlnmGvu9Sz1pwkp1of2fmna1Wxo3Yqzaab5Wv6nH7DhOVR+qH9VT5uzamclQb7/61XyhrpBwsBlZp9XP5lF1SqA8DQwGPK7ujZR5eA7cHjjWYTIAhqjXxY1kP5qnW1kq/Cg+o66NlDVii/pdwnWnqZfViYFjWRimPjAJ9phwsYxWL4i7D2aBaxNAOgvfqgdDNerQgchz0DFRxqs94gYXZoaeQUZmoxtEoc4bcT1Mj6IPRB62mT+kPmp8iuJNWWTHmoU3FnslPth+4F0VN6iiLFdvietAzEwV7DqlB3udeVNcTkpDl7j8esdMm59/h+/0d+o5cY0mt+LGQL1mSRNsD5MgaTGKnxybhhsjDUwLE8YnCc/KeaGj/QREXj0hbmXkV0d5yRJsBt8jdYTUFwGbxM0n59XFZmZ8wJIewEPDCQArj2cmE1hR8PqSSg5I/yVoHrIEe756VxqvzHJRBbs/pQWb3IRJD+Ab7nOoXwJGl4F54dof1BnRgpz8E8H2OTvpAQgCEmyC3iZuVYLk2bSTaxLMHb1SfEOzBJsvwntS/DPUZli8Fi0wCGynuNcag3sHXSaTZUfgeJCZ5hN1WaQsCB8yyCSUZdLlmsj1uU8cWYK9Xr0txQ2gPpaaPeI2Y8CvDI6o79Wf4jZgcLLVYR+FMl/O30PqUNOzxPyiXgwcD7JTXJCRazEfcO8069nNJuks7jOc6z82qcfguKHONaOwP5LUIU1RBbs/pQWblQTel/hUUATk+H3RgwUyRxp/9WXBb6d2S7mxqO10sZNXFqvMstgq+QPEbiOelRLydZAWcXsQ86IFOfFfYHskvE9cFO3mDsm3Jvc7i5h3nyeRKth/MNjADVnetZoDCZa/cRNmRUVFRUVFRRn8AqkLwwo2bzlOAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHkAAAAZCAYAAAAVDoETAAADzklEQVR4Xu2YWahOURTHl1xlliHX7BpLCWVKiUxFSLlE4UGKklAylTEp8mAornkqJFIy5kGivHgQERkeSIoXpSjKsP7t/3L2PZ39fYe693Pv3f/61dc+Z59vr/Xfe699jkhUVFRUVFRUVFRJVK5sVlamL6gaKXPILeWcclRpR0wHlMvksHJMua30J3VdzZRNyjXlLFksLj9QJ3JIOa7cUGYTu6fW1VRZTXYpz5W11e5wGiXOLNCebfMlCbSMbQjsG/muXJX6YW4TclBZI86wDeSluAWCHOwl/Vy3P33AJLbVuqLJ+VSnTfbVUrkj2SbvU04R01BxAYIBbNumdCalFhIPWqQvUDCqghTbSieTR+KeCY0kC8WZifZ7xEyGkE+wxGsrmUIm++3+td7KezKFbTC5B0HQWPXFEthV2arsIOi7Qlx9H0Oai6t9qPOVpDE6F1BPggnaJnUNY8LZYyYpJNx7gqAWD1RmKYOJxYddEdfBW2W8uPjPEOQrSxa/5cCP33Lgx285KBZ/pkImtxI3O9MmY7W+IdPZhgHYisdKx8CxhWM1hVYUkjNX+UC2i7sXyXxNdos70HRRHpOJ6JxDfZUqcUYD3+A8sryAF8o0pbVyhNj2DWFHA8+UX+LiGUFCsvgtB378lgM/fstB3virKWRyB+Up24uZjJntG4q+T5RFJCRMiHdkONv855shfsLT4ywkGG21cYvkNxjy//OB0pbtSDLATmblCqsXXBC38mE0VjUYwnuyhPgtB378lgM//pBPuRTqHE1uACb77f61bpIYA5OxZSEBqBVWL6yvbeEhIUBLBn5DWZPoX03GeDYSvOena3Qh+f+J2mpbsxnzRdz4eikXCWoxNEjc1g3wbaGM7WnZs9LxWw78+EM+5VKoM4JCcPj4AUzp1Ye681XcKdJOkv+DyTB4lSSHLBzGquTvjLbYs0z+LO7giTFmxTmaXBc3/iyV3GQIh4IrBAcFCK8UdwkShi3rpNKRQDg54uBgB4uQasJk21HMYF++0XnMtleo+5Js1zZp8ArZXdy3BFvJ+DJmGkfwHcImSFo1brLVS3zo+KF8Us4TqyN48GmyXtz74U1lGIEQwDJxr1EApuLz5k5JPiZkCf/xUPlJ8BvJw6sIahp4pSwQd1LHGMFHZamEVUlmpC9QMHodCW2jJhv/fmWPMk+SyTaV9+AZ9rxL4uJfLonxKG9ZsvgtB378lgM/fstBsfirKZrcAEzOK9v++oirMbZtp1VOxkpyAKkvwkSukCT+UA4Q9wRxubK8RUVFRUVFRUVFRUXVaf0GdX5VbGfaCbAAAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,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>

[image12]: <data:image/png;base64,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>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAAA30lEQVR4Xu3TwQoBURQG4CMpxYZEslBewVPYWNkpJc9iZWepxKPMSsnCE1jKwkJZKvH/zbk5XcNomo2av76mufd275kzXZF/SR7q0PSUoQANZeeK6i2pblaFAezhoEbQhhaM1RE2uramIlOBHUyVTUedYOjNRaYLV+grGzfGea6LDU/kya4Km7li5fyC2KS2WQ5WcNYnLdRSwkOI41z7Nbb5flzjEzXfj2t8oub7YbWs+qd+Malsxma65gcSXh8bvnPc/ZSPze/BVt3hBmt53cOJhNU84KJmUJIsWbKYPAGarjj+SKqvAgAAAABJRU5ErkJggg==>