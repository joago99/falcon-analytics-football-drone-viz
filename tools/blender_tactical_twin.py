"""Gemelo digital táctico: importa trayectorias CSV a Blender (bpy).

Fase 3 de docs/analisis-tactico-cv.md: el CSV exportado por el pipeline
(frame, id_jugador, equipo, campo_x_metros, campo_y_metros) se convierte
en una escena 3D con el terreno FIFA 105x68, cilindros por jugador y
keyframes de traslación fotograma a fotograma.

Uso dentro de Blender (Editor de Texto -> Run Script):
  1. Abrir Blender, borrar la escena default.
  2. Copiar este archivo al editor de texto y editar RUTA_CSV.
  3. Ejecutar. Los jugadores se animan automáticamente.

Requisitos: Blender 3.x+ (Python embebido, bpy).
"""
import bpy
import csv
import os

RUTA_CSV = "//DJI_20260829135649_0067_D_work_trayectorias.csv"  # ajustar
COLOR_EQUIPO_A = (0.2, 0.8, 0.2, 1.0)   # verde
COLOR_EQUIPO_B = (0.8, 0.2, 0.2, 1.0)   # rojo
COLOR_SIN_EQUIPO = (0.6, 0.6, 0.6, 1.0)  # gris

ruta_abs = bpy.path.abspath(RUTA_CSV)
if not os.path.exists(ruta_abs):
    raise FileNotFoundError(f"CSV no encontrado: {ruta_abs}")

# --- Limpiar escena default ---
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# --- Terreno FIFA 105x68 m ---
bpy.ops.mesh.primitive_plane_add(size=1, location=(52.5, 34.0, 0.0))
campo = bpy.context.active_object
campo.name = "Plano_Terreno"
campo.scale = (105.0, 68.0, 1.0)

# --- Materiales por equipo ---
def crear_material(nombre, color):
    mat = bpy.data.materials.new(nombre)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    return mat

mat_a = crear_material("Equipo_A", COLOR_EQUIPO_A)
mat_b = crear_material("Equipo_B", COLOR_EQUIPO_B)
mat_x = crear_material("Sin_Equipo", COLOR_SIN_EQUIPO)

# --- Colección de jugadores ---
coleccion = bpy.data.collections.new("Rastreo_Tactico")
bpy.context.scene.collection.children.link(coleccion)

instancias = {}
with open(ruta_abs, newline="", encoding="utf-8") as f:
    lector = csv.DictReader(f)
    for fila in lector:
        if not fila["campo_x_metros"] or fila["campo_x_metros"] == "nan":
            continue
        num_frame = int(fila["frame"])
        pid = str(fila["id_jugador"])
        x = float(fila["campo_x_metros"])
        y = float(fila["campo_y_metros"])
        equipo = fila.get("equipo", "?")

        if pid not in instancias:
            bpy.ops.mesh.primitive_cylinder_add(radius=0.45, depth=1.8,
                                                location=(x, y, 0.9))
            jugador = bpy.context.active_object
            jugador.name = f"Jugador_{pid}"
            jugador.data.name = f"Malla_Jugador_{pid}"
            mat = mat_a if equipo == "A" else (mat_b if equipo == "B" else mat_x)
            jugador.data.materials.append(mat)
            coleccion.objects.link(jugador)
            bpy.context.scene.collection.objects.unlink(jugador)
            instancias[pid] = jugador

        nodo = instancias[pid]
        nodo.location = (x, y, 0.9)
        nodo.keyframe_insert(data_path="location", frame=num_frame)

print(f"Gemelo táctico creado: {len(instancias)} jugadores, "
      f"{sum(1 for _ in open(ruta_abs))-1} registros animados")
