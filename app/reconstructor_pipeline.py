from typing import List
import os

def detect_objects(image_path: str) -> List[int]:
    """
    Analizează imaginea 2D și returnează o listă de ID-uri pentru fiecare obiect
    detectat (de ex. [12, 34, 56]).
    În implementarea reală vei folosi un model de segmentare/obiecte (YOLO, Mask R-CNN etc.).
    """
    # TODO: înlocuiește cu inferența ta
    # ex. masks = model.detect(image_path)
    # return [mask.id for mask in masks]
    print("Detecting objects in", image_path)
    return [0]  # stub 

def build_mesh(image_path: str, object_id: int) -> str:
    """
    Construiește mesh-ul 3D pentru obiectul dat.
    - image_path: calea PNG original
    - object_id: index-ul/ID-ul obiectului din detect_objects
    Returnează calea fișierului .glb generat (ex. "/data/scene1_obj0.glb").
    """
    # TODO: apelează funcție de reconstrucție 3D (Instant-NGP, Point-E, PyTorch3D etc.)
    out_path = os.path.splitext(image_path)[0] + f"_obj{object_id}.glb"
    print(f"Building mesh for object {object_id} from {image_path}, saving to {out_path}")
    # ... rulează modelul și salvează la out_path ...
    return out_path

def merge_meshes(mesh_paths: List[str]) -> str:
    """
    Primește lista de fișiere .glb generate și le combină într-o singură scenă.
    Returnează calea fișierului final .glb (ex. "/data/scene1_combined.glb").
    """
    # TODO: folosește trimesh, pygltflib sau alt tool pentru a uni mesh-urile
    print("Merging meshes:", mesh_paths)
    final = mesh_paths[0]  # stub: doar primul
    return final
