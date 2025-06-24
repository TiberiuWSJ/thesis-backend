import app.models.user    # registers User
import app.models.scene   # registers Scene

from app.celery_app import celery_app
from app.database import engine
from sqlmodel import Session
from app.models.scene import Scene, SceneStatus
from app.reconstructor_pipeline import detect_objects, build_mesh, merge_meshes

@celery_app.task(bind=True)
def reconstruct_scene(self, scene_id: int):
    # 1) Load the scene and mark IN_PROGRESS, then grab input_path
    with Session(engine) as session:
        scene = session.get(Scene, scene_id)
        scene.status = SceneStatus.IN_PROGRESS
        session.add(scene)
        session.commit()
        # extract what you'll need later *before* the session closes
        input_path: str = scene.input_path

    # 2) Run your ML pipeline
    object_ids = detect_objects(input_path)  # uses the local variable

    total = len(object_ids)
    for idx, oid in enumerate(object_ids, start=1):
        mesh_path = build_mesh(input_path, oid)
        # update progress in a fresh session
        with Session(engine) as session:
            sc = session.get(Scene, scene_id)
            sc.progress = idx / total
            session.add(sc)
            session.commit()

    # 3) Merge meshes
    result_path = merge_meshes([build_mesh(input_path, oid) for oid in object_ids])

    # 4) Mark COMPLETE
    with Session(engine) as session:
        sc = session.get(Scene, scene_id)
        sc.status = SceneStatus.COMPLETED
        sc.progress = 1.0
        sc.result_path = result_path
        session.add(sc)
        session.commit()

    return result_path
