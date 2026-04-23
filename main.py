import mujoco
import mujoco.viewer


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(
        "dobot_mujoco/env/assets/dobot_table_scene.xml"
    )
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
