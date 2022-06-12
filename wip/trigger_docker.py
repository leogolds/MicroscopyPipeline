import docker
from pathlib import Path

from torch import detach

stack_path = Path(r"3pos\pos35\C2_enhanced_short.h5")
assert stack_path.exists()
model_path = Path(r"3pos\pos35\MyProject_pos35_red.ilp")
assert model_path.exists()

client = docker.client.DockerClient()

command = [
    f"--project=/data/{model_path.name}",
    # '--export_source="Probabilities"',
    f"/data/{stack_path.name}",
]

volumes = {
    # f"{model_path.parent.absolute()}": {"bind": "/model/", "mode": "ro"},
    f"{stack_path.parent.absolute()}": {"bind": "/data/", "mode": "rw"},
}
# print(volumes)
# exit()

container = client.containers.run(
    image="ilastik-container", detach=True, command=" ".join(command), volumes=volumes
)

for line in container.logs(stream=True):
    print(line.decode("utf-8"))

# container = client.containers.run("alpine", "echo hello world", detach=True)
# for i in container.logs(stream=True):
#     print(i)
