from src.utils import execute_command, write_to_file, current_time, detect_used_software, check_environment

class Setup:
    def __init__(
            self,
            containers: list,
            scripts_folder: str,
            software: str
    ):
        self.containers = containers
        self.path = scripts_folder
        self.software = software

    def build_images(self):
        for container in self.containers:
            print(f"Building image {container['name']}:{container['version']}")
            execute_command(f"{self.software} pull docker.io/{container['name']}:{container['version']}")
            execute_command(f"{self.software} save {container['name']}:{container['version']} > {self.path}/{container['name']}.tar")
            execute_command(f"{self.software} save {container['name']}:{container['version']} > {self.path}/temp_{container['name']}.tar")
            execute_command(f"{self.software} rmi {container['name']}:{container['version']}")
            print(f"Done!")
