import os
import subprocess
import time
from datetime import datetime


def process_data(pid):
    return [execute_command(f"pidstat -u -h -p {pid} -T ALL -r 1 1 | sed -n '4p'")]


def process_threads(pid):
    return execute_command(f"cat /proc/{pid}/status | grep Threads | awk '{{print $2}}'")


def process_swap(pid):
    return execute_command(f"cat /proc/{pid}/status | grep Swap | awk '{{print $2}}'")


def current_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def write_to_file(filename, header, content):
    """
    Write header and content to a file.

    :param filename: The name of the file to write to.
    :type filename: str
    :param header: The header to write at the beginning of the file.
    :type header: str
    :param content: The content to append to the file.
    :type content: str
    :return: None
    """
    with open(filename, "a+") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        if file_size == 0:
            file.write(f"{header}\n")
        file.write(f"{content}\n")


def execute_command(command, informative=False, continue_if_error=False, error_informative=True) -> str:
    """
    Execute the given command.

    :param command: The command to be executed.
    :param informative: Optional. If True, the output of the command will be printed. Default is False.
    :param continue_if_error: Optional. If True, the program will continue execution even if the command returns an error code. Default is False.
    :param error_informative: Optional. If True, the error message will be printed if the command returns an error code. Default is True.
    :return: The output of the command as a string.

    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return_code = process.wait()

    if return_code != 0:
        if error_informative:
            print(f'\nERROR {current_time()}: {error.decode("utf-8").strip()}\nCOMMAND: ${command}')
        if not continue_if_error:
            exit(return_code)
    else:
        if informative:
            print(output.decode("utf-8").strip())

        return output.decode("utf-8").strip().replace("\n", "")


def get_time(command) -> int:
    """
    Get the execution time of a command.

    :param command: The command to be executed.
    :return: The execution time of the command in nanoseconds.
    """
    start_time = time.perf_counter_ns()
    execute_command(command)
    end_time = time.perf_counter_ns()
    return end_time - start_time


def detect_used_software() -> str:
    docker = execute_command("which docker", continue_if_error=True, error_informative=False)
    podman = execute_command("which podman", continue_if_error=True, error_informative=False)

    if docker is not None and podman is not None:
        print("Both docker and podman are installed. Please, uninstall one of them.")
        exit(1)

    if docker is None and podman is None:
        print("Neither docker nor podman is installed. Please, install one of them.")
        exit(1)

    container = "docker"
    if docker is not None:
        container = "docker"

    if podman is not None:
        container = "podman"

    return container


def check_environment() -> str:
    system = (execute_command("lsb_release -i", continue_if_error=True, error_informative=False)
              .replace("No LSB modules are available.", "")
              .replace("Distributor ID:", "").strip())

    version = (execute_command("lsb_release -r", continue_if_error=True, error_informative=False)
               .replace("No LSB modules are available.", "")
               .replace("Release:", "").strip())

    software = detect_used_software()

    software_version = execute_command(f"{software} -v", continue_if_error=False, error_informative=True)

    message = f""""
Software={software}
Software Version={software_version}
OS={system}
OS Version={version}
    """
    return message
