"""Testes unitarios de partes que o golden master nao exercita."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from src import lifecycle, stressload, utils
from src.config import load_config
from src.runtime import ContainerRuntime

REPO = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def test_repository_config_loads(self):
        config = load_config(str(REPO / "config.yaml"))
        self.assertGreater(config.general.runs, 0)
        self.assertEqual([c.name for c in config.containers], ["postgres", "redis", "rabbitmq", "nginx"])
        self.assertEqual(config.logs_path, f"{config.general.scripts_folder}/logs")

    def test_energy_section_is_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = (REPO / "config.yaml").read_text().split("energy:")[0] + \
                "stressload:" + (REPO / "config.yaml").read_text().split("stressload:")[1]
            path = Path(tmp) / "config.yaml"
            path.write_text(text)
            config = load_config(str(path))
        self.assertFalse(config.energy.enable_rapl)
        self.assertFalse(config.energy.enable_tomada)


class WriteToFileTest(unittest.TestCase):
    def test_concurrent_writers_write_header_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/out.csv"
            threads = [threading.Thread(target=utils.write_to_file, args=(path, "h", str(i))) for i in range(50)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            lines = Path(path).read_text().splitlines()
        self.assertEqual(lines[0], "h")
        self.assertEqual(lines.count("h"), 1)
        self.assertEqual(len(lines), 51)


class CommandErrorTest(unittest.TestCase):
    def test_failed_command_raises_or_returns_none(self):
        with mock.patch("builtins.print"):
            self.assertIsNone(utils.execute_command("exit 3", continue_if_error=True))
            with self.assertRaises(utils.CommandError) as ctx:
                utils.execute_command("exit 3")
        self.assertEqual(ctx.exception.code, 3)
        self.assertIsInstance(ctx.exception, SystemExit)  # mesma semantica do exit() original

    def test_output_is_joined_in_one_line(self):
        self.assertEqual(utils.execute_command("printf 'a\\nb\\n'"), "ab")


class StartRetryTest(unittest.TestCase):
    def test_gives_up_after_max_retries(self):
        runtime = ContainerRuntime("docker")
        calls = []

        def fake_execute(command, informative=False, continue_if_error=False, error_informative=True):
            calls.append(command)
            if " run " in command:
                if continue_if_error:
                    return None
                raise utils.CommandError(command, 125)
            return "true" if "inspect" in command else ""

        with mock.patch.object(utils, "execute_command", fake_execute), \
                mock.patch.object(lifecycle, "execute_command", fake_execute), \
                mock.patch.object(lifecycle.time, "sleep"), mock.patch("builtins.print"):
            result = lifecycle.ContainerLifecycle(runtime).start("c", "img", 1, 2)

        self.assertIsNone(result)
        runs = [c for c in calls if " run " in c]
        self.assertEqual(len(runs), 1 + lifecycle.START_MAX_RETRIES + 1)


class StressloadGiveUpTest(unittest.TestCase):
    def test_skips_container_that_never_starts(self):
        """Antes, desistir de subir um container prendia a thread num loop infinito."""
        config = mock.Mock(min_qtt_containers=3, max_qtt_containers=3)
        container = mock.Mock(min_container_wait_time=0, max_container_wait_time=0, port=80)
        container.name = "nginx"

        with tempfile.TemporaryDirectory() as tmp:
            load = stressload.StressLoad(ContainerRuntime("docker"), config, tmp, tmp, False)
            load.lifecycle = mock.Mock()
            load.lifecycle.start.side_effect = [None, 1, 1]
            with mock.patch.object(stressload, "execute_command") as execute, \
                    mock.patch.object(stressload.time, "sleep"), mock.patch("builtins.print"):
                done = load.container_stressload(container, 8185, 0)
            errors = Path(tmp, "errors.csv").read_text().splitlines()

        self.assertEqual(done, 2)
        execute.assert_called_once_with("docker rm -v -f temp_nginx-0-0", continue_if_error=False,
                                        error_informative=False)
        self.assertEqual(load.lifecycle.stop.call_count, 2)
        self.assertEqual(errors[0], "event;container;date_time")
        self.assertTrue(errors[1].startswith("start;temp_nginx;"))


class KernelLogTest(unittest.TestCase):
    def _monitoring(self, tmp, disk_device=None):
        from src.monitoring import MonitoringEnvironment
        config = mock.Mock(logs_path=tmp, containers=[], energy=mock.Mock())
        config.general.scripts_folder = tmp
        config.monitoring.disk_device = disk_device
        return MonitoringEnvironment(config, ContainerRuntime("docker"), "env")

    def test_only_new_dmesg_lines_are_logged(self):
        from src import monitoring
        outputs = iter([["[1] Killed process 1"], [],
                        ["[1] Killed process 1", "[2] Killed process 2"], ["[5] oom"]])
        with tempfile.TemporaryDirectory() as tmp:
            env = self._monitoring(tmp)
            with mock.patch.object(monitoring, "execute_command_lines", lambda command: next(outputs)):
                env.kernel_killed_process_monitoring("t1")
                env.kernel_killed_process_monitoring("t2")
            killed = Path(tmp, "killed.csv").read_text().splitlines()
            oom = Path(tmp, "omm.csv").read_text().splitlines()
        self.assertEqual(killed, ["log;date_time", "[1] Killed process 1;t1", "[2] Killed process 2;t2"])
        self.assertEqual(oom, ["log;date_time", "[5] oom;t2"])

    def test_configured_disk_skips_detection(self):
        from src import monitoring
        with tempfile.TemporaryDirectory() as tmp:
            env = self._monitoring(tmp, disk_device="nvme0n1")
            with mock.patch.object(monitoring, "detect_root_disk") as detect, \
                    mock.patch.object(env, "_start_thread"), mock.patch("builtins.print"):
                env.start()
        detect.assert_not_called()
        self.assertEqual(env.disk_device, "nvme0n1")

    def test_build_manifest_is_copied_to_logs(self):
        from src import monitoring
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp, "manifest.txt")
            manifest.write_text("docker=27.2.0\n")
            env = self._monitoring(tmp, disk_device="vda")
            with mock.patch.object(monitoring, "BUILD_MANIFEST", str(manifest)), \
                    mock.patch.object(env, "_start_thread"), mock.patch("builtins.print"):
                env.start()
            self.assertEqual(Path(tmp, "build-manifest.txt").read_text(), "docker=27.2.0\n")

    def test_root_disk_falls_back_to_vda(self):
        from src import runtime
        with mock.patch.object(runtime, "execute_command", return_value=None), mock.patch("builtins.print"):
            self.assertEqual(runtime.detect_root_disk(), "vda")


if __name__ == "__main__":
    unittest.main()
