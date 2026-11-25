"""Nome da pasta do cenario -> (sistema, software, versao). So' usa a biblioteca padrao."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plot"))
from scenario import extract_categorical_info  # noqa: E402

UNKNOWN = ("Unknown OS", "Unknown Software", "N/A")


class ScenarioTest(unittest.TestCase):
    def test_versioned_names(self):
        cases = {
            "debian12_docker_27.2.0": ("Debian 12", "Docker", "27.2.0"),
            "ubuntu24.04_podman_5.1.2": ("Ubuntu 24.04", "Podman", "5.1.2"),
            "ubuntu22_docker_v28.0.1": ("Ubuntu 22", "Docker", "28.0.1"),
            "fedora45_podman_5.1.2_rejuvenation": ("Fedora 45", "Podman", "5.1.2"),
            "Debian13_Docker_27.2.0": ("Debian 13", "Docker", "27.2.0"),
            "arch_docker_27.2.0": ("Arch", "Docker", "27.2.0"),
        }
        for folder, expected in cases.items():
            with self.subTest(folder=folder):
                self.assertEqual(extract_categorical_info(folder), expected)

    def test_names_without_version_are_unknown(self):
        for folder in ["debian12_docker_novo", "ubuntu22_docker_velho", "ubuntu24_podman", "xyz",
                       "debian12_docker_27.2.0x"]:
            with self.subTest(folder=folder):
                self.assertEqual(extract_categorical_info(folder), (None, None, None))
                self.assertEqual(extract_categorical_info(folder, unknown=UNKNOWN), UNKNOWN)


if __name__ == "__main__":
    unittest.main()
