import os
import base64
import tempfile
import unittest
from unittest.mock import patch

import app_config
import install


class AppConfigTests(unittest.TestCase):
    def test_removed_config_keys_are_not_carried_forward(self):
        merged = app_config.deep_merge(
            app_config.DEFAULT_CONFIG,
            {"paths": {"comfyui_dir": "/old/comfy"}, "installer": {"managed_comfyui": True}},
        )
        self.assertNotIn("comfyui_dir", merged["paths"])
        self.assertNotIn("installer", merged)

    def test_packaged_macos_uses_application_support_and_standard_hf_cache(self):
        with tempfile.TemporaryDirectory() as home:
            environment = {"HOME": home, "IMAGE_GUI_PACKAGED": "1"}
            with patch.dict(os.environ, environment, clear=True), patch("platform.system", return_value="Darwin"):
                support = os.path.join(home, "Library", "Application Support", "Image GUI")
                self.assertEqual(app_config.config_dir(), support)
                self.assertEqual(app_config.runtime_venv_dir(), os.path.join(support, "venv"))
                self.assertEqual(app_config.chat_history_path(), os.path.join(support, "chat_history.enc"))
                self.assertEqual(
                    app_config.huggingface_cache_root(),
                    os.path.join(home, ".cache", "huggingface"),
                )
                self.assertEqual(app_config.default_output_root(), os.path.join(home, "Pictures", "Image GUI"))

    def test_hf_hub_cache_override_takes_precedence(self):
        environment = {"HF_HOME": "/tmp/hf-home", "HF_HUB_CACHE": "/tmp/shared-hf/hub"}
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(app_config.huggingface_cache_root(), "/tmp/shared-hf")

    def test_runtime_environment_preserves_standard_hf_override(self):
        config = app_config.deep_merge(
            app_config.DEFAULT_CONFIG,
            {"paths": {"huggingface_cache_dir": "/tmp/saved-cache"}},
        )
        environment = {"HF_HOME": "/tmp/external-hf-home"}
        with patch.dict(os.environ, environment, clear=True):
            app_config.apply_runtime_environment(config)
            self.assertEqual(os.environ["HF_HOME"], "/tmp/external-hf-home")
            self.assertNotIn("HF_HUB_CACHE", os.environ)

    def test_packaged_install_uses_only_the_hash_lock(self):
        requirements = install.requirements_for_current_platform(packaged=True)
        self.assertEqual([path.name for path in requirements], ["requirements-lock-macos.txt"])

    def test_source_macos_install_is_one_constrained_environment(self):
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
            requirements = install.requirements_for_current_platform(packaged=False)
        self.assertEqual(
            [path.name for path in requirements],
            ["requirements.txt", "requirements-macos.txt", "requirements-macos-constraints.txt"],
        )

    def test_packaged_setup_creates_a_valid_chat_history_key(self):
        with tempfile.TemporaryDirectory() as home:
            environment = {"HOME": home, "IMAGE_GUI_PACKAGED": "1"}
            with patch.dict(os.environ, environment, clear=True), patch("platform.system", return_value="Darwin"):
                install.ensure_application_settings()
                config = app_config.load_user_config()
                key = base64.b64decode(config["secrets"]["chat_history_key_b64"], validate=True)
                self.assertEqual(len(key), 32)


if __name__ == "__main__":
    unittest.main()
