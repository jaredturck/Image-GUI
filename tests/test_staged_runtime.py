import unittest
from unittest.mock import patch

import torch

from planner_runtime import LazyStagedComponent, StagedComponentManager, pipeline_quantization_config, registry_pipeline_component_map


class DummyComponent(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.dtype = torch.float32

    def forward(self, value):
        return value + 1


class StagedRuntimeTests(unittest.TestCase):
    def test_kandinsky_component_mapping_matches_pipeline_slots(self):
        profile = {
            "components": {
                "qwen_text_encoder": {"role": "large_text_encoder"},
                "clip_text_encoder": {"role": "small_text_encoder"},
                "transformer": {"role": "image_dit"},
                "vae": {"role": "vae"},
            }
        }
        specs = {
            "text_encoder": ["transformers", "QwenModel"],
            "text_encoder_2": ["transformers", "CLIPTextModel"],
            "transformer": ["diffusers", "Transformer"],
            "vae": ["diffusers", "AutoencoderKL"],
        }
        mapping = registry_pipeline_component_map(profile, specs)
        self.assertEqual(mapping["qwen_text_encoder"], "text_encoder")
        self.assertEqual(mapping["clip_text_encoder"], "text_encoder_2")
        self.assertEqual(mapping["transformer"], "transformer")
        self.assertEqual(mapping["vae"], "vae")

    def test_phase_transition_releases_previous_component(self):
        profile = {
            "execution_phases": [
                {"name": "text_encoding", "required_components": ["text_encoder"]},
                {"name": "denoising", "required_components": ["transformer"]},
            ]
        }
        manager = StagedComponentManager("dummy", {}, profile)
        text_proxy = LazyStagedComponent(manager, "text_encoder", "text_encoder", DummyComponent, "dummy", {})
        transformer_proxy = LazyStagedComponent(manager, "transformer", "transformer", DummyComponent, "dummy", {})
        manager.add_proxy("text_encoder", text_proxy)
        manager.add_proxy("transformer", transformer_proxy)

        def fake_load(component_class, model_id, component_name, **kwargs):
            return DummyComponent(component_name)

        with patch("planner_runtime.load_component", side_effect=fake_load), patch("planner_runtime.clear_accelerator_cache") as clear_cache:
            self.assertEqual(text_proxy(torch.tensor(1)).item(), 2)
            self.assertIsNotNone(text_proxy.real_component)
            self.assertIsNone(transformer_proxy.real_component)

            self.assertEqual(transformer_proxy(torch.tensor(2)).item(), 3)
            self.assertIsNone(text_proxy.real_component)
            self.assertIsNotNone(transformer_proxy.real_component)
            clear_cache.assert_called()

            manager.release_all()
            self.assertIsNone(transformer_proxy.real_component)


    def test_proxy_presents_expected_component_identity_without_loading(self):
        profile = {"execution_phases": [{"name": "text", "required_components": ["text_encoder"]}]}
        manager = StagedComponentManager("dummy", {}, profile)
        proxy = LazyStagedComponent(manager, "text_encoder", "text_encoder", DummyComponent, "dummy", {})
        manager.add_proxy("text_encoder", proxy)
        self.assertIs(proxy.__class__, DummyComponent)
        self.assertEqual(proxy.__module__, DummyComponent.__module__)
        self.assertIsInstance(proxy, DummyComponent)
        self.assertIsNone(proxy.real_component)

    def test_config_backed_attribute_does_not_load_weights(self):
        profile = {"execution_phases": [{"name": "decode", "required_components": ["vae"]}]}
        manager = StagedComponentManager("dummy", {}, profile)
        proxy = LazyStagedComponent(
            manager,
            "vae",
            "vae",
            DummyComponent,
            "dummy",
            {"temporal_compression_ratio": 4, "spatial_compression_ratio": 16},
        )
        manager.add_proxy("vae", proxy)
        self.assertEqual(proxy.temporal_compression_ratio, 4)
        self.assertEqual(proxy.spatial_compression_ratio, 16)
        self.assertIsNone(proxy.real_component)

    def test_sd3_component_mapping_uses_all_text_encoder_slots(self):
        profile = {
            "components": {
                "clip_l": {"role": "small_text_encoder"},
                "clip_g": {"role": "small_text_encoder"},
                "t5_encoder": {"role": "large_text_encoder"},
                "transformer": {"role": "image_dit"},
                "vae": {"role": "vae"},
            }
        }
        specs = {
            "text_encoder": ["transformers", "CLIPTextModelWithProjection"],
            "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
            "text_encoder_3": ["transformers", "T5EncoderModel"],
            "transformer": ["diffusers", "SD3Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }
        mapping = registry_pipeline_component_map(profile, specs)
        self.assertEqual(mapping["clip_l"], "text_encoder")
        self.assertEqual(mapping["clip_g"], "text_encoder_2")
        self.assertEqual(mapping["t5_encoder"], "text_encoder_3")

    def test_private_introspection_does_not_load_weights(self):
        profile = {"execution_phases": [{"name": "text", "required_components": ["text_encoder"]}]}
        manager = StagedComponentManager("dummy", {}, profile)
        proxy = LazyStagedComponent(manager, "text_encoder", "text_encoder", DummyComponent, "dummy", {})
        manager.add_proxy("text_encoder", proxy)
        self.assertFalse(hasattr(proxy, "_hf_hook"))
        self.assertIsNone(proxy.real_component)


if __name__ == "__main__":
    unittest.main()
