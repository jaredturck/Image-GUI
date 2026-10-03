import unittest

from app_config import DEFAULT_CONFIG
from hardware_detection import reserve_for_mps
from hardware_planner import plan_attempts
from model_registry import LAUNCHER_MODEL_IDS, MODEL_PROFILES, VALIDATION_ERRORS, candidate_memory_estimate, get_model_profile
from plan_history import workload_covers
from planner_runtime import build_block_device_map


def fake_hardware(signature, backend, system_ram_gib, gpu_sizes=None, bitsandbytes=True, vllm=True):
    gpus = []
    for index, total_vram_gib in enumerate(gpu_sizes or []):
        reserve_gib = 1.0 if total_vram_gib <= 12 else 2.0
        gpus.append({
            "index": index,
            "name": f"Test GPU {index}",
            "total_vram_gib": float(total_vram_gib),
            "reserve_gib": reserve_gib,
            "usable_vram_gib": float(total_vram_gib) - reserve_gib,
            "compute_capability": [8, 6],
            "supports_bfloat16": True,
            "supports_float16": True,
            "supports_bitsandbytes_int8": bitsandbytes,
            "supports_bitsandbytes_nf4": bitsandbytes,
        })

    ram_reserve = 1.0 if backend == "mps" else 6.0
    return {
        "backend": backend,
        "gpus": gpus,
        "system_ram_gib": float(system_ram_gib),
        "system_ram_reserve_gib": ram_reserve,
        "usable_system_ram_gib": max(0.0, float(system_ram_gib) - ram_reserve),
        "supports_bfloat16": backend in ["cuda", "mps"],
        "supports_float16": True,
        "supports_bitsandbytes_int8": bitsandbytes and backend in ["cuda", "mps"],
        "supports_bitsandbytes_nf4": bitsandbytes and backend in ["cuda", "mps"],
        "supports_vllm": vllm and backend == "cuda",
        "signature": signature,
        "platform": "darwin" if backend == "mps" else "linux",
        "software": {},
    }


class PlannerTests(unittest.TestCase):
    def test_registry_is_valid(self):
        self.assertEqual(VALIDATION_ERRORS, [])

    def test_removed_model_stacks_are_absent(self):
        removed = {
            "anima",
            "Qwen/Qwen3-Coder-Next-GGUF:Q4_K_M",
            "real_esrgan",
            "realesrgan",
        }
        self.assertTrue(removed.isdisjoint(LAUNCHER_MODEL_IDS))
        self.assertIsNone(get_model_profile("anima"))
        self.assertIsNone(get_model_profile("Qwen/Qwen3-Coder-Next-GGUF:Q4_K_M"))

    def test_dual_24_gib_supports_every_launcher_model(self):
        hardware = fake_hardware("dual_24", "cuda", 128, [24, 24])
        unsupported = []
        for model_reference in LAUNCHER_MODEL_IDS:
            result = plan_attempts(model_reference, hardware=hardware)
            if result["status"] != "ready":
                unsupported.append(model_reference)
        self.assertEqual(unsupported, [])

    def test_flux2_preserves_existing_fast_path(self):
        hardware = fake_hardware("dual_24_flux", "cuda", 128, [24, 24])
        result = plan_attempts("flux_2", hardware=hardware)
        first = result["attempts"][0]
        self.assertEqual(first["plan_id"], "flux2_current_dual_3090_staged_int8")
        self.assertEqual(first["max_memory"], {0: "22.00GiB", 1: "22.00GiB"})
        self.assertEqual(first["selected_physical_gpu_ids"], [0, 1])

    def test_exact_fast_path_ignores_ineligible_preferred_gpu(self):
        hardware = fake_hardware("mixed_exact", "cuda", 128, [12, 24, 24])
        result = plan_attempts("flux_2", hardware=hardware, preferred_gpu=0)
        first = result["attempts"][0]
        self.assertEqual(first["template_id"], "current_exact_fast_path")
        self.assertEqual(first["selected_physical_gpu_ids"], [1, 2])

    def test_cpu_fallback_uses_cpu_execution(self):
        hardware = fake_hardware("cpu_32", "cpu", 32, [], bitsandbytes=False, vllm=False)
        result = plan_attempts("openai-community/gpt2-large", hardware=hardware)
        self.assertEqual(result["status"], "ready")
        first = result["attempts"][0]
        self.assertEqual(first["placement"], "cpu_only")
        self.assertEqual(first["execution_backend"], "cpu")
        self.assertEqual(first["execution_device"], "cpu")

    def test_unequal_gpu_block_map_is_proportional(self):
        plan = {
            "logical_gpu_count": 2,
            "max_memory": {0: "7GiB", 1: "18GiB"},
        }
        device_map = build_block_device_map("blocks", 48, plan=plan)
        counts = {0: 0, 1: 0}
        for device in device_map.values():
            counts[device] += 1
        self.assertEqual(counts, {0: 13, 1: 35})


    def test_flux2_int4_fallback_is_progressive(self):
        profile = get_model_profile("flux_2")
        plans = [
            plan
            for plan in profile["candidate_plans"]
            if plan.get("template_id") == "resident_int4_single"
        ]
        self.assertGreaterEqual(len(plans), 2)
        self.assertEqual(plans[0]["component_precision"]["text_encoder"], "int4")
        self.assertEqual(plans[0]["component_precision"]["transformer"], "int8")
        self.assertEqual(plans[1]["component_precision"]["transformer"], "int4")

    def test_explicit_workload_is_restored_after_retry_planning(self):
        hardware = fake_hardware("retry_workload", "cuda", 64, [24])
        workload = {"width": 1536, "height": 1024, "batch_size": 1}
        result = plan_attempts("flux_1", workload=workload, hardware=hardware)
        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["attempts"][0]["restore_workload"])
        self.assertEqual(result["attempts"][0]["workload"], workload)

    def test_default_planning_does_not_override_gui_defaults(self):
        hardware = fake_hardware("default_workload", "cuda", 64, [24])
        result = plan_attempts("flux_1", hardware=hardware)
        self.assertEqual(result["status"], "ready")
        self.assertFalse(result["attempts"][0]["restore_workload"])


    def test_vllm_model_skips_exact_path_when_vllm_is_missing(self):
        hardware = fake_hardware("no_vllm_fallback", "cuda", 128, [24, 24], vllm=False)
        result = plan_attempts("Qwen/Qwen3.6-35B-A3B", hardware=hardware)
        self.assertEqual(result["status"], "ready")
        self.assertNotEqual(result["attempts"][0]["template_id"], "current_exact_fast_path")
        self.assertNotEqual(result["attempts"][0]["template_id"], "vllm_native")


    def test_cpu_bitsandbytes_can_make_a_large_llm_plan_practical(self):
        hardware = fake_hardware("cpu_bnb", "cpu", 32, [], bitsandbytes=False, vllm=False)
        hardware["supports_bitsandbytes_int8"] = True
        hardware["supports_bitsandbytes_nf4"] = True
        result = plan_attempts("mistralai/Mistral-7B-Instruct-v0.3", hardware=hardware)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["attempts"][0]["template_id"], "cpu_int8")
        self.assertEqual(result["attempts"][0]["component_precision"]["language_model"], "int8")

    def test_cuda_or_cpu_exact_path_selects_a_cuda_gpu(self):
        hardware = fake_hardware("rmbg_cuda", "cuda", 32, [8, 12])
        result = plan_attempts("rmbg_1_4", hardware=hardware)
        self.assertEqual(result["status"], "ready")
        first = result["attempts"][0]
        self.assertEqual(first["template_id"], "current_exact_fast_path")
        self.assertEqual(first["selected_physical_gpu_ids"], [1])
        self.assertEqual(first["execution_device"], "cuda:0")

    def test_vae_is_not_quantized_in_flux2_int4_metadata(self):
        profile = get_model_profile("flux_2")
        int4_plans = [
            plan
            for plan in profile["candidate_plans"]
            if "int4" in plan.get("template_id", "")
        ]
        self.assertTrue(int4_plans)
        for plan in int4_plans:
            self.assertEqual(plan["component_precision"]["vae"], "native")

    def test_mps_kandinsky_uses_staging_without_changing_resolution(self):
        hardware = fake_hardware("mps_8", "mps", 8, [], bitsandbytes=True, vllm=False)
        result = plan_attempts("kandinsky_5_t2i_lite_sft", hardware=hardware)
        self.assertEqual(result["status"], "ready")
        first = result["attempts"][0]
        self.assertEqual(first["execution_backend"], "mps")
        self.assertEqual(first["placement"], "mps_model_specific_staging")
        self.assertEqual(first["workload"]["width"], 1280)
        self.assertEqual(first["workload"]["height"], 768)
        self.assertEqual(first["workload"]["num_images_per_prompt"], 1)
        self.assertEqual(first["component_precision"]["qwen_text_encoder"], "int4")
        self.assertEqual(first["component_precision"]["clip_text_encoder"], "native")
        self.assertEqual(first["component_precision"]["vae"], "native")

    def test_mps_more_memory_selects_higher_quality_plans(self):
        medium = plan_attempts("kandinsky_5_t2i_lite_sft", hardware=fake_hardware("mps_16", "mps", 16, [], bitsandbytes=True, vllm=False))
        large = plan_attempts("kandinsky_5_t2i_lite_sft", hardware=fake_hardware("mps_32", "mps", 32, [], bitsandbytes=True, vllm=False))
        self.assertEqual(medium["attempts"][0]["template_id"], "staged_int8_single")
        self.assertEqual(large["attempts"][0]["template_id"], "resident_native_single")
        self.assertEqual(medium["attempts"][0]["workload"]["num_images_per_prompt"], 3)
        self.assertEqual(large["attempts"][0]["workload"]["num_images_per_prompt"], 3)

    def test_mps_quantization_capability_is_respected(self):
        hardware = fake_hardware("mps_8_no_bnb", "mps", 8, [], bitsandbytes=False, vllm=False)
        result = plan_attempts("kandinsky_5_t2i_lite_sft", hardware=hardware)
        self.assertEqual(result["status"], "cannot_run")

    def test_kandinsky_progressive_quantization_protects_vae_and_clip(self):
        profile = get_model_profile("kandinsky_5_t2i_lite_sft")
        plans = [plan for plan in profile["candidate_plans"] if plan.get("template_id") == "staged_int4_single"]
        self.assertGreaterEqual(len(plans), 2)
        self.assertEqual(plans[0]["component_precision"]["qwen_text_encoder"], "int4")
        self.assertEqual(plans[0]["component_precision"]["transformer"], "int8")
        self.assertEqual(plans[0]["component_precision"]["clip_text_encoder"], "native")
        self.assertEqual(plans[0]["component_precision"]["vae"], "native")
        self.assertEqual(plans[1]["component_precision"]["transformer"], "int4")
        self.assertEqual(plans[1]["component_precision"]["vae"], "native")

    def test_staged_candidates_exist_for_most_diffusion_profiles(self):
        staged_profiles = [profile for profile in MODEL_PROFILES.values() if profile.get("placement_support", {}).get("custom_staging")]
        supported = [profile for profile in staged_profiles if any("model_specific_staging" in plan.get("placement", "") for plan in profile.get("candidate_plans", []))]
        self.assertGreaterEqual(len(staged_profiles), 20)
        self.assertEqual(len(supported), len(staged_profiles))

    def test_concurrent_image_count_changes_phase_memory_but_not_resolution(self):
        profile = get_model_profile("kandinsky_5_t2i_lite_sft")
        plan = next(plan for plan in profile["candidate_plans"] if plan.get("template_id") == "staged_int8_single")
        default = dict(profile["default_workload"])
        single = dict(default)
        single["num_images_per_prompt"] = 1
        default_estimate = candidate_memory_estimate(profile, plan["component_precision"], plan["placement"], default)
        single_estimate = candidate_memory_estimate(profile, plan["component_precision"], plan["placement"], single)
        self.assertLess(single_estimate["required_total_usable_vram_gib"], default_estimate["required_total_usable_vram_gib"])
        self.assertEqual(single["width"], default["width"])
        self.assertEqual(single["height"], default["height"])

    def test_mps_llm_uses_quantized_plan_when_native_does_not_fit(self):
        hardware = fake_hardware("mps_llm_16", "mps", 16, [], bitsandbytes=True, vllm=False)
        result = plan_attempts("mistralai/Mistral-7B-Instruct-v0.3", hardware=hardware)
        self.assertEqual(result["status"], "ready")
        first = result["attempts"][0]
        self.assertEqual(first["execution_backend"], "mps")
        self.assertEqual(first["component_precision"]["language_model"], "int8")



    def test_mps_64_supports_all_mps_capable_image_video_profiles(self):
        hardware = fake_hardware("mps_64_all", "mps", 64, [], bitsandbytes=True, vllm=False)
        unsupported = []
        for model_key, profile in MODEL_PROFILES.items():
            if profile.get("category") not in ["image_diffusion", "image_diffusion_upscale", "video_diffusion"]:
                continue
            support = str(profile.get("backend_support", {}).get("mps", ""))
            if "unsupported" in support or "not_supported" in support:
                continue
            if plan_attempts(model_key, hardware=hardware)["status"] != "ready":
                unsupported.append(model_key)
        self.assertEqual(unsupported, [])

    def test_mps_64_has_a_plan_for_every_retained_launcher_model(self):
        hardware = fake_hardware("mps_64_catalogue", "mps", 64, [], bitsandbytes=True, vllm=False)
        unsupported = [
            model_reference
            for model_reference in LAUNCHER_MODEL_IDS
            if plan_attempts(model_reference, hardware=hardware)["status"] != "ready"
        ]
        self.assertEqual(unsupported, [])

    def test_mps_reserve_scales_from_physical_unified_memory(self):
        self.assertEqual(reserve_for_mps(8, DEFAULT_CONFIG), 1.0)
        self.assertEqual(reserve_for_mps(32, DEFAULT_CONFIG), 3.2)
        self.assertEqual(reserve_for_mps(192, DEFAULT_CONFIG), 8.0)

    def test_video_workload_policy_preserves_resolution(self):
        profile = get_model_profile("hunyuan_video_1_5_t2v")
        self.assertIn("width", profile["workload_policy"]["fixed"])
        self.assertIn("height", profile["workload_policy"]["fixed"])
        self.assertIn("num_videos_per_prompt", profile["workload_policy"]["adjustable"])

    def test_plan_history_ignores_sequential_batch_size(self):
        validated = {"width": 1280, "height": 768, "batch_size": 1, "num_images_per_prompt": 1}
        requested = {"width": 1280, "height": 768, "batch_size": 10, "num_images_per_prompt": 1}
        self.assertTrue(workload_covers(validated, requested))


if __name__ == "__main__":
    unittest.main()
