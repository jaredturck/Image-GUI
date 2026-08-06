import unittest

from hardware_planner import plan_attempts
from model_registry import LAUNCHER_MODEL_IDS, VALIDATION_ERRORS, get_model_profile
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

    return {
        "backend": backend,
        "gpus": gpus,
        "system_ram_gib": float(system_ram_gib),
        "usable_system_ram_gib": max(0.0, float(system_ram_gib) - 6.0),
        "supports_bfloat16": backend == "cuda",
        "supports_float16": True,
        "supports_bitsandbytes_int8": bitsandbytes and backend == "cuda",
        "supports_bitsandbytes_nf4": bitsandbytes and backend == "cuda",
        "supports_vllm": vllm and backend == "cuda",
        "signature": signature,
        "platform": "linux",
        "software": {},
    }


class PlannerTests(unittest.TestCase):
    def test_registry_is_valid(self):
        self.assertEqual(VALIDATION_ERRORS, [])

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


    def test_vllm_native_artifact_requires_vllm(self):
        hardware = fake_hardware("no_vllm", "cuda", 128, [24, 24], vllm=False)
        result = plan_attempts("Qwen/Qwen3-Coder-Next-GGUF:Q4_K_M", hardware=hardware)
        self.assertEqual(result["status"], "cannot_run")

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


if __name__ == "__main__":
    unittest.main()
