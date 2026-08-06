import os, dotenv

dotenv.load_dotenv()
if os.environ['USE_HHD'] == 'True':
    os.environ["HF_HOME"] = "/mnt/8TB_HDD/hf_cache"
    os.environ["HF_HUB_CACHE"] = "/mnt/8TB_HDD/hf_cache/hub"
    os.environ["TRANSFORMERS_CACHE"] = "/mnt/8TB_HDD/hf_cache/hub"

import os, sys, json, threading, time, gc, re, base64, queue, io, asyncio, uuid
import numpy as np
import torch
import sounddevice as sd
import customtkinter as ctk
from PIL import Image
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

def get_flag_value(flag):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

GEMMA_4_MODEL_ID = "google/gemma-4-31B-it"
QWEN3_CODER_GGUF_MODEL_ID = "Qwen/Qwen3-Coder-Next-GGUF:Q4_K_M"

CHAT_MODEL_QUANTIZATION = {
    GEMMA_4_MODEL_ID: "8bit",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "8bit",
    "Qwen/Qwen2.5-32B-Instruct": "8bit",
    "tiiuae/Falcon-H1-34B-Instruct": "8bit",
    "Qwen/Qwen3.6-27B": "4bit",
}

CHAT_MODEL_CPU_OFFLOAD_IDS = set()

VLLM_MODEL_CONFIGS = {
    "Qwen/Qwen3.6-35B-A3B-FP8": {
        "tensor_parallel_size": 2,
        "dtype": "float16",
        "gpu_memory_utilization": 0.90,
        "max_model_len": 8192,
        "language_model_only": True,
        "enforce_eager": True,
        "disable_log_stats": True,
    },
    QWEN3_CODER_GGUF_MODEL_ID: {
        "tokenizer": "Qwen/Qwen3-Coder-Next",
        "hf_config_path": "Qwen/Qwen3-Coder-Next",
        "load_format": "gguf",
        "tensor_parallel_size": 2,
        "dtype": "float16",
        "gpu_memory_utilization": 0.99,
        "max_model_len": 4096,
        "language_model_only": True,
        "enforce_eager": True,
        "disable_log_stats": True,
    },
}

PROCESSOR_CHAT_MODEL_IDS = {
    "Qwen/Qwen3.6-27B",
    GEMMA_4_MODEL_ID,
}

REASONING_MODEL_IDS = {
    "LiquidAI/LFM2.5-1.2B-Thinking",
    "microsoft/Phi-4-reasoning",
    "Qwen/Qwen3-14B",
    "openai/gpt-oss-20b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Qwen/Qwen3.6-27B",
    GEMMA_4_MODEL_ID,
    "Qwen/Qwen3.6-35B-A3B-FP8",
    QWEN3_CODER_GGUF_MODEL_ID,
}

REASONING_SUMMARY_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REASONING_SUMMARY_CPU_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

class StopGenerationCriteria(StoppingCriteria):
    def __init__(self, gui):
        self.gui = gui

    def __call__(self, input_ids, scores, **kwargs):
        if self.gui.stop_generation:
            if not self.gui.stop_reason:
                self.gui.stop_reason = "stopped"
            return True

        if self.gui.max_generation_seconds and self.gui.stream_t0:
            if time.time() - self.gui.stream_t0 >= self.gui.max_generation_seconds:
                self.gui.stop_reason = "time"
                return True

        return False

class QueueTextStreamer:
    def __init__(self):
        self.text_queue = queue.Queue()

    def put(self, text):
        self.text_queue.put(text)

    def end(self):
        self.text_queue.put(None)

class ChatGUI:
    def __init__(self, model_id):
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        if model_id == "Qwen/Qwen3.6-35B-A3B":
            model_id = "Qwen/Qwen3.6-35B-A3B-FP8"

        self.model_id = model_id or "unknown-model"
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.store_path = os.path.join(self.base_dir, "chat_history.enc")

        self.model_lock = threading.Lock()
        self.model_loading = False
        self.pipe = None
        self.vllm_loop = None
        self.vllm_tokenizer = None
        self.vllm_request_id = None

        self.voice_model_id = "openai/whisper-medium"
        self.voice_lock = threading.Lock()
        self.voice_pipe = None
        self.voice_device = None
        self.voice_loading = False
        self.voice_recording = False
        self.voice_stream = None
        self.voice_chunks = []
        self.voice_sample_rate = 16000

        self.reasoning_enabled = self.model_id in REASONING_MODEL_IDS
        self.reasoning_model_id = REASONING_SUMMARY_MODEL_ID
        self.reasoning_lock = threading.Lock()
        self.reasoning_pipe = None
        self.reasoning_loading = False
        self.reasoning_queue = queue.Queue()
        self.reasoning_worker_started = False
        self.reasoning_processed_blocks = 0
        self.reasoning_done = False
        self.reasoning_done_added = False
        self.reasoning_summary_active = False
        self.reasoning_generation_id = 0

        self.system_prompt = "You are a helpful assistant."
        self.max_new_tokens = 256000
        self.max_generation_seconds = 300
        self.message_page_chars = 50000
        self.temperature = 0.7
        self.top_p = 0.95
        self.max_context_messages = 16

        self.store = {}
        self.chat_ids = []
        self.selected_chat_id = None

        self.chat_items = {}
        self.msg_widgets = []
        self.latex_images = {}

        self.app = ctk.CTk()
        self.app.title(f"Chat - {self.model_id}")
        if self.reasoning_enabled:
            self.app.geometry("1520x800")
        else:
            self.app.geometry("1200x800")

        self.left_panel = None
        self.right_panel = None
        self.reasoning_panel = None
        self.reasoning_view = None
        self.reasoning_status = None
        self.chat_list = None
        self.chat_view = None
        self.model_title = None
        self.status_pill = None
        self.input_box = None
        self.send_btn = None
        self.mic_btn = None
        self.new_btn = None
        self.delete_btn = None

        self.streamer = None
        self.stream_text = ""
        self.stream_done = False
        self.stream_label = None
        self.stream_meta_label = None
        self.stream_chat_id = None
        self.stream_t0 = 0.0
        self.stop_generation = False
        self.stop_reason = ""

        self.load_store()
        self.ensure_model_store()
        self.build_ui()
        self.refresh_chat_list()
        self.select_chat(self.chat_ids[0] if self.chat_ids else None)

        threading.Thread(target=self.load_model, daemon=True).start()
    
    def stop_message(self):
        self.stop_generation = True
        self.send_btn.configure(state="disabled")
        self.set_status("Stopping...")

    def get_chat_history_key(self):
        return base64.b64decode(os.environ["CHAT_HISTORY_KEY_B64"])

    def encrypt_store_bytes(self, data):
        key = self.get_chat_history_key()
        nonce = os.urandom(12)
        encrypted = AESGCM(key).encrypt(nonce, data, None)
        return nonce + encrypted

    def decrypt_store_bytes(self, data):
        key = self.get_chat_history_key()
        nonce = data[:12]
        encrypted = data[12:]
        return AESGCM(key).decrypt(nonce, encrypted, None)

    def load_store(self):
        tmp = self.store_path + ".tmp"
        if os.path.isfile(tmp):
            os.remove(tmp)

        if not os.path.isfile(self.store_path):
            self.store = {}
            return

        with open(self.store_path, "rb") as f:
            encrypted = f.read()

        data = self.decrypt_store_bytes(encrypted)
        self.store = json.loads(data.decode("utf-8"))

    def save_store(self):
        data = json.dumps(self.store, indent=2, ensure_ascii=False).encode("utf-8")
        encrypted = self.encrypt_store_bytes(data)

        tmp = self.store_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(encrypted)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, self.store_path)

    def ensure_model_store(self):
        if self.model_id not in self.store or not isinstance(self.store.get(self.model_id), dict):
            self.store[self.model_id] = {"chats": {}, "chat_order": []}

        m = self.store[self.model_id]
        if "chats" not in m or not isinstance(m.get("chats"), dict):
            m["chats"] = {}
        if "chat_order" not in m or not isinstance(m.get("chat_order"), list):
            m["chat_order"] = []

        if not m["chat_order"]:
            chat_id = self.make_chat_id()
            m["chat_order"] = [chat_id]
            m["chats"][chat_id] = self.new_chat_obj(chat_id)
            self.save_store()

        self.chat_ids = list(self.store[self.model_id]["chat_order"])

    def make_chat_id(self):
        return "chat_" + str(int(time.time() * 1000))

    def new_chat_obj(self, chat_id):
        idx = 1
        names = set()

        for cid, c in self.store[self.model_id]["chats"].items():
            name = c.get("name") or cid
            names.add(name)

        while True:
            name = f"Chat {idx}"
            if name not in names:
                break
            idx += 1

        return {
            "id": chat_id,
            "name": name,
            "created_at": int(time.time()),
            "messages": []
        }
    
    def create_stream_bubble(self):
        if self.stream_chat_id and self.stream_chat_id != self.selected_chat_id:
            self.stream_label = None
            self.stream_meta_label = None
            return

        row = ctk.CTkFrame(self.chat_view, fg_color="transparent")
        row.grid(sticky="ew", padx=10, pady=6)
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)

        bubble = ctk.CTkFrame(row, corner_radius=16, fg_color="#3a3a3a")
        bubble.grid(row=0, column=0, sticky="w", padx=6)
        bubble.grid_columnconfigure(0, weight=1)

        lab = ctk.CTkLabel(bubble, text="", text_color="#e6e6e6", justify="left", wraplength=640)
        lab.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))

        meta = ctk.CTkLabel(bubble, text="Live", text_color="#a8a8a8", font=ctk.CTkFont(size=12))
        meta.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 10))

        bubble.bind("<Double-Button-1>", lambda e: self.copy_message(self.get_display_message_text(self.stream_text, streaming=True)))
        lab.bind("<Double-Button-1>", lambda e: self.copy_message(self.get_display_message_text(self.stream_text, streaming=True)))

        self.msg_widgets.append((row, bubble, lab))
        self.stream_label = lab
        self.stream_meta_label = meta
        self.scroll_to_bottom()
    
    def append_message_only(self, role, content, chat_id=None):
        if chat_id:
            chat = self.get_chat_by_id(chat_id)
        else:
            chat = self.get_selected_chat()

        if not chat:
            return

        chat["messages"].append({
            "role": role,
            "content": content,
            "ts": int(time.time()),
        })
        self.save_store()
        self.refresh_chat_list()

    def clean_gemma_response_text(self, text):
        text = text or ""
        for token in ("<turn|>", "<|turn>", "<bos>", "<eos>"):
            text = text.replace(token, "")
        return text.strip()

    def get_reasoning_parts(self, text):
        text = text or ""

        if self.model_id == GEMMA_4_MODEL_ID:
            marker = "<|channel>thought"
            start = text.find(marker)
            if start < 0:
                return "", self.clean_gemma_response_text(text), False

            start = text.find("\n", start + len(marker))
            if start < 0:
                return "", "", False
            start += 1

            end_marker = "<channel|>"
            end = text.find(end_marker, start)
            if end < 0:
                return text[start:], "", False

            thinking = text[start:end]
            visible = self.clean_gemma_response_text(text[end + len(end_marker):])
            return thinking, visible, True

        start = text.find("<think>")
        end = text.find("</think>")

        if start >= 0:
            start += len("<think>")
        else:
            start = 0

        if end >= 0:
            return text[start:end], text[end + len("</think>"):].strip(), True

        return text[start:], "", False

    def get_display_message_text(self, text, streaming=False):
        text = text or ""

        if not self.reasoning_enabled:
            return text

        _, visible, closed = self.get_reasoning_parts(text)
        if closed:
            return visible or "Thinking..."

        if streaming:
            return "Thinking..."

        if self.model_id == GEMMA_4_MODEL_ID:
            return self.clean_gemma_response_text(text)

        return text

    def poll_streamer(self):
        s = self.streamer
        if s is None:
            return

        q = getattr(s, "text_queue", None)
        if q is None:
            return

        updated = False

        while not q.empty():
            chunk = q.get()
            if chunk is None:
                self.stream_done = True
                break

            self.stream_text += chunk
            updated = True

        if updated:
            self.process_reasoning_stream()

        if updated and self.stream_label is not None:
            display_text = self.get_display_message_text(self.stream_text, streaming=True)
            page_count = self.get_page_count(display_text)
            visible = self.get_page_text(display_text, page_count - 1)
            self.stream_label.configure(text=visible)

            if self.stream_meta_label is not None:
                if display_text == "Thinking...":
                    self.stream_meta_label.configure(text="Live")
                elif len(display_text) > self.message_page_chars:
                    self.stream_meta_label.configure(text=f"Live: latest {len(visible):,} of {len(display_text):,} chars")
                else:
                    self.stream_meta_label.configure(text=f"Live: {len(display_text):,} chars")

            self.scroll_to_bottom()

        if not self.stream_done:
            return self.app.after(100, self.poll_streamer)

        final = (self.stream_text or "").strip()
        if not final:
            final = "(stopped)" if self.stop_generation else "(no response)"

        chat_id = self.stream_chat_id
        self.append_message_only("assistant", final, chat_id)

        if chat_id == self.selected_chat_id:
            self.render_messages()

        dt = time.time() - float(self.stream_t0 or time.time())
        self.send_btn.configure(text="Send", state="normal", command=self.send_message)

        if self.stop_reason == "time":
            self.set_status(f"Time limit reached ({dt:.1f}s)")
        elif self.stop_reason == "tokens":
            self.set_status(f"Token limit reached ({dt:.1f}s)")
        elif self.stop_reason == "stopped" or self.stop_generation:
            self.set_status(f"Stopped ({dt:.1f}s)")
        else:
            self.set_status(f"Ready ({dt:.1f}s)")

        self.streamer = None
        self.stream_text = ""
        self.stream_done = False
        self.stream_label = None
        self.stream_meta_label = None
        self.stream_chat_id = None
        self.stream_t0 = 0.0
        self.stop_generation = False

    def build_ui(self):
        if self.reasoning_enabled:
            self.app.grid_columnconfigure(0, weight=1, minsize=320)
            self.app.grid_columnconfigure(1, weight=3, minsize=820)
            self.app.grid_columnconfigure(2, weight=1, minsize=340)
        else:
            self.app.grid_columnconfigure(0, weight=1, minsize=320)
            self.app.grid_columnconfigure(1, weight=3, minsize=820)
        self.app.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(self.app, corner_radius=16)
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)
        self.left_panel = left

        left_header = ctk.CTkFrame(left, fg_color="transparent")
        left_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        left_header.grid_columnconfigure(0, weight=1)

        left_title = ctk.CTkLabel(left_header, text="Chats", font=ctk.CTkFont(size=18, weight="bold"))
        left_title.grid(row=0, column=0, sticky="w")

        self.chat_list = ctk.CTkScrollableFrame(left, corner_radius=16)
        self.chat_list.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.chat_list.grid_columnconfigure(0, weight=1)

        right = ctk.CTkFrame(self.app, corner_radius=16)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 8 if self.reasoning_enabled else 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        self.right_panel = right

        header = ctk.CTkFrame(right, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        header.grid_columnconfigure(0, weight=1)

        self.model_title = ctk.CTkLabel(header, text=self.model_id, font=ctk.CTkFont(size=18, weight="bold"))
        self.model_title.grid(row=0, column=0, sticky="w")

        self.new_btn = ctk.CTkButton(header, text="New chat", width=90, command=self.create_chat)
        self.new_btn.grid(row=0, column=1, padx=(0, 8))

        self.delete_btn = ctk.CTkButton(header, text="Delete", width=70, fg_color="#444444", hover_color="#3a3a3a", command=self.delete_chat)
        self.delete_btn.grid(row=0, column=2, padx=(0, 8))

        self.status_pill = ctk.CTkLabel(header, text="Loading...", corner_radius=999, fg_color="#333333", text_color="#d0d0d0", padx=10, pady=4)
        self.status_pill.grid(row=0, column=3, sticky="e")

        self.chat_view = ctk.CTkScrollableFrame(right, corner_radius=16)
        self.chat_view.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))
        self.chat_view.grid_columnconfigure(0, weight=1)

        input_row = ctk.CTkFrame(right, corner_radius=16)
        input_row.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 14))
        input_row.grid_columnconfigure(0, weight=1)

        self.input_box = ctk.CTkTextbox(input_row, height=90)
        self.input_box.grid(row=0, column=0, rowspan=2, sticky="ew", padx=(10, 10), pady=10)

        button_col = ctk.CTkFrame(input_row, fg_color="transparent")
        button_col.grid(row=0, column=1, rowspan=2, sticky="ns", padx=(0, 10), pady=10)
        button_col.grid_rowconfigure(0, weight=1)
        button_col.grid_rowconfigure(1, weight=1)

        self.send_btn = ctk.CTkButton(button_col, text="Send", width=110, command=self.send_message)
        self.send_btn.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.mic_btn = ctk.CTkButton(button_col, text="🎙", width=110, fg_color="#444444", hover_color="#3a3a3a", command=self.toggle_voice_recording)
        self.mic_btn.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        if self.reasoning_enabled:
            reasoning = ctk.CTkFrame(self.app, corner_radius=16)
            reasoning.grid(row=0, column=2, sticky="nsew", padx=(0, 12), pady=12)
            reasoning.grid_columnconfigure(0, weight=1)
            reasoning.grid_rowconfigure(1, weight=1)
            self.reasoning_panel = reasoning

            reasoning_header = ctk.CTkFrame(reasoning, fg_color="transparent")
            reasoning_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
            reasoning_header.grid_columnconfigure(0, weight=1)

            reasoning_title = ctk.CTkLabel(reasoning_header, text="Reasoning", font=ctk.CTkFont(size=18, weight="bold"))
            reasoning_title.grid(row=0, column=0, sticky="w")

            self.reasoning_status = ctk.CTkLabel(
                reasoning_header,
                text="Loading...",
                corner_radius=999,
                fg_color="#333333",
                text_color="#d0d0d0",
                padx=10,
                pady=4,
            )
            self.reasoning_status.grid(row=0, column=1, sticky="e")

            self.reasoning_view = ctk.CTkScrollableFrame(reasoning, corner_radius=16)
            self.reasoning_view.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
            self.reasoning_view.grid_columnconfigure(0, weight=1)

        self.app.bind_all("<Control-Return>", self.send_message)
        self.app.bind_all("<MouseWheel>", self.on_mousewheel)
        self.app.bind_all("<Button-4>", self.on_mousewheel)
        self.app.bind_all("<Button-5>", self.on_mousewheel)

    def set_reasoning_status(self, text):
        if self.reasoning_status is not None:
            self.reasoning_status.configure(text=text)

    def clear_reasoning_panel(self):
        if self.reasoning_view is None:
            return

        for w in self.reasoning_view.winfo_children():
            w.destroy()

    def reset_reasoning_summary(self):
        if not self.reasoning_enabled:
            return

        self.reasoning_generation_id += 1
        self.reasoning_processed_blocks = 0
        self.reasoning_done = False
        self.reasoning_done_added = False
        self.reasoning_summary_active = False
        self.clear_reasoning_panel()
        self.set_reasoning_status("Waiting...")

    def add_reasoning_summary(self, summary, generation_id):
        if not self.reasoning_enabled:
            return
        if generation_id != self.reasoning_generation_id:
            return
        if self.reasoning_view is None:
            return

        summary = (summary or "").strip()
        if len(summary) < 5:
            return

        idx = sum(1 for w in self.reasoning_view.winfo_children() if isinstance(w, ctk.CTkFrame)) + 1

        item = ctk.CTkFrame(self.reasoning_view, corner_radius=14, fg_color="#2b2b2b")
        item.grid(sticky="ew", padx=8, pady=8)
        item.grid_columnconfigure(0, weight=1)

        lab = ctk.CTkLabel(
            item,
            text=f"{idx}. {summary}",
            text_color="#e6e6e6",
            justify="left",
            anchor="w",
            width=220,
            wraplength=210,
        )
        lab.grid(row=0, column=0, sticky="ew", padx=12, pady=10)

        item.bind("<Double-Button-1>", lambda e, t=summary: self.copy_message(t))
        lab.bind("<Double-Button-1>", lambda e, t=summary: self.copy_message(t))
        self.set_reasoning_status("Live")

    def add_reasoning_done(self, generation_id):
        if generation_id != self.reasoning_generation_id:
            return
        if self.reasoning_done_added:
            return
        if self.reasoning_view is None:
            return

        self.reasoning_done_added = True

        lab = ctk.CTkLabel(
            self.reasoning_view,
            text="Done",
            text_color="#8a8a8a",
            font=ctk.CTkFont(size=12),
        )
        lab.grid(sticky="w", padx=18, pady=(4, 12))

    def process_reasoning_stream(self):
        if not self.reasoning_enabled:
            return
        if self.reasoning_done:
            return

        thinking, _, closed = self.get_reasoning_parts(self.stream_text)

        thinking = thinking.replace("\r\n", "\n").replace("\r", "\n")
        chunks = re.split(r"\n\s*\n", thinking)

        if not closed and not re.search(r"\n\s*\n\s*$", thinking):
            chunks = chunks[:-1]

        blocks = []
        for chunk in chunks:
            block = chunk.strip()
            if block:
                blocks.append(block)

        if len(blocks) > self.reasoning_processed_blocks:
            new_blocks = blocks[self.reasoning_processed_blocks:]
            for block in new_blocks:
                self.reasoning_queue.put((self.reasoning_generation_id, block))
            self.reasoning_processed_blocks = len(blocks)
            self.set_reasoning_status("Summarizing...")

        if closed:
            self.reasoning_done = True
            if self.reasoning_queue.empty() and not self.reasoning_summary_active:
                self.add_reasoning_done(self.reasoning_generation_id)
            self.set_reasoning_status("Complete")

    def refresh_chat_list(self):
        for w in self.chat_list.winfo_children():
            w.destroy()
        self.chat_items = {}

        m = self.store[self.model_id]
        self.chat_ids = list(m["chat_order"])

        for cid in self.chat_ids:
            c = m["chats"].get(cid)
            if not c:
                continue

            item = ctk.CTkFrame(self.chat_list, corner_radius=14, fg_color="#2b2b2b")
            item.grid(sticky="ew", padx=10, pady=8)
            item.grid_columnconfigure(0, weight=1)

            name = c.get("name") or cid
            title = ctk.CTkLabel(item, text=name, font=ctk.CTkFont(size=15, weight="bold"))
            title.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 2))

            meta = ctk.CTkLabel(item, text=f"{len(c.get('messages') or [])} messages", text_color="#9a9a9a", font=ctk.CTkFont(size=12))
            meta.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 10))

            item.bind("<Button-1>", lambda e, chat_id=cid: self.select_chat(chat_id))
            title.bind("<Button-1>", lambda e, chat_id=cid: self.select_chat(chat_id))
            meta.bind("<Button-1>", lambda e, chat_id=cid: self.select_chat(chat_id))

            self.chat_items[cid] = item

        self.update_chat_list_selection()

    def update_chat_list_selection(self):
        for cid, frame in self.chat_items.items():
            frame.configure(fg_color="#3a3a3a" if cid == self.selected_chat_id else "#2b2b2b")

    def select_chat(self, chat_id):
        if not chat_id:
            return
        if chat_id == self.selected_chat_id:
            return

        self.selected_chat_id = chat_id
        self.update_chat_list_selection()
        self.render_messages()

    def get_chat_by_id(self, chat_id):
        m = self.store.get(self.model_id) or {}
        chats = m.get("chats") or {}
        return chats.get(chat_id)

    def get_selected_chat(self):
        return self.get_chat_by_id(self.selected_chat_id)

    def create_chat(self):
        m = self.store[self.model_id]
        chat_id = self.make_chat_id()
        m["chats"][chat_id] = self.new_chat_obj(chat_id)
        m["chat_order"].append(chat_id)
        self.save_store()
        self.refresh_chat_list()
        self.select_chat(chat_id)

    def delete_chat(self):
        if not self.selected_chat_id:
            return

        m = self.store[self.model_id]
        cid = self.selected_chat_id

        if cid == self.stream_chat_id:
            self.stop_generation = True
            if not self.stop_reason:
                self.stop_reason = "stopped"
            self.send_btn.configure(state="disabled")
            self.set_status("Stopping deleted chat...")

        if cid in m["chats"]:
            del m["chats"][cid]
        if cid in m["chat_order"]:
            m["chat_order"].remove(cid)

        if not m["chat_order"]:
            new_id = self.make_chat_id()
            m["chat_order"] = [new_id]
            m["chats"][new_id] = self.new_chat_obj(new_id)

        self.save_store()
        self.refresh_chat_list()
        self.select_chat(m["chat_order"][0])

    def clear_chat_view(self):
        for w in self.chat_view.winfo_children():
            w.destroy()
        self.msg_widgets = []

        if self.streamer is not None:
            self.stream_label = None
            self.stream_meta_label = None

    def scroll_to_bottom(self):
        return
    
    def copy_message(self, text):
        if not text:
            return
        self.app.clipboard_clear()
        self.app.clipboard_append(text)
        self.set_status("Copied")

    def bold_unicode_char(self, ch):
        if "A" <= ch <= "Z":
            return chr(ord(ch) - ord("A") + 0x1D5D4)
        if "a" <= ch <= "z":
            return chr(ord(ch) - ord("a") + 0x1D5EE)
        if "0" <= ch <= "9":
            return chr(ord(ch) - ord("0") + 0x1D7EC)
        return ch

    def bold_unicode_text(self, text):
        return "".join(self.bold_unicode_char(ch) for ch in text)

    def mono_unicode_char(self, ch):
        if "A" <= ch <= "Z":
            return chr(ord(ch) - ord("A") + 0x1D608)
        if "a" <= ch <= "z":
            return chr(ord(ch) - ord("a") + 0x1D622)
        if "0" <= ch <= "9":
            return ch
        return ch

    def mono_unicode_text(self, text):
        return "".join(self.mono_unicode_char(ch) for ch in text)

    def format_inline_markdown(self, text):
        text = re.sub(
            r"`([^`\n]+?)`",
            lambda m: self.mono_unicode_text(m.group(1)),
            text,
        )

        text = re.sub(
            r"\*\*(.+?)\*\*",
            lambda m: self.bold_unicode_text(m.group(1)),
            text,
        )

        return text

    def get_page_count(self, text):
        return max(1, (len(text) + self.message_page_chars - 1) // self.message_page_chars)

    def get_page_text(self, text, page):
        start = page * self.message_page_chars
        end = start + self.message_page_chars
        return text[start:end]

    def split_markdown_segments(self, text):
        segments = []
        text_buf = []
        code_buf = []
        in_code = False
        lang = ""

        for line in text.splitlines(True):
            stripped = line.strip()

            if stripped.startswith("```"):
                if in_code:
                    code = "".join(code_buf).rstrip("\n")
                    if code:
                        segments.append(("code", lang, code))
                    code_buf = []
                    lang = ""
                    in_code = False
                else:
                    normal = "".join(text_buf).strip("\n")
                    if normal:
                        segments.append(("text", "", normal))
                    text_buf = []
                    lang = stripped[3:].strip() or "code"
                    in_code = True
                continue

            if in_code:
                code_buf.append(line)
            else:
                text_buf.append(line)

        if in_code:
            code = "".join(code_buf).rstrip("\n")
            if code:
                segments.append(("code", lang, code))
        else:
            normal = "".join(text_buf).strip("\n")
            if normal:
                segments.append(("text", "", normal))

        return segments

    def split_latex_segments(self, text):
        segments = []
        pattern = re.compile(r"(\$\$(.+?)\$\$|\\\[(.+?)\\\])", re.DOTALL)
        pos = 0

        for m in pattern.finditer(text):
            before = text[pos:m.start()]
            if before:
                segments.append(("text", "", before))

            expr = (m.group(2) or m.group(3) or "").strip()
            if expr:
                segments.append(("latex", "", expr))

            pos = m.end()

        rest = text[pos:]
        if rest:
            segments.append(("text", "", rest))

        return segments or [("text", "", text)]

    def render_latex_image(self, expr, fg):
        key = (expr, fg)
        if key in self.latex_images:
            return self.latex_images[key]

        fig = Figure(figsize=(8, 1), dpi=160)
        fig.patch.set_alpha(0)
        canvas = FigureCanvasAgg(fig)

        fig.text(0.01, 0.5, "$" + expr + "$", fontsize=18, color=fg, va="center")
        canvas.draw()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.08, transparent=True)
        buf.seek(0)

        img = Image.open(buf).convert("RGBA")
        max_w = 620

        if img.width > max_w:
            scale = max_w / img.width
            img = img.resize((max_w, max(1, int(img.height * scale))), Image.LANCZOS)

        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.latex_images[key] = ctk_img
        return ctk_img

    def add_latex_segment(self, parent, expr, fg, row):
        expr = expr.strip()
        if not expr:
            return row

        try:
            img = self.render_latex_image(expr, fg)
        except Exception:
            return self.add_normal_text_block(parent, "\\[" + expr + "\\]", fg, row)

        lab = ctk.CTkLabel(parent, image=img, text="")
        lab.grid(row=row, column=0, sticky="w", padx=12, pady=(8, 6))
        lab.bind("<Double-Button-1>", lambda e, t=expr: self.copy_message(t))
        return row + 1

    def add_normal_text_block(self, parent, text, fg, row):
        normal = text.strip("\n")
        if not normal.strip():
            return row

        display_text = self.format_inline_markdown(normal)

        lab = ctk.CTkLabel(parent, text=display_text, text_color=fg, justify="left", wraplength=640)
        lab.grid(row=row, column=0, sticky="w", padx=12, pady=(8, 4))
        lab.bind("<Double-Button-1>", lambda e, t=normal: self.copy_message(t))
        return row + 1

    def add_heading_line(self, parent, level, text, fg, row):
        if not text.strip():
            return row

        size = 22
        if level == 2:
            size = 18
        elif level == 3:
            size = 16

        display_text = self.format_inline_markdown(text)

        lab = ctk.CTkLabel(
            parent,
            text=display_text,
            text_color=fg,
            justify="left",
            wraplength=640,
            font=ctk.CTkFont(size=size, weight="bold"),
        )
        lab.grid(row=row, column=0, sticky="w", padx=12, pady=(14, 6))
        lab.bind("<Double-Button-1>", lambda e, t=text: self.copy_message(t))
        return row + 1

    def add_horizontal_rule(self, parent, row):
        line = ctk.CTkFrame(parent, height=2, fg_color="#5a5a5a")
        line.grid(row=row, column=0, sticky="ew", padx=12, pady=12)
        return row + 1

    def parse_table_row(self, line):
        line = line.strip()

        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]

        return [cell.strip() for cell in line.split("|")]

    def is_table_row_line(self, line):
        if "|" not in line:
            return False

        cells = self.parse_table_row(line)
        return len(cells) >= 2

    def is_table_separator_line(self, line):
        if not self.is_table_row_line(line):
            return False

        cells = self.parse_table_row(line)

        for cell in cells:
            test = cell.replace("-", "").replace(":", "").strip()
            if test:
                return False

        return True

    def add_table_segment(self, parent, lines, fg, row):
        rows = []

        for i, line in enumerate(lines):
            if i == 1 and self.is_table_separator_line(line):
                continue
            rows.append(self.parse_table_row(line))

        if not rows:
            return row

        col_count = max(len(r) for r in rows)
        if col_count < 2:
            return row

        table = ctk.CTkFrame(parent, corner_radius=10, fg_color="#202020")
        table.grid(row=row, column=0, sticky="ew", padx=12, pady=(8, 8))

        for c in range(col_count):
            table.grid_columnconfigure(c, weight=1)

        wrap = max(90, int(580 / col_count))

        for r, cells in enumerate(rows):
            for c in range(col_count):
                text = cells[c] if c < len(cells) else ""
                bg = "#303030" if r == 0 else "#262626"

                cell = ctk.CTkFrame(table, corner_radius=4, fg_color=bg)
                cell.grid(row=r, column=c, sticky="nsew", padx=1, pady=1)
                cell.grid_columnconfigure(0, weight=1)

                display_text = self.format_inline_markdown(text)

                lab = ctk.CTkLabel(
                    cell,
                    text=display_text,
                    text_color=fg,
                    justify="left",
                    anchor="w",
                    wraplength=wrap,
                    font=ctk.CTkFont(size=13, weight="bold" if r == 0 else "normal"),
                )
                lab.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
                lab.bind("<Double-Button-1>", lambda e, t=text: self.copy_message(t))

        return row + 1

    def add_text_segment(self, parent, text, fg, row):
        if not text.strip():
            return row

        lines = text.splitlines()
        normal_buf = []
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            is_table = (
                i + 1 < len(lines)
                and self.is_table_row_line(line)
                and self.is_table_separator_line(lines[i + 1])
            )

            if is_table:
                normal = "\n".join(normal_buf).strip("\n")
                if normal:
                    row = self.add_normal_text_block(parent, normal, fg, row)
                    normal_buf = []

                table_lines = [lines[i], lines[i + 1]]
                i += 2

                while i < len(lines) and self.is_table_row_line(lines[i]):
                    table_lines.append(lines[i])
                    i += 1

                row = self.add_table_segment(parent, table_lines, fg, row)
                continue

            heading_match = re.match(r"^(#{1,})\s+(.+)$", stripped)
            heading_level = 0
            heading_text = ""

            if heading_match:
                heading_level = min(len(heading_match.group(1)), 3)
                heading_text = heading_match.group(2).strip()

            is_bullet = stripped.startswith("- ") or stripped.startswith("* ")
            is_rule = stripped == "---"
            is_heading = heading_level > 0

            if is_bullet or is_rule or is_heading:
                normal = "\n".join(normal_buf).strip("\n")
                if normal:
                    row = self.add_normal_text_block(parent, normal, fg, row)
                    normal_buf = []

            if is_rule:
                row = self.add_horizontal_rule(parent, row)
                i += 1
                continue

            if is_heading:
                row = self.add_heading_line(parent, heading_level, heading_text, fg, row)
                i += 1
                continue

            if is_bullet:
                bullet_text = stripped[2:].strip()

                bullet_row = ctk.CTkFrame(parent, fg_color="transparent")
                bullet_row.grid(row=row, column=0, sticky="ew", padx=12, pady=2)
                bullet_row.grid_columnconfigure(1, weight=1)

                dot = ctk.CTkLabel(bullet_row, text="•", text_color=fg, width=18)
                dot.grid(row=0, column=0, sticky="nw", padx=(0, 4))

                bullet_display = self.format_inline_markdown(bullet_text)

                lab = ctk.CTkLabel(bullet_row, text=bullet_display, text_color=fg, justify="left", wraplength=600)
                lab.grid(row=0, column=1, sticky="w")
                lab.bind("<Double-Button-1>", lambda e, t=bullet_text: self.copy_message(t))

                row += 1
                i += 1
                continue

            normal_buf.append(line)
            i += 1

        normal = "\n".join(normal_buf).strip("\n")
        if normal:
            row = self.add_normal_text_block(parent, normal, fg, row)

        return row

    def add_code_segment(self, parent, lang, code, row):
        frame = ctk.CTkFrame(parent, corner_radius=12, fg_color="#151515")
        frame.grid(row=row, column=0, sticky="ew", padx=10, pady=8)
        frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            frame,
            text=lang or "code",
            text_color="#a8a8a8",
            font=ctk.CTkFont(size=12),
        )
        header.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))

        lines = max(1, code.count("\n") + 1)
        height = min(max(70, lines * 18 + 20), 420)

        box = ctk.CTkTextbox(
            frame,
            width=620,
            height=height,
            fg_color="#0b0b0b",
            text_color="#e6e6e6",
            font=ctk.CTkFont(family="monospace", size=13),
            wrap="none",
            border_width=0,
        )
        box.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        box.insert("1.0", code)
        box.configure(state="disabled")

        frame.bind("<Double-Button-1>", lambda e, t=code: self.copy_message(t))
        header.bind("<Double-Button-1>", lambda e, t=code: self.copy_message(t))
        box.bind("<Double-Button-1>", lambda e, t=code: self.copy_message(t))

        return row + 1

    def render_message_content(self, parent, text, fg):
        for w in parent.winfo_children():
            w.destroy()

        segments = self.split_markdown_segments(text)
        row = 0

        if not segments:
            row = self.add_text_segment(parent, text, fg, row)

        for kind, lang, content in segments:
            if kind == "code":
                row = self.add_code_segment(parent, lang, content, row)
            else:
                for sub_kind, _, sub_content in self.split_latex_segments(content):
                    if sub_kind == "latex":
                        row = self.add_latex_segment(parent, sub_content, fg, row)
                    else:
                        row = self.add_text_segment(parent, sub_content, fg, row)

    def set_message_page(self, page_frame, counter, prev_btn, next_btn, text, page_state, page, fg):
        page_count = self.get_page_count(text)
        page_state["page"] = max(0, min(page, page_count - 1))

        page_text = self.get_page_text(text, page_state["page"])
        self.render_message_content(page_frame, page_text, fg)

        counter.configure(text=f"Page {page_state['page'] + 1} / {page_count}")

        prev_btn.configure(state="normal" if page_state["page"] > 0 else "disabled")
        next_btn.configure(state="normal" if page_state["page"] < page_count - 1 else "disabled")

    def next_message_page(self, page_frame, counter, prev_btn, next_btn, text, page_state, fg):
        self.set_message_page(page_frame, counter, prev_btn, next_btn, text, page_state, page_state["page"] + 1, fg)

    def prev_message_page(self, page_frame, counter, prev_btn, next_btn, text, page_state, fg):
        self.set_message_page(page_frame, counter, prev_btn, next_btn, text, page_state, page_state["page"] - 1, fg)

    def add_bubble(self, role, text):
        row = ctk.CTkFrame(self.chat_view, fg_color="transparent")
        row.grid(sticky="ew", padx=10, pady=6)
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)

        is_user = role == "user"
        bg = "#1f6feb" if is_user else "#3a3a3a"
        fg = "#ffffff" if is_user else "#e6e6e6"

        bubble = ctk.CTkFrame(row, corner_radius=16, fg_color=bg)
        col = 1 if is_user else 0
        sticky = "e" if is_user else "w"
        bubble.grid(row=0, column=col, sticky=sticky, padx=6)
        bubble.grid_columnconfigure(0, weight=1)

        bubble.bind("<Double-Button-1>", lambda e, t=text: self.copy_message(t))

        if is_user:
            lab = ctk.CTkLabel(bubble, text=text, text_color=fg, justify="left", wraplength=640)
            lab.grid(row=0, column=0, sticky="w", padx=12, pady=10)
            lab.bind("<Double-Button-1>", lambda e, t=text: self.copy_message(t))
            self.msg_widgets.append((row, bubble, lab))
            return

        display_text = self.get_display_message_text(text, streaming=False)
        bubble.bind("<Double-Button-1>", lambda e, t=display_text: self.copy_message(t))

        if len(display_text) <= self.message_page_chars:
            content = ctk.CTkFrame(bubble, fg_color="transparent")
            content.grid(row=0, column=0, sticky="ew", padx=0, pady=2)
            content.grid_columnconfigure(0, weight=1)

            self.render_message_content(content, display_text, fg)
            self.msg_widgets.append((row, bubble, content))
            return

        page_state = {"page": 0}

        page_frame = ctk.CTkFrame(bubble, fg_color="transparent")
        page_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=2)
        page_frame.grid_columnconfigure(0, weight=1)

        nav = ctk.CTkFrame(bubble, fg_color="transparent")
        nav.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        nav.grid_columnconfigure(1, weight=1)

        prev_btn = ctk.CTkButton(nav, text="Previous", width=90)
        prev_btn.grid(row=0, column=0, sticky="w", padx=(0, 8))

        counter = ctk.CTkLabel(nav, text="Page - / -", text_color="#a8a8a8")
        counter.grid(row=0, column=1)

        next_btn = ctk.CTkButton(nav, text="Next", width=90)
        next_btn.grid(row=0, column=2, sticky="e", padx=(8, 0))

        prev_btn.configure(command=lambda: self.prev_message_page(page_frame, counter, prev_btn, next_btn, display_text, page_state, fg))
        next_btn.configure(command=lambda: self.next_message_page(page_frame, counter, prev_btn, next_btn, display_text, page_state, fg))

        self.set_message_page(page_frame, counter, prev_btn, next_btn, display_text, page_state, 0, fg)
        self.msg_widgets.append((row, bubble, page_frame))

    def render_messages(self):
        self.clear_chat_view()

        chat = self.get_selected_chat()
        if not chat:
            return

        msgs = chat.get("messages") or []
        for m in msgs:
            self.add_bubble(m.get("role"), m.get("content") or "")

        self.scroll_to_bottom()

    def append_message(self, role, content):
        chat = self.get_selected_chat()
        if not chat:
            return

        chat["messages"].append({
            "role": role,
            "content": content,
            "ts": int(time.time()),
        })
        self.save_store()
        self.add_bubble(role, content)
        self.scroll_to_bottom()
        self.refresh_chat_list()

    def toggle_voice_recording(self):
        if self.voice_recording:
            self.stop_voice_recording()
        else:
            self.start_voice_recording()

    def voice_callback(self, indata, frames, time_info, status):
        self.voice_chunks.append(indata.copy())

    def start_voice_recording(self):
        self.voice_chunks = []
        self.voice_recording = True
        self.voice_stream = sd.InputStream(
            samplerate=self.voice_sample_rate,
            channels=1,
            dtype="float32",
            callback=self.voice_callback,
        )
        self.voice_stream.start()

        if self.mic_btn is not None:
            self.mic_btn.configure(text="Stop", fg_color="#aa3333", hover_color="#883333")

        self.set_status("Recording...")

    def stop_voice_recording(self):
        self.voice_recording = False

        if self.voice_stream is not None:
            self.voice_stream.stop()
            self.voice_stream.close()
            self.voice_stream = None

        if self.mic_btn is not None:
            self.mic_btn.configure(state="disabled", text="...")

        self.set_status("Transcribing...")
        threading.Thread(target=self.transcribe_voice_recording, daemon=True).start()

    def load_voice_model(self):
        with self.voice_lock:
            if self.voice_pipe is not None:
                return self.voice_pipe
            self.voice_loading = True

        p = None
        device_name = "CPU"

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.app.after(0, self.set_status, "Loading voice model on GPU 1...")

            try:
                p = pipeline(
                    "automatic-speech-recognition",
                    model=self.voice_model_id,
                    device=1,
                    torch_dtype=torch.bfloat16,
                )
                device_name = "GPU 1"
            except Exception:
                p = None
                self.clear_cuda_cache()

        if p is None:
            self.app.after(0, self.set_status, "Loading voice model on CPU...")

            p = pipeline(
                "automatic-speech-recognition",
                model=self.voice_model_id,
                device=-1,
                torch_dtype=torch.float32,
            )
            device_name = "CPU"

        with self.voice_lock:
            self.voice_pipe = p
            self.voice_device = device_name
            self.voice_loading = False

        return p

    def transcribe_voice_recording(self):
        if not self.voice_chunks:
            self.app.after(0, self.finish_voice_transcription, "")
            return

        audio = np.concatenate(self.voice_chunks, axis=0)

        if audio.ndim == 2:
            audio = audio[:, 0]

        audio = audio.astype(np.float32)

        p = self.load_voice_model()
        self.app.after(0, self.set_status, f"Transcribing on {self.voice_device or 'CPU'}...")

        result = p(
            {
                "array": audio,
                "sampling_rate": self.voice_sample_rate,
            },
            return_timestamps=True,
        )

        text = ""
        if isinstance(result, dict):
            text = (result.get("text") or "").strip()

        self.app.after(0, self.finish_voice_transcription, text)

    def finish_voice_transcription(self, text):
        if self.mic_btn is not None:
            self.mic_btn.configure(state="normal", text="🎙", fg_color="#444444", hover_color="#3a3a3a")

        if not text:
            self.set_status("No speech detected")
            return

        current = self.input_box.get("1.0", "end").strip()
        if current:
            self.input_box.insert("end", " " + text)
        else:
            self.input_box.insert("end", text)

        self.set_status("Transcribed")
    
    def widget_inside(self, w, parent):
        if isinstance(w, str):
            try:
                w = self.app.nametowidget(w)
            except:
                return False

        while w:
            if w == parent:
                return True
            w = getattr(w, "master", None)

        return False

    def on_mousewheel(self, event):
        canvas = None

        if self.chat_view and hasattr(self.chat_view, "_parent_canvas"):
            if self.widget_inside(event.widget, self.chat_view._parent_canvas):
                canvas = self.chat_view._parent_canvas

        if not canvas and self.chat_list and hasattr(self.chat_list, "_parent_canvas"):
            if self.widget_inside(event.widget, self.chat_list._parent_canvas):
                canvas = self.chat_list._parent_canvas

        if not canvas and self.reasoning_view and hasattr(self.reasoning_view, "_parent_canvas"):
            if self.widget_inside(event.widget, self.reasoning_view._parent_canvas):
                canvas = self.reasoning_view._parent_canvas

        if not canvas:
            return

        if getattr(event, "num", None) == 4:
            canvas.yview_scroll(-1, "units")
            return
        if getattr(event, "num", None) == 5:
            canvas.yview_scroll(1, "units")
            return

        delta = getattr(event, "delta", 0)
        if delta:
            canvas.yview_scroll(-1 if delta > 0 else 1, "units")

    def set_status(self, text):
        self.status_pill.configure(text=text)

    def set_send_enabled(self, enabled):
        self.send_btn.configure(state="normal" if enabled else "disabled")

    async def create_vllm_engine(self):
        config = VLLM_MODEL_CONFIGS[self.model_id]
        engine_args = AsyncEngineArgs(
            model=self.model_id,
            **config,
        )
        return AsyncLLM.from_engine_args(engine_args)

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        self.app.after(0, self.set_status, "Loading...")
        self.app.after(0, self.set_send_enabled, False)

        if self.model_id in VLLM_MODEL_CONFIGS:
            self.vllm_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.vllm_loop)
            p = self.vllm_loop.run_until_complete(self.create_vllm_engine())
            config = VLLM_MODEL_CONFIGS[self.model_id]
            tokenizer_id = config.get("tokenizer", self.model_id)
            self.vllm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        else:
            model_kwargs = {
                "dtype": torch.bfloat16,
                "device_map": "balanced",
                "max_memory": {
                    0: "22GiB",
                    1: "22GiB",
                },
            }

            if self.model_id in ("LiquidAI/LFM2.5-1.2B-Thinking", "openai/gpt-oss-20b"):
                model_kwargs["device_map"] = {"": "cuda:0"}

            if self.model_id in CHAT_MODEL_CPU_OFFLOAD_IDS:
                model_kwargs["max_memory"]["cpu"] = "80GiB"

            quantization = CHAT_MODEL_QUANTIZATION.get(self.model_id)

            if quantization == "8bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            elif quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            pipeline_task = "image-text-to-text" if self.model_id in PROCESSOR_CHAT_MODEL_IDS else "text-generation"

            p = pipeline(
                pipeline_task,
                model=self.model_id,
                model_kwargs=model_kwargs,
            )

        with self.model_lock:
            self.pipe = p
            self.model_loading = False

        if self.reasoning_enabled:
            self.app.after(0, self.set_status, "Loading reasoning...")
            self.load_reasoning_model()

        self.app.after(0, self.set_status, "Ready")
        self.app.after(0, self.set_send_enabled, True)

        if self.vllm_loop is not None:
            self.vllm_loop.run_forever()

    def load_reasoning_model(self):
        if not self.reasoning_enabled:
            return

        with self.reasoning_lock:
            if self.reasoning_pipe is not None:
                return
            self.reasoning_loading = True

        self.app.after(0, self.set_reasoning_status, "Loading...")

        if (
            self.model_id in VLLM_MODEL_CONFIGS
            or self.model_id == GEMMA_4_MODEL_ID
            or self.model_id == "Qwen/Qwen3.6-27B"
            or self.model_id == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        ):
            p = pipeline(
                "text-generation",
                model=REASONING_SUMMARY_CPU_MODEL_ID,
                model_kwargs={"torch_dtype": torch.float32},
                device=-1,
            )
        else:
            try:
                p = pipeline(
                    "text-generation",
                    model=self.reasoning_model_id,
                    model_kwargs={
                        "device_map": {"": "cuda:0"},
                        "quantization_config": BitsAndBytesConfig(
                            load_in_8bit=True,
                        ),
                    },
                )
            except torch.OutOfMemoryError:
                print("Loading reasoning model on CPU")
                p = pipeline(
                    "text-generation",
                    model=REASONING_SUMMARY_CPU_MODEL_ID,
                    model_kwargs={"torch_dtype": torch.float32},
                    device=-1,
                )

        p.generation_config.max_length = None
        p.generation_config.max_new_tokens = 32
        p.generation_config.do_sample = False

        tokenizer = getattr(p, "tokenizer", None)
        if tokenizer is not None and tokenizer.eos_token_id is not None:
            p.generation_config.pad_token_id = tokenizer.eos_token_id

        with self.reasoning_lock:
            self.reasoning_pipe = p
            self.reasoning_loading = False

        if not self.reasoning_worker_started:
            threading.Thread(target=self.reasoning_worker, daemon=True).start()
            self.reasoning_worker_started = True

        self.app.after(0, self.set_reasoning_status, "Ready")

    def clean_reasoning_summary(self, text):
        text = (text or "").strip()
        text = re.sub(r"^(step\s+name|short\s+summary|summary)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"[*_`#>-]+", "", text).strip()
        text = text.replace('"', "").replace("'", "").strip()
        text = text.split("\n")[0].strip()

        m = re.search(r"^(.+?[.!?])(?:\s|$)", text)
        if m:
            text = m.group(1).strip()

        words = text.split()
        if len(words) > 12:
            text = " ".join(words[:12]).rstrip(".,;:") + "."

        text = re.sub(r"\b(and|or|with|for|to|of|about|including)$", "", text, flags=re.IGNORECASE).strip()
        text = text.rstrip(".,;:") + "."

        if len(text) > 110:
            text = text[:110].rsplit(" ", 1)[0].rstrip(".,;:") + "."

        return text or "(summary unavailable)"

    def summarize_reasoning_block(self, block):
        with self.reasoning_lock:
            p = self.reasoning_pipe

        if p is None:
            return "(summary model unavailable)"

        prompt = (
            "Summarize into 5 to 7 words.\n\n"
            "Text:\n"
            + block.strip()
            + "\n\nSummary:"
        )

        kwargs = {
            "return_full_text": False,
            "clean_up_tokenization_spaces": False,
        }

        out = p(prompt, **kwargs)

        text = ""
        if out and isinstance(out, list) and out[0] and isinstance(out[0], dict):
            text = out[0].get("generated_text") or ""

        return self.clean_reasoning_summary(text)

    def reasoning_worker(self):
        while True:
            generation_id, block = self.reasoning_queue.get()

            if generation_id != self.reasoning_generation_id:
                self.reasoning_queue.task_done()
                continue

            self.reasoning_summary_active = True

            blocks = [block]
            self.reasoning_queue.task_done()

            while not self.reasoning_queue.empty():
                next_generation_id, next_block = self.reasoning_queue.get_nowait()

                if next_generation_id == generation_id:
                    blocks.append(next_block)

                self.reasoning_queue.task_done()

            block = "\n\n".join(blocks)
            summary = self.summarize_reasoning_block(block)
            self.reasoning_summary_active = False

            self.app.after(0, self.add_reasoning_summary, summary, generation_id)

            if generation_id == self.reasoning_generation_id and self.reasoning_done and self.reasoning_queue.empty():
                self.app.after(0, self.add_reasoning_done, generation_id)

    def ensure_model_loaded(self):
        with self.model_lock:
            ready = self.pipe is not None
            loading = self.model_loading

        if ready:
            return

        if loading:
            while True:
                with self.model_lock:
                    ready = self.pipe is not None
                    loading = self.model_loading
                if ready or not loading:
                    break
                time.sleep(0.2)
            return

        self.load_model()

    def build_context_messages(self, chat_id=None):
        if chat_id:
            chat = self.get_chat_by_id(chat_id)
        else:
            chat = self.get_selected_chat()

        msgs = (chat.get("messages") or []) if chat else []
        tail = msgs[-self.max_context_messages:] if len(msgs) > self.max_context_messages else msgs

        out = [{"role": "system", "content": self.system_prompt}]
        for m in tail:
            role = m.get("role") or ""
            content = m.get("content") or ""
            if role == "assistant" and self.model_id == GEMMA_4_MODEL_ID:
                content = self.get_display_message_text(content, streaming=False)
            if role in ("user", "assistant") and content.strip():
                out.append({"role": role, "content": content})
        return out

    def format_plain_prompt(self, messages):
        lines = []
        for m in messages:
            r = m.get("role")
            c = (m.get("content") or "").strip()
            if not c:
                continue
            if r == "system":
                lines.append(c)
            elif r == "user":
                lines.append("User: " + c)
            elif r == "assistant":
                lines.append("Assistant: " + c)
        lines.append("Assistant:")
        return "\n".join(lines)

    def build_vllm_prompt(self, messages):
        return self.vllm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    async def collect_vllm_reply(self, messages):
        prompt = self.build_vllm_prompt(messages)
        request_id = "chat-" + str(time.time_ns()) + "-" + uuid.uuid4().hex[:8]
        sampling_params = SamplingParams(
            max_tokens=min(int(self.max_new_tokens), 4096),
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

        reply = ""
        async for output in self.pipe.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            if output.outputs:
                reply = output.outputs[0].text

        return reply.strip()

    async def stream_vllm_reply(self, messages, streamer):
        prompt = self.build_vllm_prompt(messages)
        request_id = "chat-" + str(time.time_ns()) + "-" + uuid.uuid4().hex[:8]
        self.vllm_request_id = request_id

        sampling_params = SamplingParams(
            max_tokens=min(int(self.max_new_tokens), 4096),
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            output_kind=RequestOutputKind.DELTA,
        )

        try:
            async for output in self.pipe.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                if self.stop_generation:
                    if not self.stop_reason:
                        self.stop_reason = "stopped"
                    await self.pipe.abort(request_id)
                    break

                if self.max_generation_seconds and self.stream_t0:
                    if time.time() - self.stream_t0 >= self.max_generation_seconds:
                        self.stop_reason = "time"
                        await self.pipe.abort(request_id)
                        break

                for completion in output.outputs:
                    if completion.text:
                        streamer.put(completion.text)

                if output.finished:
                    finish_reason = output.outputs[0].finish_reason if output.outputs else None
                    self.stop_reason = "tokens" if finish_reason == "length" else "complete"
                    break
        finally:
            self.vllm_request_id = None
            streamer.end()

    def generate_reply(self):
        self.ensure_model_loaded()

        with self.model_lock:
            p = self.pipe

        if p is None:
            return ""

        messages = self.build_context_messages()

        if self.model_id in VLLM_MODEL_CONFIGS:
            future = asyncio.run_coroutine_threadsafe(
                self.collect_vllm_reply(messages),
                self.vllm_loop,
            )
            return future.result()

        prompt = self.format_plain_prompt(messages)

        out = p(
            prompt,
            max_new_tokens=int(self.max_new_tokens),
            do_sample=True,
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            return_full_text=False,
        )

        text = ""
        if out and isinstance(out, list) and out[0] and isinstance(out[0], dict):
            text = (out[0].get("generated_text") or "").strip()

        return text

    def send_message(self, event=None):
        text = self.input_box.get("1.0", "end").strip()
        if not text:
            return

        chat_id = self.selected_chat_id

        self.input_box.delete("1.0", "end")
        self.append_message("user", text)

        self.stop_generation = False
        self.stop_reason = ""
        self.stream_chat_id = chat_id
        self.reset_reasoning_summary()
        self.set_status("Generating...")
        self.send_btn.configure(text="Cancel", state="normal", command=self.stop_message)
        threading.Thread(target=self.run_reply, args=(chat_id,), daemon=True).start()

    def finish_send(self, reply, dt, chat_id=None):
        self.append_message_only("assistant", reply, chat_id)

        if chat_id is None or chat_id == self.selected_chat_id:
            self.render_messages()

        self.send_btn.configure(text="Send", state="normal", command=self.send_message)
        self.set_status(f"Ready ({dt:.1f}s)")

    def run_reply(self, chat_id):
        self.ensure_model_loaded()
        with self.model_lock:
            p = self.pipe

        if p is None:
            self.app.after(0, self.finish_send, "(no response)", 0.0, chat_id)
            return

        messages = self.build_context_messages(chat_id)

        if self.model_id in VLLM_MODEL_CONFIGS:
            streamer = QueueTextStreamer()
            self.streamer = streamer
            self.stream_text = ""
            self.stream_done = False
            self.stream_label = None
            self.stream_meta_label = None
            self.stream_chat_id = chat_id

            self.app.after(0, self.create_stream_bubble)
            while self.stream_label is None and not self.stream_done and self.stream_chat_id == self.selected_chat_id:
                time.sleep(0.02)

            self.stream_t0 = time.time()
            self.app.after(100, self.poll_streamer)

            future = asyncio.run_coroutine_threadsafe(
                self.stream_vllm_reply(messages, streamer),
                self.vllm_loop,
            )
            future.result()
            return

        tokenizer = getattr(p, "tokenizer", None)
        model = getattr(p, "model", None)
        if tokenizer is None or model is None:
            self.app.after(0, self.finish_send, "(no response)", 0.0, chat_id)
            return

        processor = getattr(p, "processor", None)

        if self.model_id in PROCESSOR_CHAT_MODEL_IDS:
            processor_tokenizer = getattr(processor, "tokenizer", None)
            if processor_tokenizer is not None:
                tokenizer = processor_tokenizer

        streamer_source = processor if self.model_id == GEMMA_4_MODEL_ID and processor is not None else tokenizer
        streamer = TextIteratorStreamer(
            streamer_source,
            skip_special_tokens=self.model_id != GEMMA_4_MODEL_ID,
            skip_prompt=True,
        )
        self.streamer = streamer
        self.stream_text = ""
        self.stream_done = False
        self.stream_label = None
        self.stream_meta_label = None
        self.stream_chat_id = chat_id
        stopping_criteria = StoppingCriteriaList([StopGenerationCriteria(self)])

        self.app.after(0, self.create_stream_bubble)
        while self.stream_label is None and not self.stream_done and self.stream_chat_id == self.selected_chat_id:
            time.sleep(0.02)

        if self.model_id == GEMMA_4_MODEL_ID:
            template_processor = processor if processor is not None else tokenizer
            prompt = template_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = template_processor(
                text=prompt,
                return_tensors="pt",
            )
        elif self.model_id == "Qwen/Qwen3.6-27B":
            template_processor = processor if processor is not None else tokenizer
            inputs = template_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=True,
            )
        else:
            prompt = self.format_plain_prompt(messages)
            inputs = tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        self.stream_t0 = time.time()
        self.app.after(100, self.poll_streamer)

        pad_id = tokenizer.eos_token_id
        generate_kwargs = {
            **inputs,
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": True,
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "pad_token_id": pad_id,
            "streamer": streamer,
            "stopping_criteria": stopping_criteria,
        }

        output_ids = model.generate(**generate_kwargs)

        generated_tokens = int(output_ids.shape[-1] - inputs["input_ids"].shape[-1])

        if not self.stop_reason:
            if generated_tokens >= int(self.max_new_tokens):
                self.stop_reason = "tokens"
            else:
                self.stop_reason = "complete"

    def clear_cuda_cache(self):
        gc.collect()
        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def run(self):
        self.app.mainloop()

if __name__ == "__main__":
    model_id = get_flag_value("--model-id") or get_flag_value("--model") or ""
    ChatGUI(model_id).run()
