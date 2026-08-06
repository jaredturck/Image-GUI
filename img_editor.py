import os
import sys
import time
import math
import shutil
import subprocess
import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageOps

MODEL_RESOLUTIONS = {
    "kandinsky_5": ["1280x768", "1024x1024"],
    "kandinsky_5_i2i": ["1280x768", "1024x1024"],
    "chronoedit": ["1280x720", "720x1280", "960x960", "1024x1024"],
    "rmbg_1_4": ["1024x1024"],

    "skyreels_v2": ["960x544", "1280x720"],
    "kandinsky_5_t2v": ["768x512"],
    "nvidia_cosmos": ["1280x704", "832x480"],
    "allegro": ["1280x720"],
    "cogvideox": ["720x480"],
    "hunyuan_video_1_5": ["1280x720"],
    "hunyuan_video_1_5_i2v": ["1280x720"],
    "kandinsky_5_t2v_pro": ["1024x768"],
    "kandinsky_5_t2v_pro_sft": ["1024x768"],
    "kandinsky_5_i2v": ["768x512", "512x512"],
    "kandinsky_5_i2v_pro_sft": ["512x512", "640x640"],
}

MODEL_NAMES = {
    "kandinsky_5": "Kandinsky 5",
    "kandinsky_5_i2i": "Kandinsky 5 I2I",
    "chronoedit": "ChronoEdit (14B)",
    "rmbg_1_4": "BRIA RMBG 1.4 (Remove BG)",

    "skyreels_v2": "SkyReels V2",
    "kandinsky_5_t2v": "Kandinsky 5 T2V (Lite distilled16)",
    "nvidia_cosmos": "Cosmos Predict2 V2W (2B)",
    "allegro": "Allegro (T2V)",
    "cogvideox": "CogVideoX 5B",
    "hunyuan_video_1_5": "Hunyuan Video 1.5 (720p T2V)",
    "hunyuan_video_1_5_i2v": "Hunyuan Video 1.5 (720p I2V)",
    "kandinsky_5_t2v_pro": "Kandinsky 5 T2V (Pro distilled 5s)",
    "kandinsky_5_t2v_pro_sft": "Kandinsky 5 T2V (Pro sft 5s)",
    "kandinsky_5_i2v": "Kandinsky 5 I2V (Lite 5s)",
    "kandinsky_5_i2v_pro_sft": "Kandinsky 5 I2V Pro (SFT 5s)",
}

MODEL_IDS_BY_NAME = {name: model_id for model_id, name in MODEL_NAMES.items()}

MODEL_KINDS = {
    "kandinsky_5": "image",
    "kandinsky_5_i2i": "image",
    "chronoedit": "image",
    "rmbg_1_4": "image",

    "skyreels_v2": "video",
    "kandinsky_5_t2v": "video",
    "nvidia_cosmos": "video",
    "allegro": "video",
    "cogvideox": "video",
    "hunyuan_video_1_5": "video",
    "hunyuan_video_1_5_i2v": "video",
    "kandinsky_5_t2v_pro": "video",
    "kandinsky_5_t2v_pro_sft": "video",
    "kandinsky_5_i2v": "video",
    "kandinsky_5_i2v_pro_sft": "video",
}

PLACEHOLDER = "-----"
CROP_MODES = ["Center crop", "Smart crop", "Face crop"]

def get_flag_value(flag):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

class ImageEditorGUI:
    def __init__(self, initial_model_id="", initial_source_image_path="", output_folder_path=""):
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.source_image_path = ""
        self.source_image_pil = None
        self.source_image_ctk = None
        self.output_image_pil = None
        self.output_image_ctk = None
        self.last_export_path = ""
        self.output_folder_path = os.path.abspath(output_folder_path) if output_folder_path else ""

        self.selected_model_kind = "image"
        self.selected_model_id = ""
        self.selected_resolution = ""
        self.selected_crop_mode = CROP_MODES[0]

        self.app = ctk.CTk()
        self.app.title("Image Resolution Editor")
        self.app.geometry("1200x800")
        self.app.grid_columnconfigure(0, weight=1, minsize=520)
        self.app.grid_columnconfigure(1, weight=2, minsize=620)
        self.app.grid_rowconfigure(0, weight=1)

        self.build_left_panel()
        self.build_right_panel()
        self.update_resolution_menu()
        self.update_output_info()

        if self.output_folder_path:
            os.makedirs(self.output_folder_path, exist_ok=True)
            self.open_folder_btn.configure(state="normal")

        if initial_source_image_path:
            self.set_source_image(initial_source_image_path)

        self.select_initial_model(initial_model_id)

    def build_left_panel(self):
        left = ctk.CTkScrollableFrame(self.app, corner_radius=16)
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left.grid_columnconfigure(0, weight=1)
        self.left_panel = left

        source_frame = ctk.CTkFrame(left, corner_radius=16)
        source_frame.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))
        source_frame.grid_columnconfigure(0, weight=1)

        source_header = ctk.CTkFrame(source_frame, fg_color="transparent")
        source_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        source_header.grid_columnconfigure(0, weight=1)

        source_title = ctk.CTkLabel(source_header, text="Source image", font=ctk.CTkFont(size=16, weight="bold"))
        source_title.grid(row=0, column=0, sticky="w")

        source_btn = ctk.CTkButton(source_header, text="Choose file", width=110, command=self.choose_source_image)
        source_btn.grid(row=0, column=1, sticky="e")

        self.source_path_label = ctk.CTkLabel(source_frame, text="No file selected", text_color="#a0a0a0", anchor="w")
        self.source_path_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))

        self.source_img_label = ctk.CTkLabel(source_frame, text="No source image", height=260, corner_radius=12, fg_color="#1a1a1a")
        self.source_img_label.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        settings_frame = ctk.CTkFrame(left, corner_radius=16)
        settings_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))
        settings_frame.grid_columnconfigure(0, weight=1)

        settings_title = ctk.CTkLabel(settings_frame, text="Settings", font=ctk.CTkFont(size=16, weight="bold"))
        settings_title.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        kind_row = ctk.CTkFrame(settings_frame, fg_color="transparent")
        kind_row.grid(row=1, column=0, sticky="ew", padx=10, pady=6)
        kind_row.grid_columnconfigure(0, weight=1)

        self.model_kind_tabs = ctk.CTkSegmentedButton(
            kind_row,
            values=["Images", "Videos"],
            command=self.on_model_kind_changed,
            corner_radius=999,
            selected_color="#1f6aa5",
            selected_hover_color="#1f6aa5",
            unselected_color="#4a4a4a",
            unselected_hover_color="#555555",
        )
        self.model_kind_tabs.grid(row=0, column=0)
        self.model_kind_tabs.set("Images")

        model_row = ctk.CTkFrame(settings_frame, fg_color="transparent")
        model_row.grid(row=2, column=0, sticky="ew", padx=10, pady=6)
        model_row.grid_columnconfigure(1, weight=1)

        model_label = ctk.CTkLabel(model_row, text="Model")
        model_label.grid(row=0, column=0, sticky="w")

        self.model_menu = ctk.CTkOptionMenu(model_row, values=self.get_model_names_for_kind(), command=self.on_model_changed)
        self.model_menu.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.model_menu.set(PLACEHOLDER)

        resolution_row = ctk.CTkFrame(settings_frame, fg_color="transparent")
        resolution_row.grid(row=3, column=0, sticky="ew", padx=10, pady=6)
        resolution_row.grid_columnconfigure(1, weight=1)

        resolution_label = ctk.CTkLabel(resolution_row, text="Resolution")
        resolution_label.grid(row=0, column=0, sticky="w")

        self.resolution_menu = ctk.CTkOptionMenu(resolution_row, values=[PLACEHOLDER], command=self.on_resolution_changed)
        self.resolution_menu.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.resolution_menu.set(PLACEHOLDER)

        crop_row = ctk.CTkFrame(settings_frame, fg_color="transparent")
        crop_row.grid(row=4, column=0, sticky="ew", padx=10, pady=(6, 10))
        crop_row.grid_columnconfigure(1, weight=1)

        crop_label = ctk.CTkLabel(crop_row, text="Crop mode")
        crop_label.grid(row=0, column=0, sticky="w")

        self.crop_mode_menu = ctk.CTkOptionMenu(crop_row, values=CROP_MODES, command=self.on_crop_mode_changed)
        self.crop_mode_menu.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.crop_mode_menu.set(self.selected_crop_mode)

        info_frame = ctk.CTkFrame(left, corner_radius=16)
        info_frame.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 14))
        info_frame.grid_columnconfigure(0, weight=1)

        info_title = ctk.CTkLabel(info_frame, text="How it works", font=ctk.CTkFont(size=14, weight="bold"), text_color="#3a9cff")
        info_title.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))

        info_text = ctk.CTkLabel(
            info_frame,
            text="Choose an image, select a model, then select one of its official supported resolutions. The source image is never modified.",
            justify="left",
            wraplength=440,
        )
        info_text.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

    def build_right_panel(self):
        right = ctk.CTkFrame(self.app, corner_radius=16)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        self.right_panel = right

        preview_header = ctk.CTkFrame(right, fg_color="transparent")
        preview_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        preview_header.grid_columnconfigure(0, weight=1)

        preview_title = ctk.CTkLabel(preview_header, text="Output preview", font=ctk.CTkFont(size=18, weight="bold"))
        preview_title.grid(row=0, column=0, sticky="w")

        self.output_info_label = ctk.CTkLabel(preview_header, text="Output: -", text_color="#a0a0a0")
        self.output_info_label.grid(row=0, column=1, sticky="e", padx=(0, 8))

        self.export_btn = ctk.CTkButton(preview_header, text="💾", width=36, command=self.export_image)
        self.export_btn.grid(row=0, column=2, sticky="e", padx=(0, 8))
        self.export_btn.configure(state="disabled")

        self.open_folder_btn = ctk.CTkButton(preview_header, text="📁", width=36, command=self.open_export_folder)
        self.open_folder_btn.grid(row=0, column=3, sticky="e", padx=(0, 8))
        self.open_folder_btn.configure(state="disabled")

        refresh_btn = ctk.CTkButton(preview_header, text="↻", width=36, command=self.update_processed_preview)
        refresh_btn.grid(row=0, column=4, sticky="e")

        preview_frame = ctk.CTkFrame(right, corner_radius=16)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)

        self.output_img_label = ctk.CTkLabel(preview_frame, text="Select an image to preview the resized output", corner_radius=12, fg_color="#1a1a1a")
        self.output_img_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        export_frame = ctk.CTkFrame(right, corner_radius=16)
        export_frame.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 14))
        export_frame.grid_columnconfigure(0, weight=1)

        self.export_path_label = ctk.CTkLabel(export_frame, text="Last exported: -", text_color="#a0a0a0", anchor="w")
        self.export_path_label.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

    def choose_source_image(self):
        path = ""
        start_dir = os.path.expanduser("~/Pictures")

        if self.source_image_path and os.path.isfile(self.source_image_path):
            start_dir = os.path.dirname(self.source_image_path)

        kdialog = shutil.which("kdialog")
        if kdialog:
            p = subprocess.run(
                [
                    kdialog,
                    "--getopenfilename",
                    start_dir,
                    "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if p.returncode == 0:
                path = p.stdout.strip()

        if not path:
            path = filedialog.askopenfilename(
                title="Choose source image",
                initialdir=start_dir,
                filetypes=[
                    ("Images", "*.png *.jpg *.jpeg *.webp *.bmp"),
                    ("All files", "*.*"),
                ],
            )

        if not path:
            return

        self.set_source_image(path)

    def set_source_image(self, path):
        if not path or not os.path.isfile(path):
            return

        with Image.open(path) as im:
            src = ImageOps.exif_transpose(im).convert("RGB")

        self.source_image_path = path
        self.source_image_pil = src

        self.update_source_image_preview()
        self.update_processed_preview()
    
    def get_best_resolution_for_source(self, source_width, source_height, model_id):
        source_ratio = source_width / source_height
        best_resolution = MODEL_RESOLUTIONS[model_id][0]
        best_difference = None

        for resolution in MODEL_RESOLUTIONS[model_id]:
            width, height = self.parse_resolution(resolution)
            resolution_ratio = width / height
            difference = abs(source_ratio - resolution_ratio)

            if best_difference is None or difference < best_difference:
                best_resolution = resolution
                best_difference = difference

        return best_resolution

    def update_source_image_preview(self):
        src_w, src_h = self.source_image_pil.size
        source_ratio = self.get_aspect_ratio(src_w, src_h)
        source_name = os.path.basename(self.source_image_path)

        thumb = self.source_image_pil.copy()
        thumb.thumbnail((400, 260))
        thumb_w, thumb_h = thumb.size

        self.source_image_ctk = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=(thumb_w, thumb_h))
        self.source_img_label.configure(image=self.source_image_ctk, text="")
        self.source_path_label.configure(text=f"{source_name} ({src_w}x{src_h}, {source_ratio})")

    def get_model_names_for_kind(self):
        names = [PLACEHOLDER]

        for model_id, name in MODEL_NAMES.items():
            if MODEL_KINDS[model_id] == self.selected_model_kind:
                names.append(name)

        return names

    def select_initial_model(self, model_id):
        if not model_id or model_id not in MODEL_NAMES:
            return

        self.selected_model_kind = MODEL_KINDS[model_id]
        tab_name = "Images" if self.selected_model_kind == "image" else "Videos"
        model_name = MODEL_NAMES[model_id]

        self.model_kind_tabs.set(tab_name)
        self.model_menu.configure(values=self.get_model_names_for_kind())
        self.model_menu.set(model_name)
        self.on_model_changed(model_name)

    def on_model_kind_changed(self, model_kind):
        self.selected_model_kind = model_kind.lower()[:-1]
        self.selected_model_id = ""
        self.selected_resolution = ""
        self.model_menu.configure(values=self.get_model_names_for_kind())
        self.model_menu.set(PLACEHOLDER)
        self.update_resolution_menu()
        self.update_output_info()
        self.update_processed_preview()

    def on_model_changed(self, model_name):
        if model_name == PLACEHOLDER:
            self.selected_model_id = ""
            self.selected_resolution = ""
            self.update_resolution_menu()
            self.update_output_info()
            self.update_processed_preview()
            return

        self.selected_model_id = MODEL_IDS_BY_NAME[model_name]

        if self.source_image_pil is not None:
            src_w, src_h = self.source_image_pil.size
            self.selected_resolution = self.get_best_resolution_for_source(src_w, src_h, self.selected_model_id)
        else:
            self.selected_resolution = MODEL_RESOLUTIONS[self.selected_model_id][0]

        self.update_resolution_menu()
        self.update_output_info()
        self.update_processed_preview()

    def on_resolution_changed(self, resolution):
        if resolution == PLACEHOLDER:
            self.selected_resolution = ""
        else:
            self.selected_resolution = resolution

        self.update_output_info()
        self.update_processed_preview()

    def on_crop_mode_changed(self, crop_mode):
        self.selected_crop_mode = crop_mode
        self.update_processed_preview()

    def update_resolution_menu(self):
        if not self.selected_model_id:
            self.resolution_menu.configure(values=[PLACEHOLDER])
            self.resolution_menu.set(PLACEHOLDER)
            return

        resolutions = MODEL_RESOLUTIONS[self.selected_model_id]
        self.resolution_menu.configure(values=resolutions)
        self.resolution_menu.set(self.selected_resolution)

    def update_output_info(self):
        if not self.selected_resolution:
            self.output_info_label.configure(text="Output: -")
            return

        width, height = self.parse_resolution(self.selected_resolution)
        output_ratio = self.get_aspect_ratio(width, height)
        self.output_info_label.configure(text=f"Output: {width}x{height} ({output_ratio})")

    def update_processed_preview(self):
        if self.source_image_pil is None or not self.selected_resolution:
            self.output_image_pil = None
            self.output_image_ctk = None
            self.output_img_label.configure(image=None, text="Select an image to preview the resized output")
            self.export_btn.configure(state="disabled")
            return

        width, height = self.parse_resolution(self.selected_resolution)
        self.output_image_pil = self.crop_and_resize_image(width, height)

        thumb = self.output_image_pil.copy()
        thumb.thumbnail((580, 560))
        thumb_w, thumb_h = thumb.size

        self.output_image_ctk = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=(thumb_w, thumb_h))
        self.output_img_label.configure(image=self.output_image_ctk, text="")
        self.export_btn.configure(state="normal")
        self.update_output_info()

    def crop_and_resize_image(self, width, height):
        src_w, src_h = self.source_image_pil.size

        if self.selected_crop_mode == "Smart crop":
            focus_x, focus_y = self.get_smart_crop_focus()
        elif self.selected_crop_mode == "Face crop":
            focus_x, focus_y = self.get_face_crop_focus()
        else:
            focus_x = src_w / 2
            focus_y = src_h / 2

        return self.crop_image_to_focus(self.source_image_pil, width, height, focus_x, focus_y)

    def crop_image_to_focus(self, img, target_width, target_height, focus_x, focus_y):
        src_w, src_h = img.size
        source_ratio = src_w / src_h
        target_ratio = target_width / target_height

        if abs(source_ratio - target_ratio) < 0.0001:
            return img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        if source_ratio > target_ratio:
            crop_h = src_h
            crop_w = int(round(crop_h * target_ratio))
            crop_w = min(crop_w, src_w)

            left = int(round(focus_x - crop_w / 2))
            left = max(0, min(left, src_w - crop_w))

            box = (left, 0, left + crop_w, src_h)
        else:
            crop_w = src_w
            crop_h = int(round(crop_w / target_ratio))
            crop_h = min(crop_h, src_h)

            top = int(round(focus_y - crop_h / 2))
            top = max(0, min(top, src_h - crop_h))

            box = (0, top, src_w, top + crop_h)

        return img.crop(box).resize((target_width, target_height), Image.Resampling.LANCZOS)

    def get_smart_crop_focus(self):
        src_w, src_h = self.source_image_pil.size

        if not hasattr(cv2, "saliency"):
            return src_w / 2, src_h / 2

        arr = np.array(self.source_image_pil)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, saliency_map = saliency.computeSaliency(bgr)

        if not ok or saliency_map is None:
            return src_w / 2, src_h / 2

        weights = saliency_map.astype("float32")
        weights = weights - weights.min()

        if weights.max() > 0:
            weights = weights / weights.max()

        total = weights.sum()

        if total <= 0:
            return src_w / 2, src_h / 2

        y_idx, x_idx = np.indices(weights.shape)
        focus_x = float((x_idx * weights).sum() / total)
        focus_y = float((y_idx * weights).sum() / total)

        map_h, map_w = weights.shape
        focus_x = focus_x * (src_w / map_w)
        focus_y = focus_y * (src_h / map_h)

        return focus_x, focus_y

    def get_cascade_path(self, filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(base_dir, "cascades", filename)

        if os.path.isfile(local_path):
            return local_path

        opencv_path = os.path.join(cv2.data.haarcascades, filename)

        if os.path.isfile(opencv_path):
            return opencv_path

        return local_path

    def get_largest_cascade_focus(self, gray, filename, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        cascade_path = self.get_cascade_path(filename)
        cascade = cv2.CascadeClassifier(cascade_path)

        if cascade.empty():
            return None

        boxes = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )

        if len(boxes) == 0:
            return None

        x, y, w, h = max(boxes, key=lambda box: box[2] * box[3])
        return x + w / 2, y + h / 2

    def get_profile_cascade_focus(self, gray):
        src_w, src_h = self.source_image_pil.size

        focus = self.get_largest_cascade_focus(
            gray,
            "haarcascade_profileface.xml",
            scale_factor=1.1,
            min_neighbors=5,
            min_size=(30, 30),
        )

        if focus is not None:
            return focus

        flipped = cv2.flip(gray, 1)
        cascade_path = self.get_cascade_path("haarcascade_profileface.xml")
        cascade = cv2.CascadeClassifier(cascade_path)

        if cascade.empty():
            return None

        boxes = cascade.detectMultiScale(
            flipped,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(boxes) == 0:
            return None

        x, y, w, h = max(boxes, key=lambda box: box[2] * box[3])
        focus_x = src_w - (x + w / 2)
        focus_y = y + h / 2

        return focus_x, focus_y

    def get_eye_cascade_focus(self, gray):
        cascade_path = self.get_cascade_path("haarcascade_eye.xml")
        cascade = cv2.CascadeClassifier(cascade_path)

        if cascade.empty():
            return None

        eyes = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(16, 16),
        )

        if len(eyes) == 0:
            return None

        eyes = sorted(eyes, key=lambda eye: eye[2] * eye[3], reverse=True)[:2]

        focus_x = 0
        focus_y = 0

        for x, y, w, h in eyes:
            focus_x += x + w / 2
            focus_y += y + h / 2

        focus_x = focus_x / len(eyes)
        focus_y = focus_y / len(eyes)

        return focus_x, focus_y

    def get_face_crop_focus(self):
        src_w, src_h = self.source_image_pil.size
        gray = cv2.cvtColor(np.array(self.source_image_pil), cv2.COLOR_RGB2GRAY)

        focus = self.get_largest_cascade_focus(
            gray,
            "haarcascade_frontalface_default.xml",
            scale_factor=1.1,
            min_neighbors=5,
            min_size=(30, 30),
        )

        if focus is not None:
            return focus

        focus = self.get_profile_cascade_focus(gray)

        if focus is not None:
            return focus

        focus = self.get_largest_cascade_focus(
            gray,
            "lbpcascade_animeface.xml",
            scale_factor=1.1,
            min_neighbors=5,
            min_size=(24, 24),
        )

        if focus is not None:
            return focus

        focus = self.get_eye_cascade_focus(gray)

        if focus is not None:
            return focus

        return self.get_smart_crop_focus()

    def choose_export_image_path(self, source_dir, source_name, default_name):
        path = ""
        source_ext = os.path.splitext(source_name)[1] or ".png"
        start_path = os.path.join(source_dir, default_name)

        kdialog = shutil.which("kdialog")
        if kdialog:
            p = subprocess.run(
                [
                    kdialog,
                    "--getsavefilename",
                    start_path,
                    "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if p.returncode == 0:
                path = p.stdout.strip()

        if not path:
            path = filedialog.asksaveasfilename(
                title="Export image",
                initialdir=source_dir,
                initialfile=default_name,
                defaultextension=source_ext,
                filetypes=[
                    ("PNG", "*.png"),
                    ("JPEG", "*.jpg *.jpeg"),
                    ("WebP", "*.webp"),
                    ("BMP", "*.bmp"),
                    ("All files", "*.*"),
                ],
            )

        if path and not os.path.splitext(path)[1]:
            path += source_ext

        return path

    def export_image(self):
        if self.output_image_pil is None or not self.source_image_path:
            return

        if self.output_folder_path:
            source_dir = self.output_folder_path
        else:
            source_dir = os.path.dirname(self.source_image_path)

        os.makedirs(source_dir, exist_ok=True)

        source_name = os.path.basename(self.source_image_path)
        ts = int(time.time())
        default_name = f"{ts}_{source_name}"

        path = self.choose_export_image_path(source_dir, source_name, default_name)

        if not path:
            return

        self.output_image_pil.save(path)
        self.last_export_path = path
        self.export_path_label.configure(text=f"Last exported: {path}")
        self.open_folder_btn.configure(state="normal")

    def open_export_folder(self):
        if self.output_folder_path:
            folder = self.output_folder_path
        elif self.last_export_path:
            folder = os.path.dirname(os.path.abspath(self.last_export_path))
        else:
            return

        os.makedirs(folder, exist_ok=True)
        subprocess.Popen(["xdg-open", folder])

    def parse_resolution(self, resolution):
        width, height = resolution.split("x")
        return int(width), int(height)

    def get_aspect_ratio(self, width, height):
        divisor = math.gcd(width, height)
        return f"{width // divisor}:{height // divisor}"

    def run(self):
        self.app.mainloop()

if __name__ == "__main__":
    gui = ImageEditorGUI(
        get_flag_value("--model-id"),
        get_flag_value("--source-image"),
        get_flag_value("--output-folder"),
    )
    gui.run()
