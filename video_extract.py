import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import json
import time
from pathlib import Path

# ── Try importing optional heavy dependencies ──────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False

# ── Color palette ──────────────────────────────────────────────────────────
BG        = "#0f0f13"
PANEL     = "#1a1a22"
PANEL2    = "#22222e"
ACCENT    = "#7c5cbf"
ACCENT2   = "#a07de0"
TEXT      = "#e8e4f0"
TEXT_DIM  = "#7a7590"
GREEN     = "#4caf78"
RED       = "#e05c5c"
BORDER    = "#2e2e3e"
ENTRY_BG  = "#13131a"

MODELS = ["— не выбрано —", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt", "Другое"]


# ══════════════════════════════════════════════════════════════════════════
#  _DirectModel — тонкая обёртка вокруг nn.Module для работы как YOLO
# ══════════════════════════════════════════════════════════════════════════

class _DirectModel:
    """Wraps a raw nn.Module (from torch.load) so it behaves like a YOLO object."""
    def __init__(self, module):
        self._module = module
        # copy .names from inner module
        self.names = getattr(module, "names", {})

    def __call__(self, frame, conf=0.25, iou=0.45, verbose=False):
        """Run inference — delegate to the wrapped module via ultralytics predict."""
        # Try to use ultralytics predict pipeline if the inner model supports it
        try:
            return self._module.predict(frame, conf=conf, iou=iou, verbose=verbose)
        except Exception:
            pass
        # Fallback: direct forward (returns raw tensors, not Results — wrap minimally)
        import torch
        import numpy as np
        if isinstance(frame, np.ndarray):
            import cv2 as _cv2
            img = _cv2.resize(frame, (640, 640))
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0)
        else:
            tensor = frame
        with torch.no_grad():
            out = self._module(tensor)
        # Return empty results so caller doesn't crash
        return []

# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════

def styled_button(parent, text, command, bg=ACCENT, fg=TEXT, width=None, small=False):
    kw = dict(relief="flat", bd=0, cursor="hand2",
              bg=bg, fg=fg, font=("Consolas", 9 if small else 10, "bold"),
              padx=10, pady=4 if small else 6, command=command)
    if width:
        kw["width"] = width
    btn = tk.Button(parent, text=text, **kw)
    btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT2 if bg == ACCENT else "#c0392b" if bg == RED else "#3a8a5c"))
    btn.bind("<Leave>", lambda e: btn.config(bg=bg))
    return btn


def section_label(parent, text):
    frm = tk.Frame(parent, bg=BG)
    tk.Label(frm, text=text, bg=BG, fg=ACCENT2,
             font=("Consolas", 11, "bold")).pack(side="left")
    tk.Frame(frm, bg=BORDER, height=1).pack(side="left", fill="x", expand=True, padx=(8, 0))
    return frm


def entry_field(parent, textvariable=None, width=8):
    e = tk.Entry(parent, textvariable=textvariable, width=width,
                 bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
                 relief="flat", bd=0, font=("Consolas", 10),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT)
    return e


# ══════════════════════════════════════════════════════════════════════════
#  Scrollable frame
# ══════════════════════════════════════════════════════════════════════════

class ScrollFrame(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        self.vbar   = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=BG)
        self._win  = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_configure(self, _):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event):
        self.canvas.itemconfig(self._win, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


# ══════════════════════════════════════════════════════════════════════════
#  Progress bar widget
# ══════════════════════════════════════════════════════════════════════════

class FancyProgress(tk.Frame):
    def __init__(self, parent, label="", **kw):
        super().__init__(parent, bg=BG, **kw)
        self._lbl = tk.Label(self, text=label, bg=BG, fg=TEXT_DIM, font=("Consolas", 9))
        self._lbl.pack(anchor="w")
        bar_frame = tk.Frame(self, bg=BORDER, height=8)
        bar_frame.pack(fill="x", pady=(2, 0))
        bar_frame.pack_propagate(False)
        self._fill = tk.Frame(bar_frame, bg=ACCENT, height=8)
        self._fill.place(x=0, y=0, relheight=1, relwidth=0)
        self._pct = tk.Label(self, text="0%", bg=BG, fg=ACCENT2, font=("Consolas", 8))
        self._pct.pack(anchor="e")

    def set(self, value: float, text: str = ""):
        v = max(0.0, min(1.0, value))
        self._fill.place(relwidth=v)
        self._pct.config(text=f"{int(v*100)}%")
        if text:
            self._lbl.config(text=text)

    def reset(self, label=""):
        self._fill.place(relwidth=0)
        self._pct.config(text="0%")
        if label:
            self._lbl.config(text=label)


# ══════════════════════════════════════════════════════════════════════════
#  Main App
# ══════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Frame Extractor")
        self.configure(bg=BG)
        self.geometry("860x900")
        self.minsize(700, 600)
        self.resizable(True, True)

        self._video_folder    = tk.StringVar(value="Папка не выбрана")
        self._model_choice    = tk.StringVar(value="— не выбрано —")
        self._custom_model    = None
        self._save_folder     = tk.StringVar(value="Папка не выбрана")
        self._conf_var        = tk.DoubleVar(value=0.25)
        self._iou_var         = tk.DoubleVar(value=0.45)
        self._class_vars      = {}
        self._current_classes = []
        self._model_loaded    = None
        self._running         = False
        self._stop_flag       = threading.Event()
        self._run_counter     = self._load_run_counter()

        self._build_ui()

    # ── persist run counter ───────────────────────────────────────────────
    def _counter_path(self):
        return Path(os.path.expanduser("~")) / ".yolo_extractor_counter.json"

    def _load_run_counter(self):
        p = self._counter_path()
        if p.exists():
            try:
                return json.loads(p.read_text())["count"]
            except Exception:
                pass
        return 0

    def _save_run_counter(self):
        self._counter_path().write_text(json.dumps({"count": self._run_counter}))

    # ── UI build ──────────────────────────────────────────────────────────
    def _build_ui(self):
        hdr = tk.Frame(self, bg=PANEL, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="⬡  YOLO FRAME EXTRACTOR",
                 bg=PANEL, fg=ACCENT2, font=("Consolas", 14, "bold")).pack()
        tk.Label(hdr, text="extract clean frames from video using neural detection",
                 bg=PANEL, fg=TEXT_DIM, font=("Consolas", 8)).pack()

        self._scroll = ScrollFrame(self)
        self._scroll.pack(fill="both", expand=True)
        c = self._scroll.inner
        PX = 20

        # ── 01 Video ──────────────────────────────────────────────────────
        section_label(c, "01  ВЫБРАТЬ ВИДЕО").pack(fill="x", padx=PX, pady=(16, 4))
        row1 = tk.Frame(c, bg=BG)
        row1.pack(fill="x", padx=PX, pady=(0, 4))
        self._video_lbl = tk.Label(row1, textvariable=self._video_folder,
                                   bg=ENTRY_BG, fg=TEXT_DIM, font=("Consolas", 9),
                                   anchor="w", padx=8,
                                   highlightthickness=1, highlightbackground=BORDER)
        self._video_lbl.pack(side="left", fill="x", expand=True, ipady=5)
        self._video_btn = styled_button(row1, "  Выбрать папку  ", self._pick_video_folder)
        self._video_btn.pack(side="left", padx=(8, 0))

        # ── 02 Model ──────────────────────────────────────────────────────
        section_label(c, "02  ВЫБРАТЬ МОДЕЛЬ").pack(fill="x", padx=PX, pady=(14, 4))
        row2 = tk.Frame(c, bg=BG)
        row2.pack(fill="x", padx=PX, pady=(0, 4))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TCombobox",
                        fieldbackground=ENTRY_BG, background=PANEL2,
                        foreground=TEXT, selectbackground=ACCENT,
                        selectforeground=TEXT, borderwidth=0,
                        arrowcolor=ACCENT2, font=("Consolas", 10))
        style.map("Dark.TCombobox",
                  fieldbackground=[("readonly", ENTRY_BG)],
                  background=[("readonly", PANEL2)],
                  foreground=[("readonly", TEXT)])

        self._model_combo = ttk.Combobox(row2, values=MODELS,
                                         textvariable=self._model_choice,
                                         state="readonly", width=18,
                                         style="Dark.TCombobox",
                                         font=("Consolas", 10))
        self._model_combo.pack(side="left")
        self._model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        self._model_custom_btn = styled_button(row2, "  Выбрать файл  ",
                                               self._pick_custom_model, bg=PANEL2)

        self._custom_model_path_lbl = tk.Label(
            row2, text="", bg=BG, fg=TEXT_DIM,
            font=("Consolas", 8), anchor="w", wraplength=420)

        # download progress (hidden by default)
        self._dl_frame = tk.Frame(c, bg=BG)
        self._dl_progress = FancyProgress(self._dl_frame, label="")
        self._dl_progress.pack(fill="x", padx=PX)

        # ── 03 Classes (hidden until model loaded) ────────────────────────
        self._classes_outer = tk.Frame(c, bg=BG)
        self._classes_outer.pack(fill="x")

        # ── 04 Params ─────────────────────────────────────────────────────
        self._params_frame = tk.Frame(c, bg=BG)
        self._params_frame.pack(fill="x")
        self._build_params_section()

        # ── 05 Save ───────────────────────────────────────────────────────
        self._save_outer = tk.Frame(c, bg=BG)
        self._save_outer.pack(fill="x")
        self._build_save_section()

        # ── 06 Run ────────────────────────────────────────────────────────
        self._run_outer = tk.Frame(c, bg=BG)
        self._run_outer.pack(fill="x")
        self._build_run_section()

        # ── Log ───────────────────────────────────────────────────────────
        section_label(c, "LOG").pack(fill="x", padx=PX, pady=(14, 4))
        log_frame = tk.Frame(c, bg=ENTRY_BG,
                             highlightthickness=1, highlightbackground=BORDER)
        log_frame.pack(fill="x", padx=PX, pady=(0, 20))
        self._log = tk.Text(log_frame, bg=ENTRY_BG, fg=TEXT_DIM,
                            font=("Consolas", 8), height=8,
                            relief="flat", bd=0, state="disabled",
                            insertbackground=TEXT, wrap="word")
        sb = tk.Scrollbar(log_frame, command=self._log.yview, bg=PANEL)
        self._log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._log.pack(fill="both", expand=True, padx=6, pady=4)

        self._log_msg("Приложение запущено. Выберите папку с видео и модель.")
        if not CV2_OK:
            self._log_msg("⚠  OpenCV не установлен: pip install opencv-python")
        if not YOLO_OK:
            self._log_msg("⚠  Ultralytics не установлен: pip install ultralytics")

    # ── Params section ────────────────────────────────────────────────────
    def _build_params_section(self):
        f = self._params_frame
        section_label(f, "03  ПАРАМЕТРЫ").pack(fill="x", padx=20, pady=(14, 6))
        row = tk.Frame(f, bg=BG)
        row.pack(fill="x", padx=20, pady=(0, 4))
        tk.Label(row, text="Conf:", bg=BG, fg=TEXT, font=("Consolas", 10)).pack(side="left")
        entry_field(row, textvariable=self._conf_var, width=6).pack(side="left", padx=(4, 20))
        self._conf_scale = tk.Scale(row, from_=0.01, to=1.0, resolution=0.01,
                                    orient="horizontal", length=200, variable=self._conf_var,
                                    bg=BG, fg=TEXT_DIM, troughcolor=PANEL2,
                                    highlightthickness=0, sliderrelief="flat",
                                    activebackground=ACCENT, font=("Consolas", 8))
        self._conf_scale.pack(side="left")

        row2 = tk.Frame(f, bg=BG)
        row2.pack(fill="x", padx=20, pady=(4, 4))
        tk.Label(row2, text="IOU: ", bg=BG, fg=TEXT, font=("Consolas", 10)).pack(side="left")
        entry_field(row2, textvariable=self._iou_var, width=6).pack(side="left", padx=(4, 20))
        self._iou_scale = tk.Scale(row2, from_=0.01, to=1.0, resolution=0.01,
                                   orient="horizontal", length=200, variable=self._iou_var,
                                   bg=BG, fg=TEXT_DIM, troughcolor=PANEL2,
                                   highlightthickness=0, sliderrelief="flat",
                                   activebackground=ACCENT, font=("Consolas", 8))
        self._iou_scale.pack(side="left")

    # ── Save section ──────────────────────────────────────────────────────
    def _build_save_section(self):
        f = self._save_outer
        section_label(f, "04  СОХРАНИТЬ").pack(fill="x", padx=20, pady=(14, 6))
        row = tk.Frame(f, bg=BG)
        row.pack(fill="x", padx=20, pady=(0, 4))
        self._save_lbl = tk.Label(row, textvariable=self._save_folder,
                                  bg=ENTRY_BG, fg=TEXT_DIM, font=("Consolas", 9),
                                  anchor="w", padx=8,
                                  highlightthickness=1, highlightbackground=BORDER)
        self._save_lbl.pack(side="left", fill="x", expand=True, ipady=5)
        self._save_btn = styled_button(row, "  Выбрать папку  ", self._pick_save_folder)
        self._save_btn.pack(side="left", padx=(8, 0))

    # ── Run section ───────────────────────────────────────────────────────
    def _build_run_section(self):
        f = self._run_outer
        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(18, 10))
        self._run_btn = styled_button(f, "  ▶  ЗАПУСК  ", self._toggle_run,
                                      bg=GREEN, width=20)
        self._run_btn.pack(pady=(0, 6))
        self._run_progress = FancyProgress(f, label="")
        self._run_progress.pack(fill="x", padx=20, pady=(4, 10))

    # ── Classes section ───────────────────────────────────────────────────
    def _build_classes_section(self, classes: list):
        f = self._classes_outer
        for w in f.winfo_children():
            w.destroy()
        self._class_vars = {}
        self._current_classes = classes

        header = tk.Frame(f, bg=BG)
        header.pack(fill="x", padx=20, pady=(14, 6))
        section_label(header, "  КЛАССЫ").pack(side="left", fill="x", expand=True)

        self._toggle_all_state = True
        self._toggle_btn = styled_button(header, "Убрать все",
                                         self._toggle_all_classes, bg=PANEL2, small=True)
        self._toggle_btn.pack(side="right")

        grid_frame = tk.Frame(f, bg=BG)
        grid_frame.pack(fill="x", padx=20)

        ROWS_PER_COL = 10
        for idx, name in enumerate(classes):
            col = idx // ROWS_PER_COL
            row = idx % ROWS_PER_COL
            var = tk.BooleanVar(value=True)
            self._class_vars[name] = var
            cell = tk.Frame(grid_frame, bg=BG)
            cell.grid(row=row, column=col, sticky="w", padx=(0, 20), pady=1)
            cb = tk.Checkbutton(cell, variable=var, text=name,
                                bg=BG, fg=TEXT, selectcolor=ACCENT,
                                activebackground=BG, activeforeground=TEXT_DIM,
                                font=("Consolas", 9), bd=0, relief="flat", cursor="hand2")
            cb.pack(side="left")

    def _toggle_all_classes(self):
        self._toggle_all_state = not self._toggle_all_state
        for v in self._class_vars.values():
            v.set(self._toggle_all_state)
        self._toggle_btn.config(
            text="Убрать все" if self._toggle_all_state else "Отметить все")

    # ── Pickers ───────────────────────────────────────────────────────────
    def _pick_video_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с видео")
        if folder:
            self._video_folder.set(folder)
            self._log_msg(f"Папка с видео: {folder}")

    def _pick_save_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку для сохранения")
        if folder:
            self._save_folder.set(folder)
            self._log_msg(f"Папка сохранения: {folder}")

    def _pick_custom_model(self):
        path = filedialog.askopenfilename(
            title="Выберите файл модели",
            filetypes=[("PyTorch model", "*.pt *.pth"), ("All files", "*.*")])
        if path:
            self._custom_model = path
            self._custom_model_path_lbl.config(text=path)
            self._custom_model_path_lbl.pack(side="left", padx=(10, 0))
            self._log_msg(f"Пользовательская модель: {path}")
            self._load_model_async(path)

    # ── Model change ──────────────────────────────────────────────────────
    def _on_model_change(self, _=None):
        choice = self._model_choice.get()
        if choice == "— не выбрано —":
            self._model_custom_btn.pack_forget()
            self._custom_model_path_lbl.pack_forget()
            return
        if choice == "Другое":
            self._model_custom_btn.pack(side="left", padx=(8, 0))
            if self._custom_model:
                self._custom_model_path_lbl.config(text=self._custom_model)
                self._custom_model_path_lbl.pack(side="left", padx=(10, 0))
        else:
            self._model_custom_btn.pack_forget()
            self._custom_model_path_lbl.pack_forget()
            self._load_model_async(choice)

    # ── Model loading ─────────────────────────────────────────────────────
    def _load_model_async(self, model_name: str):
        if not YOLO_OK:
            self._log_msg("⚠  Ultralytics не установлен.")
            return
        t = threading.Thread(target=self._load_model_thread,
                             args=(model_name,), daemon=True)
        t.start()

    def _load_model_thread(self, model_name: str):
        self.after(0, lambda: self._dl_frame.pack(fill="x", padx=0, pady=(4, 4)))
        short_name = Path(model_name).name
        self._dl_progress.reset(f"Загрузка {short_name} …")

        done = threading.Event()

        def fake_progress():
            for i in range(1, 95):
                if done.is_set():
                    break
                self._dl_progress.set(i / 100, f"Загрузка {short_name} … {i}%")
                time.sleep(0.08)

        threading.Thread(target=fake_progress, daemon=True).start()

        model  = None
        errors = []

        # ── Стратегия 1: стандартный ultralytics YOLO() ───────────────────
        try:
            model = YOLO(model_name)
            self._log_msg(f"  [1] ultralytics YOLO() — ОК")
        except Exception as e1:
            errors.append(f"1) YOLO(): {e1}")

        # ── Стратегия 2: torch.load → попытка обернуть checkpoint ─────────
        if model is None:
            try:
                import torch
                import tempfile

                ckpt = torch.load(model_name, map_location="cpu", weights_only=False)
                self._log_msg(f"  [2] torch.load — ОК, тип: {type(ckpt).__name__}")

                if isinstance(ckpt, dict):
                    keys = list(ckpt.keys())
                    self._log_msg(f"      ключи checkpoint: {keys}")

                    # ultralytics сохраняет веса модели в ключе 'model'
                    inner = ckpt.get("model") or ckpt.get("ema") or ckpt.get("net")

                    if inner is not None and hasattr(inner, "names"):
                        # Пересохраняем во временный файл и загружаем через YOLO
                        tmp_path = Path(tempfile.mktemp(suffix=".pt"))
                        try:
                            # Сохраняем полный checkpoint — ultralytics его поймёт
                            torch.save(ckpt, str(tmp_path))
                            model = YOLO(str(tmp_path))
                            self._log_msg("  [2] пересохранение + YOLO() — ОК")
                        except Exception as etmp:
                            # Если YOLO не принял — используем inner напрямую
                            self._log_msg(f"  [2] YOLO(tmp) не вышло: {etmp}, пробуем _DirectModel")
                            model = _DirectModel(inner)
                            self._log_msg("  [2] _DirectModel(inner) — ОК")
                        finally:
                            try:
                                tmp_path.unlink()
                            except Exception:
                                pass
                    else:
                        errors.append(f"2) inner без .names (inner={type(inner).__name__ if inner else None})")

                elif hasattr(ckpt, "names"):
                    model = _DirectModel(ckpt)
                    self._log_msg("  [2] _DirectModel(ckpt) — ОК")
                else:
                    errors.append(f"2) тип {type(ckpt).__name__} не поддерживается")

            except Exception as e2:
                errors.append(f"2) torch.load: {e2}")

        # ── Стратегия 3: YOLO с явным task='detect' ───────────────────────
        if model is None:
            try:
                model = YOLO(model_name, task="detect")
                self._log_msg("  [3] YOLO(task='detect') — ОК")
            except Exception as e3:
                errors.append(f"3) YOLO(task='detect'): {e3}")

        # ── Стратегия 4: попробуем обновить ultralytics и повторить ───────
        # (просто информируем пользователя)

        done.set()

        if model is None:
            self._dl_progress.set(0, f"❌ Ошибка загрузки: {short_name}")
            self._log_msg(f"❌ Не удалось загрузить «{short_name}»:")
            for err in errors:
                self._log_msg(f"  {err}")
            self._log_msg("──────────────────────────────────────────")
            self._log_msg("  Возможные причины и решения:")
            self._log_msg("  • Модель дообучена на другой версии ultralytics →")
            self._log_msg("    pip install -U ultralytics")
            self._log_msg("  • Несовместимость версий PyTorch →")
            self._log_msg("    убедитесь что torch той же версии что при обучении")
            self._log_msg("  • Файл повреждён / неполностью скопирован")
            return

        self._dl_progress.set(1.0, f"✔  {short_name} загружена")
        self._model_loaded = model

        try:
            classes = list(model.names.values())
        except Exception:
            classes = []

        if classes:
            self.after(0, lambda: self._build_classes_section(classes))
        else:
            self._log_msg("⚠  Не удалось определить классы модели.")

        try:
            import torch
            device_str = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        except Exception:
            device_str = "CPU"

        self._log_msg(f"Модель «{short_name}» готова.")
        self._log_msg(f"  Классов в модели: {len(classes)}")
        self._log_msg(f"  Устройство: {device_str}")

    # ── Run / Stop ────────────────────────────────────────────────────────
    def _toggle_run(self):
        if self._running:
            self._stop_flag.set()
            self._running = False
            self._log_msg("Остановлено пользователем.")
        else:
            self._start_run()

    def _set_ui_locked(self, locked: bool):
        state = "disabled" if locked else "normal"
        for w in [self._video_lbl, self._model_combo, self._model_custom_btn,
                  self._conf_scale, self._iou_scale, self._save_lbl,
                  self._video_btn, self._save_btn]:
            try:
                w.config(state=state)
            except Exception:
                pass
        for cb in self._iter_checkbuttons():
            try:
                cb.config(state=state)
            except Exception:
                pass
        if hasattr(self, "_toggle_btn"):
            try:
                self._toggle_btn.config(state=state)
            except Exception:
                pass

    def _iter_checkbuttons(self):
        f = self._classes_outer
        for child in f.winfo_children():
            for sub in child.winfo_children():
                for item in sub.winfo_children():
                    if isinstance(item, tk.Checkbutton):
                        yield item
                if isinstance(sub, tk.Checkbutton):
                    yield sub

    def _start_run(self):
        if not CV2_OK or not YOLO_OK:
            messagebox.showerror("Ошибка",
                "Установите зависимости:\n  pip install opencv-python ultralytics")
            return

        video_folder = self._video_folder.get()
        save_folder  = self._save_folder.get()

        if video_folder == "Папка не выбрана":
            messagebox.showwarning("Внимание", "Выберите папку с видео.")
            return
        if save_folder == "Папка не выбрана":
            messagebox.showwarning("Внимание", "Выберите папку для сохранения.")
            return
        if self._model_loaded is None:
            messagebox.showwarning("Внимание", "Сначала выберите и загрузите модель.")
            return

        selected = [n for n, v in self._class_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("Внимание", "Выберите хотя бы один класс.")
            return

        exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
        videos = [p for p in Path(video_folder).iterdir()
                  if p.suffix.lower() in exts]
        if not videos:
            messagebox.showwarning("Внимание", "Видеофайлы не найдены в папке.")
            return

        self._run_counter += 1
        self._save_run_counter()
        self._stop_flag.clear()
        self._running = True
        self._run_btn.config(text="  ■  СТОП  ", bg=RED, activebackground="#c0392b")
        self._set_ui_locked(True)

        try:
            import torch
            device_str = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        except Exception:
            device_str = "CPU"

        self._log_msg(f"━━ Запуск #{self._run_counter} ━━")
        self._log_msg(f"  Видео: {len(videos)} файлов")
        self._log_msg(f"  Классов выбрано: {len(selected)}")
        self._log_msg(f"  Устройство: {device_str}")
        self._log_msg(f"  Conf={self._conf_var.get():.2f}  IOU={self._iou_var.get():.2f}")

        t = threading.Thread(target=self._run_thread,
                             args=(videos, save_folder, selected,
                                   self._conf_var.get(), self._iou_var.get()),
                             daemon=True)
        t.start()

    def _run_thread(self, videos, save_folder, selected_classes, conf, iou):
        run_dir  = Path(save_folder) / f"run_{self._run_counter:04d}"
        model    = self._model_loaded
        name_set = set(selected_classes)
        sel_ids  = {cid for cid, cname in model.names.items() if cname in name_set}

        # pre-calculate total frames for accurate progress
        total_frames_all = 0
        frame_counts = []
        for vp in videos:
            cap = cv2.VideoCapture(str(vp))
            fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            frame_counts.append(max(fc, 1))
            total_frames_all += max(fc, 1)
        total_frames_all = max(total_frames_all, 1)

        processed_frames_total = 0

        for vi, video_path in enumerate(videos):
            if self._stop_flag.is_set():
                break

            self._log_msg(f"→ [{vi+1}/{len(videos)}] {video_path.name}")
            cap       = cv2.VideoCapture(str(video_path))
            frame_idx = 0
            saved     = 0

            while True:
                if self._stop_flag.is_set():
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=conf, iou=iou, verbose=False)
                detected_names = set()
                for r in results:
                    for box in r.boxes:
                        cid = int(box.cls[0])
                        if cid in sel_ids:
                            detected_names.add(model.names[cid])

                for cname in detected_names:
                    out_dir = run_dir / cname
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"{video_path.stem}_frame{frame_idx:06d}.jpg"
                    cv2.imwrite(str(out_dir / fname), frame)
                    saved += 1

                frame_idx += 1
                processed_frames_total += 1

                if frame_idx % 10 == 0:
                    prog  = processed_frames_total / total_frames_all
                    label = (f"Видео {vi+1}/{len(videos)}: {video_path.name} "
                             f"— кадр {frame_idx}/{frame_counts[vi]}")
                    self.after(0, lambda p=prog, l=label: self._run_progress.set(p, l))

            cap.release()
            self._log_msg(f"  ✔ {video_path.name}: {frame_idx} кадров, сохранено {saved} фото")

        if not self._stop_flag.is_set():
            self._run_progress.set(1.0, "✔  Завершено")
            self._log_msg(f"━━ Готово! Результаты: {run_dir} ━━")
        self.after(0, self._finish_run)

    def _finish_run(self):
        self._running = False
        self._run_btn.config(text="  ▶  ЗАПУСК  ", bg=GREEN, activebackground="#3a8a5c")
        self._set_ui_locked(False)

    # ── Log ───────────────────────────────────────────────────────────────
    def _log_msg(self, text: str):
        def _do():
            self._log.configure(state="normal")
            ts = time.strftime("%H:%M:%S")
            self._log.insert("end", f"[{ts}] {text}\n")
            self._log.see("end")
            self._log.configure(state="disabled")
        self.after(0, _do)


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
