
import os
import io
import math
import tempfile
import threading
import time

import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")          # off-screen backend — avoids Tk conflicts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── colour palette ────────────────────────────────────────────────────────────
BG       = "#0d0d1a"
BG2      = "#161628"
CARD     = "#1e1e35"
ACCENT1  = "#7c3aed"   # violet
ACCENT2  = "#06b6d4"   # cyan
ACCENT3  = "#f59e0b"   # amber
GREEN    = "#10b981"
RED      = "#ef4444"
TEXT     = "#e2e8f0"
SUBTEXT  = "#94a3b8"
WHITE    = "#ffffff"

FONT_TITLE  = ("Segoe UI", 16, "bold")
FONT_HEAD   = ("Segoe UI", 11, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_MONO   = ("Courier New", 9)
FONT_SMALL  = ("Segoe UI", 8)

# ── helpers ───────────────────────────────────────────────────────────────────

def pil_from_bgr(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def resize_fit(img: Image.Image, w: int, h: int) -> Image.Image:
    img.thumbnail((w, h), Image.LANCZOS)
    return img

def file_size_kb(img_bgr: np.ndarray, quality: int = 85) -> float:
    """Encode to JPEG in-memory and return size in KB."""
    buf = io.BytesIO()
    pil_from_bgr(img_bgr).save(buf, format="JPEG", quality=quality)
    return buf.tell() / 1024

def psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """PSNR in dB.  Uses Frobenius norm per-channel."""
    original  = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse)

def energy_retention(S_full: np.ndarray, k: int) -> float:
    total = np.sum(S_full**2)
    if total == 0: return 100.0
    return np.sum(S_full[:k]**2) / total * 100.0

def compression_ratio(m: int, n: int, k: int) -> float:
    return (m * n) / max(1, k * (m + n + 1))


# ── SVD core ─────────────────────────────────────────────────────────────────

def svd_compress(img_bgr: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    """
    Compress image using truncated SVD.

    Returns
    -------
    compressed : uint8 BGR image
    info       : dict with mathematical metrics
    """
    img_f = img_bgr.astype(np.float64)
    channels = cv2.split(img_f)
    out_channels = []
    channel_info = []

    for c in channels:
        m, n = c.shape
        # Full SVD  →  U ∈ ℝ^{m×r}, S ∈ ℝ^r, Vt ∈ ℝ^{r×n}
        U, S, Vt = np.linalg.svd(c, full_matrices=False)

        r = len(S)
        k_eff = min(k, r)

        # Low-rank reconstruction: Aₖ = U[:, :k] · diag(S[:k]) · Vt[:k, :]
        Uk  = U[:, :k_eff]
        Sk  = S[:k_eff]
        Vtk = Vt[:k_eff, :]

        reconstructed = Uk @ np.diag(Sk) @ Vtk
        out_channels.append(reconstructed)

        channel_info.append({
            "singular_values" : S,
            "k_eff"           : k_eff,
            "rank"            : r,
            "energy"          : energy_retention(S, k_eff),
            "cond_number"     : float(S[0] / S[-1]) if S[-1] != 0 else float("inf"),
        })

    compressed = np.clip(cv2.merge(out_channels), 0, 255).astype(np.uint8)

    m, n = img_bgr.shape[:2]
    k_eff = channel_info[0]["k_eff"]

    info = {
        "k"              : k_eff,
        "channels"       : channel_info,
        "energy_mean"    : float(np.mean([c["energy"] for c in channel_info])),
        "compression_ratio": compression_ratio(m, n, k_eff),
        "psnr"           : psnr(img_bgr, compressed),
        "rank"           : channel_info[0]["rank"],
    }
    return compressed, info


# ── main application ──────────────────────────────────────────────────────────

class SVDApp:
    PREVIEW_W = 460
    PREVIEW_H = 340

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SVD Image Compression  ·  Linear Algebra Tool")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.minsize(1100, 700)

        self.image_bgr: np.ndarray | None = None
        self.compressed_bgr: np.ndarray | None = None
        self.image_path: str = ""
        self._compress_thread: threading.Thread | None = None
        self._pending_k: int | None = None

        # slider variables
        self.var_k        = tk.IntVar(value=50)
        self.var_quality  = tk.IntVar(value=85)   # JPEG output quality

        self._build_ui()

    # ─── UI CONSTRUCTION ──────────────────────────────────────────────────────

    def _build_ui(self):
        # ── title bar ──────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=ACCENT1, height=50)
        hdr.pack(fill="x")
        tk.Label(hdr, text="⚡  SVD Image Compression  ·  Linear Algebra Project",
                 font=FONT_TITLE, bg=ACCENT1, fg=WHITE).pack(side="left", padx=20, pady=10)
        tk.Label(hdr, text="A = U · Σ · Vᵀ",
                 font=("Courier New", 12, "italic"), bg=ACCENT1, fg="#d8b4fe").pack(side="right", padx=20)

        # ── main layout: left controls | centre images | right metrics ──────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=8, pady=6)

        left   = tk.Frame(body, bg=BG2, width=220)
        centre = tk.Frame(body, bg=BG)
        right  = tk.Frame(body, bg=BG2, width=240)

        left.pack(side="left", fill="y", padx=(0,6))
        right.pack(side="right", fill="y", padx=(6,0))
        centre.pack(side="left", fill="both", expand=True)

        left.pack_propagate(False)
        right.pack_propagate(False)

        self._build_controls(left)
        self._build_images(centre)
        self._build_metrics(right)

    # ── left panel ─────────────────────────────────────────────────────────

    def _build_controls(self, parent):
        def section(text):
            f = tk.Frame(parent, bg=BG2)
            f.pack(fill="x", padx=8, pady=(10,2))
            tk.Label(f, text=text, font=FONT_HEAD, bg=BG2, fg=ACCENT2).pack(anchor="w")
            sep = tk.Frame(parent, bg=ACCENT2, height=1)
            sep.pack(fill="x", padx=8, pady=(0,6))

        def styled_btn(parent, text, cmd, color=ACCENT1):
            b = tk.Button(parent, text=text, command=cmd,
                          bg=color, fg=WHITE, font=FONT_BODY,
                          relief="flat", cursor="hand2",
                          activebackground=ACCENT2, activeforeground=WHITE,
                          padx=10, pady=6)
            b.pack(fill="x", padx=10, pady=3)
            return b

        # ── File ──────────────────────────────────────────────────────────
        section("📂  File")
        styled_btn(parent, "Load Image", self.load_image, ACCENT1)
        self.btn_save = styled_btn(parent, "💾  Save Compressed", self.save_image, GREEN)
        self.btn_save.config(state="disabled")

        # ── SVD Rank ──────────────────────────────────────────────────────
        section("🔢  SVD Rank  (k)")
        tk.Label(parent, text="Controls how many singular values\nare kept in reconstruction.",
                 font=FONT_SMALL, bg=BG2, fg=SUBTEXT, justify="left").pack(anchor="w", padx=10)

        self.lbl_k = tk.Label(parent, text="k = 50", font=FONT_HEAD, bg=BG2, fg=ACCENT3)
        self.lbl_k.pack(pady=(4,0))

        self.slider_k = ttk.Scale(parent, from_=1, to=300,
                                  variable=self.var_k,
                                  command=self._on_k_changed)
        self.slider_k.pack(fill="x", padx=10, pady=4)

        # fine-grain spinbox
        sb_frame = tk.Frame(parent, bg=BG2)
        sb_frame.pack(fill="x", padx=10, pady=(0,4))
        tk.Label(sb_frame, text="Exact k:", font=FONT_SMALL, bg=BG2, fg=SUBTEXT).pack(side="left")
        self.spin_k = tk.Spinbox(sb_frame, from_=1, to=300, textvariable=self.var_k,
                                  font=FONT_SMALL, bg=CARD, fg=WHITE, width=5,
                                  command=lambda: self._on_k_changed(None))
        self.spin_k.pack(side="right")

        # ── Output Quality ────────────────────────────────────────────────
        section("🎚️  Output JPEG Quality")
        tk.Label(parent, text="Affects saved file size.\nHigher = better quality.",
                 font=FONT_SMALL, bg=BG2, fg=SUBTEXT, justify="left").pack(anchor="w", padx=10)

        self.lbl_q = tk.Label(parent, text="Quality = 85", font=FONT_HEAD, bg=BG2, fg=GREEN)
        self.lbl_q.pack(pady=(4,0))

        self.slider_q = ttk.Scale(parent, from_=1, to=100,
                                   variable=self.var_quality,
                                   command=self._on_quality_changed)
        self.slider_q.pack(fill="x", padx=10, pady=4)

        # ── Analysis ──────────────────────────────────────────────────────
        section("📉  Analysis")
        styled_btn(parent, "Compression vs Quality", self.plot_compression_curve, ACCENT3)

        # ── status bar ────────────────────────────────────────────────────
        tk.Frame(parent, bg=BG2).pack(fill="y", expand=True)
        self.lbl_status = tk.Label(parent, text="No image loaded",
                                   font=FONT_SMALL, bg=BG2, fg=SUBTEXT,
                                   wraplength=200, justify="center")
        self.lbl_status.pack(padx=8, pady=8)

    # ── centre panel (side-by-side) ─────────────────────────────────────────

    def _build_images(self, parent):
        # column headers
        hdr = tk.Frame(parent, bg=BG)
        hdr.pack(fill="x", pady=(4,0))

        for title, color in [("📷  Original", ACCENT2), ("🗜️  Compressed (SVD)", ACCENT1)]:
            col = tk.Frame(hdr, bg=BG)
            col.pack(side="left", expand=True, fill="x")
            tk.Label(col, text=title, font=FONT_HEAD, bg=BG, fg=color).pack()

        # image frames
        img_row = tk.Frame(parent, bg=BG)
        img_row.pack(fill="both", expand=True, pady=4)

        lf = tk.Frame(img_row, bg=CARD, bd=2, relief="flat")
        rf = tk.Frame(img_row, bg=CARD, bd=2, relief="flat")
        lf.pack(side="left", expand=True, fill="both", padx=(0,4))
        rf.pack(side="left", expand=True, fill="both", padx=(4,0))

        self.lbl_orig = tk.Label(lf, bg=CARD, text="Load an image to begin",
                                  fg=SUBTEXT, font=FONT_BODY)
        self.lbl_orig.pack(expand=True, fill="both")

        self.lbl_comp = tk.Label(rf, bg=CARD, text="Compressed image appears here",
                                  fg=SUBTEXT, font=FONT_BODY)
        self.lbl_comp.pack(expand=True, fill="both")

        # size bars below images
        size_row = tk.Frame(parent, bg=BG)
        size_row.pack(fill="x", pady=(0,4))

        lf2 = tk.Frame(size_row, bg=BG)
        rf2 = tk.Frame(size_row, bg=BG)
        lf2.pack(side="left", expand=True, fill="x")
        rf2.pack(side="left", expand=True, fill="x")

        self.lbl_orig_info = tk.Label(lf2, text="—", font=FONT_SMALL, bg=BG, fg=SUBTEXT)
        self.lbl_orig_info.pack()
        self.lbl_comp_info = tk.Label(rf2, text="—", font=FONT_SMALL, bg=BG, fg=SUBTEXT)
        self.lbl_comp_info.pack()

        # progress bar for compression
        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=(0,4))

    # ── right panel (metrics) ────────────────────────────────────────────────

    def _build_metrics(self, parent):
        tk.Label(parent, text="📊  Metrics", font=FONT_HEAD, bg=BG2, fg=ACCENT3).pack(pady=(10,2), padx=8, anchor="w")
        tk.Frame(parent, bg=ACCENT3, height=1).pack(fill="x", padx=8, pady=(0,8))

        # metric boxes
        self.metric_vars = {}
        metrics = [
            ("Energy Retained",   "%",     "Σσᵢ² / Σσᵢ² × 100"),
            ("Compression Ratio", "×",     "Original / SVD storage"),
            ("Rank (r)",          "",      "True rank of image"),
            ("k used",            "",      "Singular values kept"),
            ("Orig Size",         "KB",    "JPEG @ chosen quality"),
            ("Comp Size",         "KB",    "JPEG @ chosen quality"),
            ("Size Reduction",    "%",     "Space saved"),
        ]

        for key, unit, tooltip in metrics:
            frame = tk.Frame(parent, bg=CARD)
            frame.pack(fill="x", padx=8, pady=2)

            tk.Label(frame, text=key, font=FONT_SMALL, bg=CARD, fg=SUBTEXT,
                     width=16, anchor="w").grid(row=0, column=0, sticky="w", padx=6, pady=2)

            var = tk.StringVar(value="—")
            self.metric_vars[key] = var
            tk.Label(frame, textvariable=var, font=("Courier New", 10, "bold"),
                     bg=CARD, fg=WHITE, width=10, anchor="e").grid(row=0, column=1, sticky="e", padx=(0,4))

            if unit:
                tk.Label(frame, text=unit, font=FONT_SMALL, bg=CARD, fg=ACCENT2,
                         width=3, anchor="w").grid(row=0, column=2, padx=(0,4))

        # quality indicator bar
        tk.Label(parent, text="PSNR Quality Band", font=FONT_SMALL, bg=BG2, fg=SUBTEXT).pack(padx=8, pady=(10,2), anchor="w")
        self.quality_canvas = tk.Canvas(parent, bg=BG2, height=24, highlightthickness=0)
        self.quality_canvas.pack(fill="x", padx=8, pady=(0,6))
        self._draw_quality_bar(None)

        # math note
        note = (
            "Low-rank theorem:\n"
            "Aₖ = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ\n"
            "minimises ‖A−Aₖ‖_F\n"
            "over all rank-k matrices."
        )
        tk.Label(parent, text=note, font=FONT_MONO, bg=BG2, fg=SUBTEXT,
                 justify="left", wraplength=220).pack(padx=10, pady=8, anchor="w")

    # ─── event handlers ───────────────────────────────────────────────────────

    def _on_k_changed(self, _):
        k = self.var_k.get()
        self.lbl_k.config(text=f"k = {k}")
        self._schedule_compress(k)

    def _on_quality_changed(self, _):
        q = self.var_quality.get()
        self.lbl_q.config(text=f"Quality = {q}")
        if self.image_bgr is not None and self.compressed_bgr is not None:
            self._refresh_size_labels()

    def _schedule_compress(self, k: int):
        """Debounce compression: only run after slider settles."""
        self._pending_k = k
        if self._compress_thread and self._compress_thread.is_alive():
            return
        self._compress_thread = threading.Thread(target=self._compress_worker, daemon=True)
        self._compress_thread.start()

    def _compress_worker(self):
        time.sleep(0.08)                          # tiny debounce
        k = self._pending_k
        if k is None or self.image_bgr is None:
            return

        self.root.after(0, self.progress.start)
        try:
            compressed, info = svd_compress(self.image_bgr, k)
            self.compressed_bgr = compressed
            self.root.after(0, lambda: self._update_display(compressed, info))
        finally:
            self.root.after(0, self.progress.stop)

    # ─── display helpers ──────────────────────────────────────────────────────

    def _imgtk(self, img_bgr: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
        pil = pil_from_bgr(img_bgr)
        pil = resize_fit(pil, w, h)
        return ImageTk.PhotoImage(pil)

    def _update_display(self, compressed: np.ndarray, info: dict):
        self._show_original()
        self._show_compressed(compressed)
        self._refresh_size_labels()
        self._update_metrics(info)
        self._draw_quality_bar(info["psnr"])
        self.btn_save.config(state="normal")

    def _show_original(self):
        if self.image_bgr is None:
            return
        tk_img = self._imgtk(self.image_bgr, self.PREVIEW_W, self.PREVIEW_H)
        self.lbl_orig.config(image=tk_img, text="")
        self.lbl_orig._img = tk_img

    def _show_compressed(self, img: np.ndarray):
        tk_img = self._imgtk(img, self.PREVIEW_W, self.PREVIEW_H)
        self.lbl_comp.config(image=tk_img, text="")
        self.lbl_comp._img = tk_img

    def _refresh_size_labels(self):
        q = self.var_quality.get()
        if self.image_bgr is not None:
            kb_o = file_size_kb(self.image_bgr, q)
            self.lbl_orig_info.config(
                text=f"{self.image_bgr.shape[1]}×{self.image_bgr.shape[0]}  |  {kb_o:.1f} KB (JPEG)",
                fg=ACCENT2)
            self.metric_vars["Orig Size"].set(f"{kb_o:.1f}")

        if self.compressed_bgr is not None:
            kb_c = file_size_kb(self.compressed_bgr, q)
            reduction = 0.0
            if self.image_bgr is not None:
                kb_o = file_size_kb(self.image_bgr, q)
                reduction = (1 - kb_c / max(kb_o, 1)) * 100
            self.lbl_comp_info.config(
                text=f"k = {self.var_k.get()}  |  {kb_c:.1f} KB  |  {reduction:.1f}% smaller",
                fg=ACCENT1)
            self.metric_vars["Comp Size"].set(f"{kb_c:.1f}")
            self.metric_vars["Size Reduction"].set(f"{reduction:.1f}")

    def _update_metrics(self, info: dict):
        psnr_val = info["psnr"]
        self.metric_vars["Energy Retained"].set(f"{info['energy_mean']:.2f}")
        self.metric_vars["Compression Ratio"].set(f"{info['compression_ratio']:.2f}")
        self.metric_vars["Rank (r)"].set(str(info["rank"]))
        self.metric_vars["k used"].set(str(info["k"]))

        self.lbl_status.config(
            text=f"k={info['k']}  PSNR={psnr_val:.1f}dB\n"
                 f"Energy={info['energy_mean']:.1f}%  CR={info['compression_ratio']:.1f}×",
            fg=GREEN)

    def _draw_quality_bar(self, psnr_val):
        c = self.quality_canvas
        c.delete("all")
        c.update_idletasks()
        W = c.winfo_width() or 200
        H = 24

        # gradient bar
        bands = [(0, RED, "Poor"), (0.33, ACCENT3, "Fair"),
                 (0.66, GREEN, "Good"), (0.85, ACCENT2, "Excellent")]
        bw = W // len(bands)
        for i, (_, col, lbl) in enumerate(bands):
            x0, x1 = i*bw, (i+1)*bw
            c.create_rectangle(x0, 4, x1, H-4, fill=col, outline="")
            c.create_text((x0+x1)//2, H//2, text=lbl, font=FONT_SMALL, fill=BG)

        # marker
        if psnr_val and psnr_val != float("inf"):
            # PSNR scale: < 20 poor, 20-30 fair, 30-40 good, >40 excellent
            norm = min(1.0, max(0.0, (psnr_val - 10) / 40))
            x = int(norm * W)
            c.create_polygon(x-6, 2, x+6, 2, x, H-2, fill=WHITE, outline="")

    # ─── file actions ─────────────────────────────────────────────────────────

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All", "*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read image.")
            return
        self.image_path = path
        self.image_bgr  = img

        # update slider max to image rank
        r = min(img.shape[:2])
        self.slider_k.config(to=r)
        self.spin_k.config(to=r)

        k = min(self.var_k.get(), r)
        self.var_k.set(k)

        self._show_original()
        self.lbl_status.config(text=f"Loaded: {os.path.basename(path)}\n{img.shape[1]}×{img.shape[0]}", fg=ACCENT2)
        self._schedule_compress(k)

    def save_image(self):
        if self.compressed_bgr is None:
            return
        default_name = "compressed_svd_k{}.jpg".format(self.var_k.get())
        path = filedialog.asksaveasfilename(
            title="Save Compressed Image",
            initialfile=default_name,
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"),
                       ("BMP", "*.bmp"), ("All", "*.*")])
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            q = self.var_quality.get()
            cv2.imwrite(path, self.compressed_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        elif ext == ".png":
            cv2.imwrite(path, self.compressed_bgr)
        else:
            cv2.imwrite(path, self.compressed_bgr)

        kb = os.path.getsize(path) / 1024
        messagebox.showinfo("Saved",
            f"Image saved to:\n{path}\n\nFile size: {kb:.1f} KB\nJPEG quality: {self.var_quality.get()}")

    # ─── plot windows ─────────────────────────────────────────────────────────

    def _open_plot_window(self, title: str, fig: plt.Figure):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg=BG)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        tk.Button(win, text="Close", command=win.destroy,
                  bg=RED, fg=WHITE, font=FONT_BODY).pack(pady=6)

    def plot_compression_curve(self):
        if self.image_bgr is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        ks     = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200]
        r_max  = min(self.image_bgr.shape[:2])
        ks     = [k for k in ks if k <= r_max]
        sizes  = []
        psnrs2 = []
        q      = self.var_quality.get()

        for ki in ks:
            comp, _ = svd_compress(self.image_bgr, ki)
            sizes.append(file_size_kb(comp, q))
            psnrs2.append(psnr(self.image_bgr, comp))

        orig_kb = file_size_kb(self.image_bgr, q)

        fig, ax1 = plt.subplots(figsize=(10, 5), facecolor="#0d0d1a")
        ax1.set_facecolor(CARD)
        for sp in ax1.spines.values(): sp.set_color(SUBTEXT)
        ax1.tick_params(colors=SUBTEXT)
        ax1.xaxis.label.set_color(SUBTEXT)
        ax1.yaxis.label.set_color(SUBTEXT)

        ax1.bar(ks, sizes, color=ACCENT1, alpha=0.7, width=[max(1, k//4) for k in ks], label="File size (KB)")
        ax1.axhline(orig_kb, color=ACCENT3, linestyle="--", linewidth=1.5, label=f"Original ({orig_kb:.1f} KB)")
        ax1.set_xlabel("k  (SVD rank)")
        ax1.set_ylabel("JPEG file size (KB)", color=ACCENT1)
        ax1.set_title("File Size & PSNR vs SVD Rank k", color=WHITE, fontsize=12)

        ax2 = ax1.twinx()
        ax2.plot(ks, psnrs2, color=ACCENT2, linewidth=2, marker="o", markersize=5, label="PSNR (dB)")
        ax2.set_ylabel("PSNR (dB)", color=ACCENT2)
        ax2.tick_params(colors=ACCENT2)

        lines1, lbl1 = ax1.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, lbl1+lbl2, facecolor=CARD, labelcolor=WHITE, edgecolor=SUBTEXT)

        fig.tight_layout()
        self._open_plot_window("Compression vs Quality", fig)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()

    # style ttk sliders
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Horizontal.TScale",
                    background=BG2, troughcolor=CARD,
                    sliderlength=18, sliderrelief="flat")
    style.configure("TProgressbar", troughcolor=BG2, background=ACCENT1)

    app = SVDApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
