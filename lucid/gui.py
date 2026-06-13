#!/usr/bin/env python3
"""Premium Tkinter GUI wrapper for LUCID pipeline.

Provides a modern dark-themed dashboard with complete parameter control,
real-time process tailing, and colored status logs.
"""

import os
import subprocess
import sys
import threading
from tkinter import (
    BooleanVar,
    DoubleVar,
    IntVar,
    StringVar,
    Tk,
    filedialog,
    ttk,
)
from tkinter import messagebox as msg

# Colors for sleek dark mode
BG_COLOR = "#121214"
SURFACE_COLOR = "#1a1a1e"
ACCENT_COLOR = "#4F46E5"      # Vibrant Indigo
ACCENT_HOVER = "#4338CA"
TEXT_COLOR = "#E2E8F0"        # Sleek off-white
TEXT_MUTED = "#94A3B8"        # Soft gray
BORDER_COLOR = "#2D3748"      # Charcoal border
GREEN_COLOR = "#10B981"
RED_COLOR = "#EF4444"

# ttk style names used as constants so changes don't drift.
STYLE_SUB_LABEL = "Sub.TLabel"
STYLE_SECONDARY_BUTTON = "Secondary.TButton"


class LucidGUI:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("LUCID - SISR Dataset Curation")
        self.root.geometry("800x720")
        self.root.minsize(700, 620)
        self.root.configure(bg=BG_COLOR)

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.configure_styles()

        # Variables
        self.input_path = StringVar()
        self.output_path = StringVar()
        self.tile_size = StringVar(value="512")   # default 512; ICNet trained for this size
        self.threshold = DoubleVar(value=0.45)
        self.dedupe_threshold = DoubleVar(value=0.96)
        self.workers = IntVar(value=4)
        self.batch_size = IntVar(value=0)
        self.multiscale = BooleanVar(value=True)  # default ON for SISR GT
        self.resume = BooleanVar(value=False)
        self.dry_run = BooleanVar(value=False)
        self.cleanup = BooleanVar(value=False)
        self.stage = StringVar(value="all")

        self.running_thread: threading.Thread | None = None
        self.process: subprocess.Popen[str] | None = None

        self.build_ui()

    def configure_styles(self):
        # Configure overall style map for custom dark theme look
        self.style.configure(".", background=BG_COLOR, foreground=TEXT_COLOR, font=("Inter", 10))
        self.style.configure("TFrame", background=BG_COLOR)

        # Label Styles
        self.style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR)
        self.style.configure("Header.TLabel", font=("Inter", 14, "bold"), foreground=TEXT_COLOR)
        self.style.configure(STYLE_SUB_LABEL, font=("Inter", 9), foreground=TEXT_MUTED)
        self.style.configure("Section.TLabel", font=("Inter", 11, "bold"), foreground=ACCENT_COLOR)

        # Entry Style
        self.style.configure(
            "TEntry",
            fieldbackground=SURFACE_COLOR,
            background=BORDER_COLOR,
            foreground=TEXT_COLOR,
            bordercolor=BORDER_COLOR,
        )

        # Button Style
        self.style.configure(
            "TButton",
            background=ACCENT_COLOR,
            foreground=TEXT_COLOR,
            borderwidth=0,
            focuscolor="",
            font=("Inter", 10, "bold"),
            padding=(10, 5),
        )
        self.style.map("TButton", background=[("active", ACCENT_HOVER)])

        self.style.configure(
            STYLE_SECONDARY_BUTTON,
            background=SURFACE_COLOR,
            foreground=TEXT_COLOR,
            borderwidth=1,
            bordercolor=BORDER_COLOR,
        )
        self.style.map(STYLE_SECONDARY_BUTTON, background=[("active", "#2d2d34")])

        # Checkbutton Style
        self.style.configure("TCheckbutton", background=BG_COLOR, foreground=TEXT_COLOR, focuscolor="")
        self.style.map("TCheckbutton", background=[("active", BG_COLOR)])

        # OptionMenu Style
        self.style.configure(
            "TMenubutton",
            background=SURFACE_COLOR,
            foreground=TEXT_COLOR,
            borderwidth=1,
            bordercolor=BORDER_COLOR,
        )

    def build_ui(self):
        # Main Layout container
        main_container = ttk.Frame(self.root, padding=20)
        main_container.pack(fill="both", expand=True)

        # Header Title
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill="x", pady=(0, 15))
        ttk.Label(title_frame, text="LUCID Dataset Curator", style="Header.TLabel").pack(side="left")
        ttk.Label(title_frame, text="v3.0.0",             style=STYLE_SUB_LABEL).pack(side="left", padx=10, pady=(4, 0))

        # --- FOLDER CHOOSER BLOCK ---
        folders_frame = ttk.LabelFrame(main_container, text=" Paths ", padding=15)
        folders_frame.pack(fill="x", pady=(0, 15))

        # Grid layout for inputs
        folders_frame.columnconfigure(1, weight=1)

        # Input Row
        ttk.Label(folders_frame, text="Input Folder:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(folders_frame, textvariable=self.input_path).grid(
            row=0, column=1, sticky="ew", padx=10, pady=5
        )
        ttk.Button(
            folders_frame,
            text="Browse",
            style=STYLE_SECONDARY_BUTTON,
            command=lambda: self.input_path.set(filedialog.askdirectory() or self.input_path.get()),
        ).grid(row=0, column=2, pady=5)

        # Output Row
        ttk.Label(folders_frame, text="Output Folder:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(folders_frame, textvariable=self.output_path).grid(
            row=1, column=1, sticky="ew", padx=10, pady=5
        )
        ttk.Button(
            folders_frame,
            text="Browse",
            style=STYLE_SECONDARY_BUTTON,
            command=lambda: self.output_path.set(filedialog.askdirectory() or self.output_path.get()),
        ).grid(row=1, column=2, pady=5)

        # --- PARAMETERS BLOCK ---
        params_container = ttk.Frame(main_container)
        params_container.pack(fill="x", pady=(0, 15))

        # Config Panel Left — complexity / dedupe sliders
        config_panel = ttk.LabelFrame(params_container, text=" Configuration ", padding=15)
        config_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))
        config_panel.columnconfigure(1, weight=1)

        # Complexity slider
        ttk.Label(config_panel, text="Complexity Threshold (ICNet):").grid(
            row=0, column=0, sticky="w", pady=5
        )
        slider_frame = ttk.Frame(config_panel)
        slider_frame.grid(row=0, column=1, sticky="ew", padx=10)
        slider_frame.columnconfigure(0, weight=1)
        self.threshold_label = ttk.Label(slider_frame, text="0.45", width=4)
        sld = ttk.Scale(
            slider_frame,
            from_=0.10,
            to=1.00,
            variable=self.threshold,
            command=lambda v: self._on_threshold_slider(float(v)),
        )
        sld.grid(row=0, column=0, sticky="ew")
        self.threshold_label.grid(row=0, column=1, padx=(5, 0))

        # Dedupe slider
        ttk.Label(config_panel, text="Deduplication Similarity:").grid(
            row=1, column=0, sticky="w", pady=5
        )
        slider_frame2 = ttk.Frame(config_panel)
        slider_frame2.grid(row=1, column=1, sticky="ew", padx=10)
        slider_frame2.columnconfigure(0, weight=1)
        lbl_val2 = ttk.Label(slider_frame2, text="0.96", width=4)
        sld2 = ttk.Scale(
            slider_frame2,
            from_=0.80,
            to=0.99,
            variable=self.dedupe_threshold,
            command=lambda v: lbl_val2.config(text=f"{float(v):.2f}"),
        )
        sld2.grid(row=0, column=0, sticky="ew")
        lbl_val2.grid(row=0, column=1, padx=(5, 0))

        # Advanced Panel Right — tile size, workers, stage
        adv_panel = ttk.LabelFrame(params_container, text=" Advanced System Settings ", padding=15)
        adv_panel.pack(side="right", fill="both", expand=True, padx=(5, 0))
        adv_panel.columnconfigure(1, weight=1)

        ttk.Label(adv_panel, text="Tile Size (px):").grid(row=0, column=0, sticky="w", pady=5)
        tile_size_combo = ttk.Combobox(
            adv_panel,
            textvariable=self.tile_size,
            values=["128", "256", "512", "1024"],
            width=8,
            state="readonly",
        )
        tile_size_combo.grid(row=0, column=1, sticky="w", padx=10)
        tile_size_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_calibration_badge())

        # Status badge: Native (512), Calibrated (256/128/1024), or Approximate
        self.calibration_badge = ttk.Label(
            adv_panel, text="",             style=STYLE_SUB_LABEL
        )
        self.calibration_badge.grid(row=0, column=2, sticky="w", padx=10)
        self._refresh_calibration_badge()

        self.btn_calibrate = ttk.Button(
            adv_panel,
            text="Calibrate…",
            style=STYLE_SECONDARY_BUTTON,
            command=self._open_calibrate,
        )
        self.btn_calibrate.grid(row=0, column=3, sticky="w", padx=(5, 0))

        ttk.Label(adv_panel, text="CPU Workers:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Spinbox(adv_panel, from_=1, to=32, textvariable=self.workers, width=8).grid(
            row=1, column=1, sticky="w", padx=10
        )

        ttk.Label(adv_panel, text="Stage:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Combobox(
            adv_panel,
            textvariable=self.stage,
            values=["all", "ingest", "dedupe", "finalize"],
            width=8,
            state="readonly",
        ).grid(row=2, column=1, sticky="w", padx=10)

        # --- FLAGS BLOCK (Bottom of Settings) ---
        flags_frame = ttk.Frame(main_container)
        flags_frame.pack(fill="x", pady=(0, 10))

        ttk.Checkbutton(flags_frame, text="Enable Multi-Scale", variable=self.multiscale).pack(
            side="left", padx=10
        )
        ttk.Checkbutton(flags_frame, text="Resume Interrupted Run", variable=self.resume).pack(
            side="left", padx=10
        )
        ttk.Checkbutton(flags_frame, text="Dry Run (Estimations Only)", variable=self.dry_run).pack(
            side="left", padx=10
        )
        ttk.Checkbutton(
            flags_frame, text="Auto-Cleanup Intermediate Files", variable=self.cleanup
        ).pack(side="left", padx=10)

        # --- LOG CONSOLE BLOCK ---
        console_frame = ttk.LabelFrame(main_container, text=" Execution Output Console ", padding=10)
        console_frame.pack(fill="both", expand=True, pady=(5, 10))

        from tkinter.scrolledtext import ScrolledText
        self.console = ScrolledText(
            console_frame,
            bg=SURFACE_COLOR,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            font=("Courier New", 9.5),
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
        )
        self.console.pack(fill="both", expand=True)

        # Log coloring tags
        self.console.tag_config("INFO", foreground=GREEN_COLOR)
        self.console.tag_config("WARN", foreground="#F59E0B")
        self.console.tag_config("ERR", foreground=RED_COLOR)
        self.console.tag_config("HEADER", foreground=ACCENT_COLOR, font=("Courier New", 10, "bold"))

        # --- CONTROL ACTION ROW ---
        action_frame = ttk.Frame(main_container)
        action_frame.pack(fill="x", pady=5)

        self.btn_run = ttk.Button(
            action_frame, text="▶  Run Unified Pipeline", command=self.start_pipeline
        )
        self.btn_run.pack(side="right", padx=5)

        self.btn_cancel = ttk.Button(
            action_frame,
            text="⏹  Stop",
            style=STYLE_SECONDARY_BUTTON,
            command=self.stop_pipeline,
            state="disabled",
        )
        self.btn_cancel.pack(side="right", padx=5)

    def write_console(self, text: str):
        """Write a line to the console. Must be called from the main thread."""
        self.console.config(state="normal")
        # Colorize specific keywords dynamically
        if "INFO" in text or "Complete" in text or "Cached" in text or "Downloaded" in text:
            self.console.insert("end", text, "INFO")
        elif "WARN" in text or "Warning" in text:
            self.console.insert("end", text, "WARN")
        elif "ERROR" in text or "FATAL" in text or "failed" in text:
            self.console.insert("end", text, "ERR")
        elif "===" in text:
            self.console.insert("end", text, "HEADER")
        else:
            self.console.insert("end", text)
        self.console.see("end")
        self.console.config(state="disabled")

    def _write_console_safe(self, text: str):
        """Thread-safe console write: marshals update to the main thread."""
        self.root.after(0, self.write_console, text)

    def start_pipeline(self):
        if not self.input_path.get() or not self.output_path.get():
            msg.showerror("Input Validation Error", "Please specify both the Input and Output directories.")
            return

        self.console.config(state="normal")
        self.console.delete("1.0", "end")
        self.console.config(state="disabled")

        cmd = [
            sys.executable,
            "-u",   # unbuffered stdout/stderr for live log streaming
            "-m",
            "lucid",
            "--input", self.input_path.get(),
            "--output", self.output_path.get(),
            "--tile-size", self.tile_size.get(),
            "--threshold", f"{self.threshold.get():.2f}",
            "--dedupe-threshold", f"{self.dedupe_threshold.get():.2f}",
            "--workers", str(self.workers.get()),
            "--stage", self.stage.get(),
        ]

        if self.multiscale.get():
            cmd.append("--multiscale")
        if self.resume.get():
            cmd.append("--resume")
        if self.dry_run.get():
            cmd.append("--dry-run")
        if self.cleanup.get():
            cmd.append("--cleanup")

        self.write_console(f"=== LAUNCHING PIPELINE ===\nCommand: {' '.join(cmd)}\n\n")
        self.btn_run.config(state="disabled")
        self.btn_cancel.config(state="normal")

        def tail():
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"  # ensure subprocess flushes output line-by-line
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                for line in iter(self.process.stdout.readline, ""):
                    # Marshal Tk widget updates to the main thread (Tk is not thread-safe)
                    self._write_console_safe(line)

                self.process.wait()
                if self.process.returncode == 0:
                    self.root.after(
                        0,
                        lambda: msg.showinfo("LUCID Pipeline Success", "Unified SISR Filtering complete!"),
                    )
                else:
                    self._write_console_safe(
                        f"\n[ERROR] Pipeline exited with code {self.process.returncode}\n"
                    )
            except Exception as e:
                self._write_console_safe(f"\n[FATAL ERROR] Exec failed: {e}\n")
            finally:
                self.root.after(0, self._reset_buttons)
                self.process = None

        self.running_thread = threading.Thread(target=tail, daemon=True)
        self.running_thread.start()

    def _reset_buttons(self):
        """Reset button states; must be called from the main thread."""
        self.btn_run.config(state="normal")
        self.btn_cancel.config(state="disabled")

    def stop_pipeline(self):
        if self.process:
            self.write_console("\n=== TERMINATING ACTIVE PIPELINE RUN ===\n")
            self.process.terminate()
            # Allow the tail thread to detect EOF and clean up buttons

    def _refresh_calibration_badge(self) -> None:
        """Update the calibration status label next to the tile-size dropdown
        and snap the threshold slider to the suggested value (if the user
        hasn't manually moved it for this tile size)."""
        from lucid.resources.icnet_calibration import (
            calibration_status_label,
            suggested_threshold,
        )
        try:
            size = int(self.tile_size.get())
        except (TypeError, ValueError):
            size = 0
        self.calibration_badge.config(text=calibration_status_label(size))
        # If the slider hasn't been manually moved for this size, snap it
        if getattr(self, "_threshold_user_moved", False) and getattr(
            self, "_threshold_user_size", None
        ) == size:
            return
        suggested = suggested_threshold(size)
        if 0.0 < suggested < 1.0:
            self.threshold.set(suggested)
            self.threshold_label.config(text=f"{suggested:.2f}")

    def _on_threshold_slider(self, value: float) -> None:
        """Called when the user moves the threshold slider; remembers the
        size so subsequent tile-size changes don't clobber their choice."""
        self.threshold_label.config(text=f"{value:.2f}")
        try:
            size = int(self.tile_size.get())
        except (TypeError, ValueError):
            size = 0
        self._threshold_user_moved = True
        self._threshold_user_size = size

    def _open_calibrate(self) -> None:
        """Open a folder picker and spawn the `lucid-calibrate` subcommand."""
        folder = filedialog.askdirectory(
            title="Pick a folder of representative images for calibration"
        )
        if not folder:
            return
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "lucid.calibrate",
                    "--input", folder,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            msg.showerror("Calibration launch failed", str(e))
            return

        self.write_console(
            f"=== RUNNING CALIBRATION on {folder} ===\n"
        )
        self.btn_calibrate.config(state="disabled")

        def tail():
            try:
                for line in iter(self.process.stdout.readline, ""):
                    self._write_console_safe(line)
                self.process.wait()
                if self.process.returncode == 0:
                    self.root.after(0, self._refresh_calibration_badge)
                    self.root.after(
                        0,
                        lambda: msg.showinfo(
                            "Calibration complete",
                            "New calibration saved to lucid/resources/. "
                            "Restart any running pipeline to pick it up.",
                        ),
                    )
                else:
                    self._write_console_safe(
                        f"\n[ERROR] Calibration exited with code "
                        f"{self.process.returncode}\n"
                    )
            except Exception as e:
                self._write_console_safe(f"\n[FATAL ERROR] Calibrate exec failed: {e}\n")
            finally:
                self.root.after(0, lambda: self.btn_calibrate.config(state="normal"))
                self.process = None

        threading.Thread(target=tail, daemon=True).start()


def run_gui() -> None:
    root = Tk()
    LucidGUI(root)
    root.mainloop()


def main() -> int:
    try:
        run_gui()
        return 0
    except Exception as e:
        print(f"GUI launch failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
