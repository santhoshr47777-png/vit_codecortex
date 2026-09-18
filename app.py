
import os, sys, random, json, re, threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
from src.training_engine import train_all

BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
DATA_EMAIL = BASE / "data" / "email" / "emails.csv"
DATA_MALWARE = BASE / "data" / "malware" / "malware.csv"
MODELS = BASE / "models"
REPORTS = BASE / "reports"

BG = "#f5f7fb"; WHITE="#ffffff"; TEXT="#17202a"; MUTED="#6b7785"
BLUE="#0879ad"; BLUE2="#e8f4fb"; RED="#d92d2d"; RED2="#fdecec"
ORANGE="#d97706"; ORANGE2="#fff4df"; GREEN="#087f5b"; GREEN2="#e9f8f2"
BORDER="#dfe5ec"; NAV="#ffffff"; NAV_ACTIVE="#087fb4"

def pct(x): return f"{float(x):.1f}%"

def email_behavior(text):
    words = re.findall(r"\b\w+\b", str(text))
    low = {w.lower() for w in words}
    urls = re.findall(r"https?://\S+|www\.\S+", str(text), re.I)
    urgent={"urgent","immediately","now","hurry","alert","warning","important","action","expire","limited"}
    finance={"money","cash","payment","bank","account","credit","loan","refund","prize","winner","reward","bonus"}
    actions={"click","claim","verify","confirm","login","download","open","submit","activate","register","unsubscribe"}
    letters=[c for c in str(text) if c.isalpha()]
    upper=sum(c.isupper() for c in letters)
    return {
        "Words":len(words),"URLs":len(urls),
        "Urgency terms":len(low&urgent),"Financial terms":len(low&finance),
        "Action terms":len(low&actions),"Uppercase %":(upper/len(letters)*100 if letters else 0),
        "Exclamations":str(text).count("!")
    }

class ThreatLensDesktop(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ThreatLens X - Enterprise SOC")
        self.geometry("1500x920"); self.minsize(1180,720); self.configure(bg=BG)
        self.email_df=self.read_email(); self.mal_df=self.read_malware()
        self.email_model=self.load("email_model.joblib")
        self.email_vec=self.load("email_tfidf.joblib")
        self.mal_model=self.load("malware_model.joblib")
        self.mal_features=self.load("malware_features.joblib")
        self.email_row=None; self.mal_row=None
        self.setup_style(); self.build_shell(); self.show_overview()

    def load(self,n):
        try:return joblib.load(MODELS/n)
        except:return None
    def read_email(self):
        try:return pd.read_csv(DATA_EMAIL)
        except:return pd.DataFrame(columns=["text","spam"])
    def read_malware(self):
        try:return pd.read_csv(DATA_MALWARE,sep="|")
        except:return pd.DataFrame()

    def setup_style(self):
        s=ttk.Style(self); s.theme_use("clam")
        s.configure("Treeview",background=WHITE,fieldbackground=WHITE,foreground=TEXT,rowheight=31,bordercolor=BORDER)
        s.configure("Treeview.Heading",background="#eef3f8",foreground=TEXT,font=("Segoe UI",9,"bold"))
        s.map("Treeview",background=[("selected","#dff1fa")],foreground=[("selected",TEXT)])
        s.configure("TNotebook",background=BG,borderwidth=0)
        s.configure("TNotebook.Tab",background="#eaf0f5",foreground=TEXT,padding=(14,8))
        s.map("TNotebook.Tab",background=[("selected",WHITE)])

    def build_shell(self):
        self.top=tk.Frame(self,bg=WHITE,height=62,highlightbackground=BORDER,highlightthickness=1); self.top.pack(fill="x")
        tk.Label(self.top,text="*",font=("Segoe UI",25,"bold"),fg="#0b83ad",bg=WHITE).pack(side="left",padx=(18,7))
        tk.Label(self.top,text="ThreatLens X",font=("Segoe UI",15,"bold"),fg=TEXT,bg=WHITE).pack(side="left")
        tk.Label(self.top,text=" ENTERPRISE SOC    v4.2",font=("Consolas",8,"bold"),fg="#315a78",bg="#eaf3fa",padx=7,pady=4).pack(side="left",padx=8)
        tk.Label(self.top,text="*  ENGINE ACTIVE",font=("Consolas",8,"bold"),fg=GREEN,bg="#edf9f5",padx=8,pady=5).pack(side="left",padx=20)
        tk.Label(self.top,text="  Global Command Bar     Ctrl+K",font=("Segoe UI",9),fg=MUTED,bg="#f0f4f8",padx=12,pady=5).pack(side="left")
        tk.Button(self.top,text="",command=lambda:self.iconify(),relief="flat",bg=WHITE,fg=MUTED,font=("Segoe UI",13)).pack(side="right",padx=5)
        tk.Button(self.top,text="",command=self.destroy,relief="flat",bg=WHITE,fg=MUTED,font=("Segoe UI",13)).pack(side="right",padx=(2,12))

        self.rail=tk.Frame(self,bg=NAV,width=245,highlightbackground=BORDER,highlightthickness=1); self.rail.pack(side="left",fill="y")
        tk.Label(self.rail,text="WORKSTATION RAIL",font=("Segoe UI",8,"bold"),fg=MUTED,bg=NAV).pack(anchor="w",padx=16,pady=(20,10))
        self.navbuttons=[]
        for label,cmd in [
            ("   Overview / Dashboard",self.show_overview),
            ("   Ingress Queue",self.show_ingress),
            ("   Threat DNA Analyzer",self.show_dna),
            ("   Model Training Center",self.show_training),
            ("[AI]   AI Deep Investigation",self.show_investigation),
            ("[RULES]   Rule Engine & Playbooks",self.show_playbooks)]:
            b=tk.Button(self.rail,text=label,command=cmd,anchor="w",relief="flat",bd=0,bg=NAV,fg=TEXT,font=("Segoe UI",9),padx=16,pady=11)
            b.pack(fill="x",padx=7,pady=1); self.navbuttons.append(b)
        tk.Label(self.rail,text="SYSTEM CONFIG",font=("Segoe UI",8,"bold"),fg=MUTED,bg=NAV).pack(side="bottom",anchor="w",padx=16,pady=(0,6))
        tk.Label(self.rail,text="[SETTINGS]  Settings & Status",font=("Segoe UI",9),fg=TEXT,bg=NAV).pack(side="bottom",anchor="w",padx=16,pady=(0,30))

        self.main=tk.Frame(self,bg=BG); self.main.pack(side="left",fill="both",expand=True)
        self.status=tk.Frame(self,bg="#eef4f8",height=28,highlightbackground=BORDER,highlightthickness=1); self.status.pack(side="bottom",fill="x")
        tk.Label(self.status,text="* CLUSTER: LOCAL-NODE-01 [HEALTHY]     KERNEL: USERSPACE ACTIVE     INGRESS LATENCY: <1ms",font=("Consolas",8),fg="#2a607c",bg="#eef4f8").pack(side="left",padx=12)
        tk.Label(self.status,text="OPERATOR: SOC ANALYST     DATA SOURCE: PROJECT DATASETS",font=("Consolas",8),fg=MUTED,bg="#eef4f8").pack(side="right",padx=12)

    def clear(self):
        for w in self.main.winfo_children():w.destroy()

    def page_header(self,txt,sub="",tag=""):
        f=tk.Frame(self.main,bg=WHITE,highlightbackground=BORDER,highlightthickness=1); f.pack(fill="x",padx=18,pady=(8,10))
        tk.Label(f,text="*  "+txt,font=("Segoe UI",18,"bold"),fg=TEXT,bg=WHITE).pack(side="left",padx=15,pady=14)
        if tag: tk.Label(f,text=tag,font=("Consolas",8,"bold"),fg=RED,bg=RED2,padx=7,pady=4).pack(side="left")
        if sub: tk.Label(f,text=sub,font=("Segoe UI",9),fg=MUTED,bg=WHITE).pack(anchor="w",padx=16,pady=(0,13))

    def metric(self,parent,label,value,note="",accent=BLUE):
        f=tk.Frame(parent,bg=WHITE,highlightbackground=BORDER,highlightthickness=1)
        f.pack(side="left",fill="both",expand=True,padx=5)
        tk.Frame(f,bg=accent,width=4).pack(side="left",fill="y")
        tk.Label(f,text=label.upper(),font=("Segoe UI",8,"bold"),fg=MUTED,bg=WHITE).pack(anchor="w",padx=13,pady=(13,3))
        tk.Label(f,text=str(value),font=("Segoe UI",23,"bold"),fg=TEXT,bg=WHITE).pack(anchor="w",padx=13)
        tk.Label(f,text=note,font=("Segoe UI",8),fg=MUTED,bg=WHITE).pack(anchor="w",padx=13,pady=(2,12))

    def panel(self,parent,title,sub=""):
        f=tk.Frame(parent,bg=WHITE,highlightbackground=BORDER,highlightthickness=1); f.pack(fill="both",expand=True,padx=5,pady=5)
        h=tk.Frame(f,bg=WHITE); h.pack(fill="x")
        tk.Label(h,text=title,font=("Segoe UI",12,"bold"),fg=TEXT,bg=WHITE).pack(side="left",padx=14,pady=(13,3))
        if sub: tk.Label(h,text=sub,font=("Consolas",8),fg="#37627c",bg="#edf5fa",padx=5,pady=3).pack(side="left",padx=5)
        return f

    def show_training(self):
        self.clear()
        self.page_header(
            "Model Training Center",
            "Live training pipeline using the existing email and malware datasets",
            "ML TRAINING"
        )

        top = tk.Frame(self.main, bg=BG)
        top.pack(fill="x", padx=18, pady=4)

        self.metric(
            top,
            "Email Dataset",
            f"{len(self.email_df):,}",
            "Training records",
            BLUE
        )
        self.metric(
            top,
            "Malware Dataset",
            f"{len(self.mal_df):,}",
            "Training records",
            RED
        )
        self.metric(
            top,
            "Learning Mode",
            "FULL RETRAIN",
            "Every application launch",
            ORANGE
        )
        self.metric(
            top,
            "Models",
            "2",
            "Email + Malware",
            GREEN
        )

        panel = tk.Frame(
            self.main,
            bg=WHITE,
            highlightbackground=BORDER,
            highlightthickness=1
        )
        panel.pack(fill="x", padx=23, pady=10)

        tk.Label(
            panel,
            text="LIVE TRAINING PIPELINE",
            font=("Segoe UI", 13, "bold"),
            fg=TEXT,
            bg=WHITE
        ).pack(anchor="w", padx=18, pady=(16, 8))

        self.training_stage = tk.Label(
            panel,
            text="Waiting for training engine...",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT,
            bg=WHITE
        )
        self.training_stage.pack(anchor="w", padx=18, pady=4)

        self.training_detail = tk.Label(
            panel,
            text="",
            font=("Consolas", 9),
            fg=MUTED,
            bg=WHITE
        )
        self.training_detail.pack(anchor="w", padx=18, pady=(0, 8))

        self.training_progress = ttk.Progressbar(
            panel,
            orient="horizontal",
            mode="determinate",
            maximum=100
        )
        self.training_progress.pack(
            fill="x",
            padx=18,
            pady=(2, 16)
        )

        self.training_status = tk.Label(
            panel,
            text="TRAINING ENGINE: INITIALIZING",
            font=("Consolas", 9, "bold"),
            fg=ORANGE,
            bg=ORANGE2,
            padx=10,
            pady=7
        )
        self.training_status.pack(anchor="w", padx=18, pady=(0, 16))

        metrics = tk.Frame(self.main, bg=BG)
        metrics.pack(fill="both", expand=True, padx=18, pady=4)

        self.email_training_panel = tk.Frame(
            metrics,
            bg=WHITE,
            highlightbackground=BORDER,
            highlightthickness=1
        )
        self.email_training_panel.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 6)
        )

        self.mal_training_panel = tk.Frame(
            metrics,
            bg=WHITE,
            highlightbackground=BORDER,
            highlightthickness=1
        )
        self.mal_training_panel.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(6, 0)
        )

        self.render_training_placeholder(
            self.email_training_panel,
            "EMAIL ML MODEL",
            "TF-IDF + Linear SVM"
        )
        self.render_training_placeholder(
            self.mal_training_panel,
            "MALWARE ML MODEL",
            "Random Forest"
        )

        controls = tk.Frame(self.main, bg=BG)
        controls.pack(fill="x", padx=18, pady=(4, 12))

        tk.Button(
            controls,
            text="RETRAIN MODELS",
            command=self.start_training,
            bg=BLUE,
            fg="white",
            activebackground="#06658f",
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padx=18,
            pady=9
        ).pack(side="right")

    def render_training_placeholder(self, parent, title, algorithm):
        tk.Label(
            parent,
            text=title,
            font=("Segoe UI", 12, "bold"),
            fg=TEXT,
            bg=WHITE
        ).pack(anchor="w", padx=16, pady=(14, 3))

        tk.Label(
            parent,
            text=algorithm,
            font=("Consolas", 8, "bold"),
            fg="#37627c",
            bg="#edf5fa",
            padx=6,
            pady=3
        ).pack(anchor="w", padx=16, pady=(0, 12))

        box = tk.Frame(parent, bg="#f7f9fc")
        box.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        tk.Label(
            box,
            text="TRAINING",
            font=("Consolas", 10, "bold"),
            fg=MUTED,
            bg="#f7f9fc"
        ).pack(anchor="w", padx=14, pady=(14, 5))

        tk.Label(
            box,
            text="Waiting for model training...",
            font=("Consolas", 9),
            fg=MUTED,
            bg="#f7f9fc"
        ).pack(anchor="w", padx=14, pady=5)

    def start_training(self):
        if getattr(self, "training_running", False):
            return

        if not hasattr(self, "training_stage"):
            self.show_training()

        self.training_running = True

        self.training_stage.config(
            text="TRAINING IN PROGRESS",
            fg=BLUE
        )
        self.training_detail.config(
            text="Preparing datasets and initializing machine-learning pipeline..."
        )
        self.training_progress["value"] = 0
        self.training_status.config(
            text="TRAINING ENGINE: RUNNING",
            fg=BLUE,
            bg=BLUE2
        )

        thread = threading.Thread(
            target=self._training_worker,
            daemon=True
        )
        thread.start()

    def _training_worker(self):
        try:
            result = train_all(self.training_callback)

            self.after(
                0,
                lambda: self.training_complete(result)
            )

        except Exception as e:
            self.after(
                0,
                lambda: self.training_failed(str(e))
            )

    def training_callback(self, stage, message, percent):
        self.after(
            0,
            lambda: self.update_training_status(
                stage,
                message,
                percent
            )
        )

    def update_training_status(self, stage, message, percent):
        if not hasattr(self, "training_stage"):
            return

        self.training_stage.config(
            text=f"{stage}  |  {percent}%"
        )
        self.training_detail.config(
            text=message
        )
        self.training_progress["value"] = percent

    def training_complete(self, result):
        self.training_running = False

        self.email_model = self.load("email_model.joblib")
        self.email_vec = self.load("email_tfidf.joblib")
        self.mal_model = self.load("malware_model.joblib")
        self.mal_features = self.load("malware_features.joblib")

        email = result["email"]
        malware = result["malware"]

        self.training_stage.config(
            text="TRAINING COMPLETE",
            fg=GREEN
        )
        self.training_detail.config(
            text=f"Completed in {result['seconds']:.1f} seconds. Models saved to models/."
        )
        self.training_progress["value"] = 100

        self.training_status.config(
            text="TRAINING ENGINE: READY",
            fg=GREEN,
            bg=GREEN2
        )

        self.render_training_metrics(
            self.email_training_panel,
            "EMAIL ML MODEL",
            "TF-IDF + Linear SVM",
            email
        )

        self.render_training_metrics(
            self.mal_training_panel,
            "MALWARE ML MODEL",
            "Random Forest",
            malware
        )

    def render_training_metrics(self, parent, title, algorithm, data):
        for w in parent.winfo_children():
            w.destroy()

        tk.Label(
            parent,
            text=title,
            font=("Segoe UI", 12, "bold"),
            fg=TEXT,
            bg=WHITE
        ).pack(anchor="w", padx=16, pady=(14, 3))

        tk.Label(
            parent,
            text=algorithm,
            font=("Consolas", 8, "bold"),
            fg="#37627c",
            bg="#edf5fa",
            padx=6,
            pady=3
        ).pack(anchor="w", padx=16, pady=(0, 10))

        tk.Label(
            parent,
            text=f"Samples: {data['samples']:,}    Features: {data['features']:,}",
            font=("Consolas", 9),
            fg=MUTED,
            bg=WHITE
        ).pack(anchor="w", padx=16, pady=5)

        for name, value in data["metrics"].items():
            row = tk.Frame(parent, bg=WHITE)
            row.pack(fill="x", padx=16, pady=3)

            tk.Label(
                row,
                text=name,
                font=("Segoe UI", 9, "bold"),
                fg=TEXT,
                bg=WHITE
            ).pack(side="left")

            tk.Label(
                row,
                text=f"{value * 100:.2f}%",
                font=("Consolas", 9, "bold"),
                fg=GREEN,
                bg=WHITE
            ).pack(side="right")

        tk.Label(
            parent,
            text="HELD-OUT VALIDATION METRICS",
            font=("Consolas", 8, "bold"),
            fg=GREEN,
            bg=GREEN2,
            padx=7,
            pady=4
        ).pack(anchor="w", padx=16, pady=(12, 14))

    def training_failed(self, error):
        self.training_running = False

        self.training_stage.config(
            text="TRAINING FAILED",
            fg=RED
        )
        self.training_detail.config(
            text=error
        )
        self.training_status.config(
            text="TRAINING ENGINE: ERROR",
            fg=RED,
            bg=RED2
        )

        messagebox.showerror(
            "ThreatLens X - Training Error",
            error
        )

    def show_overview(self):
        self.clear(); self.page_header("Threat Tactical Overview","Dataset-driven email + malware SOC command center","DEFCON 3  //  ELEVATED POSTURE")
        r=tk.Frame(self.main,bg=BG);r.pack(fill="x",padx=18)
        spam=int(self.email_df["spam"].sum()) if "spam" in self.email_df else 0
        mal=int((1-self.mal_df["legitimate"]).sum()) if "legitimate" in self.mal_df else 0
        self.metric(r,"Total Email Samples",len(self.email_df),"Existing dataset",BLUE)
        self.metric(r,"Email Threats",spam,"Spam-labelled",RED)
        self.metric(r,"Malware Samples",len(self.mal_df),"Existing dataset",BLUE)
        self.metric(r,"Malicious Samples",mal,"Dataset-labelled",RED)

        row=tk.Frame(self.main,bg=BG);row.pack(fill="both",expand=True,padx=18,pady=12)
        a=self.panel(row,"Mailstream Gateway","EMAIL INTELLIGENCE")
        tk.Label(a,text="Randomized dataset investigation",font=("Segoe UI",12,"bold"),fg=TEXT,bg=WHITE).pack(anchor="w",padx=16,pady=(16,7))
        tk.Label(a,text=f"{len(self.email_df):,} messages available\n{spam:,} spam-labelled records\n{len(self.email_df)-spam:,} legitimate-labelled records",justify="left",font=("Segoe UI",10),fg=MUTED,bg=WHITE).pack(anchor="w",padx=16)
        tk.Button(a,text="OPEN EMAIL INVESTIGATION",command=self.show_ingress,bg=BLUE,fg="white",relief="flat",font=("Segoe UI",9,"bold"),padx=12,pady=8).pack(anchor="w",padx=16,pady=15)
        b=self.panel(row,"Binaries & Endpoints","MALWARE INTELLIGENCE")
        tk.Label(b,text="Random dataset sample analysis",font=("Segoe UI",12,"bold"),fg=TEXT,bg=WHITE).pack(anchor="w",padx=16,pady=(16,7))
        tk.Label(b,text=f"{len(self.mal_df):,} malware records available\n{mal:,} malicious-labelled records\n{len(self.mal_df)-mal:,} legitimate-labelled records",justify="left",font=("Segoe UI",10),fg=MUTED,bg=WHITE).pack(anchor="w",padx=16)
        tk.Button(b,text="OPEN MALWARE INVESTIGATION",command=self.show_dna,bg="#0d6f96",fg="white",relief="flat",font=("Segoe UI",9,"bold"),padx=12,pady=8).pack(anchor="w",padx=16,pady=15)
        c=self.panel(row,"Investigation Pipeline","UNIFIED ANALYSIS")
        tk.Label(c,text="DETECT  ->  EXPLAIN  ->  COMPARE\n\nTHREAT DNA  ->  NOVELTY  ->  RISK\n\nEMAIL <-> MALWARE CONTEXT\n\nNote: dataset labels are ground truth for their own samples; no unsupported email-to-malware linkage is claimed.",justify="left",font=("Segoe UI",10),fg=MUTED,bg=WHITE).pack(anchor="w",padx=16,pady=18)

    def show_ingress(self):
        self.clear()
        self.page_header(
            "Email Threat Investigation",
            "Investigate randomized messages directly from emails.csv",
            "EMAIL INTELLIGENCE"
        )

        # Scrollable investigation workspace
        outer = tk.Frame(self.main, bg=BG)
        outer.pack(fill="both", expand=True, padx=18, pady=4)

        self.email_canvas = canvas = tk.Canvas(
            outer,
            bg=BG,
            highlightthickness=0,
            bd=0
        )
        scrollbar = ttk.Scrollbar(
            outer,
            orient="vertical",
            command=canvas.yview
        )

        workspace = tk.Frame(canvas, bg=BG)

        workspace.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas_window = canvas.create_window(
            (0, 0),
            window=workspace,
            anchor="nw"
        )

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def resize_workspace(event):
            canvas.itemconfig(
                canvas_window,
                width=event.width
            )

        canvas.bind("<Configure>", resize_workspace)

        # Mouse wheel scrolling
        def mousewheel(event):
            canvas.yview_scroll(
                int(-1 * (event.delta / 120)),
                "units"
            )

        canvas.bind_all("<MouseWheel>", mousewheel)

        # --------------------------------------------------
        # INPUT CONTROL BAR
        # --------------------------------------------------

        control = tk.Frame(
            workspace,
            bg=WHITE,
            highlightbackground=BORDER,
            highlightthickness=1
        )
        control.pack(fill="x", pady=(0, 10))

        tk.Label(
            control,
            text="INVESTIGATION SOURCE",
            font=("Segoe UI", 8, "bold"),
            fg=MUTED,
            bg=WHITE
        ).pack(side="left", padx=(15, 8), pady=13)

        tk.Button(
            control,
            text="[RANDOM] RANDOM EMAIL",
            command=lambda: self.pick_email(None),
            bg=BLUE,
            fg="white",
            activebackground="#06658f",
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=8
        ).pack(side="left", padx=3)

        tk.Button(
            control,
            text="RANDOM SPAM",
            command=lambda: self.pick_email(1),
            bg=RED2,
            fg=RED,
            activebackground="#f9dcdc",
            relief="flat",
            font=("Segoe UI", 8, "bold"),
            padx=12,
            pady=8
        ).pack(side="left", padx=3)

        tk.Button(
            control,
            text="RANDOM LEGITIMATE",
            command=lambda: self.pick_email(0),
            bg=GREEN2,
            fg=GREEN,
            activebackground="#d9f2e8",
            relief="flat",
            font=("Segoe UI", 8, "bold"),
            padx=12,
            pady=8
        ).pack(side="left", padx=3)

        # --------------------------------------------------
        # EMAIL VIEWER
        # --------------------------------------------------

        viewer = tk.Frame(
            workspace,
            bg=WHITE,
            highlightbackground=BORDER,
            highlightthickness=1
        )
        viewer.pack(fill="x", pady=5)

        header = tk.Frame(viewer, bg=WHITE)
        header.pack(fill="x", padx=16, pady=(13, 5))

        tk.Label(
            header,
            text="EMAIL CONTENT",
            font=("Segoe UI", 12, "bold"),
            fg=TEXT,
            bg=WHITE
        ).pack(side="left")

        tk.Label(
            header,
            text="RAW DATASET MESSAGE",
            font=("Consolas", 8, "bold"),
            fg="#35627c",
            bg="#edf5fa",
            padx=7,
            pady=4
        ).pack(side="right")

        email_frame = tk.Frame(viewer, bg=WHITE)
        email_frame.pack(fill="x", padx=16, pady=(5, 15))

        email_scroll = ttk.Scrollbar(
            email_frame,
            orient="vertical"
        )

        self.emailbox = tk.Text(
            email_frame,
            height=13,
            bg="#07131f",
            fg="#eaf5ff",
            insertbackground="white",
            relief="flat",
            highlightbackground="#18374d",
            highlightthickness=1,
            font=("Consolas", 10),
            wrap="word",
            padx=14,
            pady=14,
            yscrollcommand=email_scroll.set
        )

        email_scroll.config(
            command=self.emailbox.yview
        )

        self.emailbox.pack(
            side="left",
            fill="both",
            expand=True
        )

        email_scroll.pack(
            side="right",
            fill="y"
        )

        # --------------------------------------------------
        # ANALYZE BUTTON
        # --------------------------------------------------

        action = tk.Frame(
            workspace,
            bg=BG
        )
        action.pack(fill="x", pady=8)

        tk.Button(
            action,
            text="[ANALYZE]  ANALYZE THREAT",
            command=self.analyze_email,
            bg=BLUE,
            fg="white",
            activebackground="#05658f",
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=24,
            pady=11
        ).pack(side="right")

        self.emailout = tk.Frame(
            workspace,
            bg=BG
        )
        self.emailout.pack(
            fill="both",
            expand=True,
            pady=(3, 25)
        )

        if self.email_row is not None:
            self.load_email_box()

    def pick_email(self, label):
        if self.email_df.empty:
            return

        d = (
            self.email_df
            if label is None
            else self.email_df[self.email_df["spam"] == label]
        )

        if len(d) == 0:
            return

        previous = (
            self.email_row.name
            if self.email_row is not None
            else None
        )

        if len(d) > 1 and previous in d.index:
            d = d.drop(index=previous)

        self.email_row = d.sample(
            n=1
        ).iloc[0]

        if hasattr(self, "emailbox"):
            self.emailbox.delete("1.0", "end")
            self.emailbox.insert(
                "1.0",
                str(self.email_row["text"])
            )

        self.clear_out()

    def load_email_box(self):
        self.emailbox.delete("1.0", "end")
        self.emailbox.insert(
            "1.0",
            str(self.email_row["text"])
        )

    def clear_out(self):
        if hasattr(self, "emailout"):
            for widget in self.emailout.winfo_children():
                widget.destroy()

    def result_card(
        self,
        parent,
        label,
        value,
        note,
        accent=BLUE
    ):
        frame = tk.Frame(
            parent,
            bg=WHITE,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=4
        )

        tk.Frame(
            frame,
            bg=accent,
            width=4
        ).pack(
            side="left",
            fill="y"
        )

        tk.Label(
            frame,
            text=label.upper(),
            font=("Segoe UI", 8, "bold"),
            fg=MUTED,
            bg=WHITE
        ).pack(
            anchor="w",
            padx=13,
            pady=(14, 4)
        )

        tk.Label(
            frame,
            text=value,
            font=("Segoe UI", 18, "bold"),
            fg=TEXT,
            bg=WHITE
        ).pack(
            anchor="w",
            padx=13
        )

        tk.Label(
            frame,
            text=note,
            font=("Segoe UI", 8),
            fg=MUTED,
            bg=WHITE
        ).pack(
            anchor="w",
            padx=13,
            pady=(4, 13)
        )

    def analyze_email(self):
        text = self.emailbox.get(
            "1.0",
            "end"
        ).strip()

        if not text:
            messagebox.showwarning(
                "ThreatLens X",
                "Select a random email first."
            )
            return

        self.clear_out()

        prediction = "UNKNOWN"
        confidence = 0.0

        try:
            X = self.email_vec.transform([text])
            decision = float(
                self.email_model.decision_function(X)[0]
            )

            prediction = (
                "SPAM"
                if decision >= 0
                else "LEGITIMATE"
            )

            confidence = (
                50 +
                min(abs(decision) * 25, 49)
            )

        except Exception as e:
            prediction = "ANALYSIS ERROR"

        ground_truth = "N/A"

        if self.email_row is not None:
            ground_truth = (
                "SPAM"
                if int(self.email_row["spam"]) == 1
                else "LEGITIMATE"
            )

        features = email_behavior(text)

        evidence = min(
            100,
            features["URLs"] * 8 +
            features["Urgency terms"] * 6 +
            features["Financial terms"] * 5 +
            features["Action terms"] * 5 +
            features["Exclamations"] * 3
        )

        risk = round(
            confidence * 0.50 +
            evidence * 0.20,
            1
        )

        severity = (
            "CRITICAL"
            if risk >= 85
            else "HIGH"
            if risk >= 70
            else "MEDIUM"
            if risk >= 40
            else "LOW"
        )

        # --------------------------------------------------
        # RESULT HEADER
        # --------------------------------------------------

        heading = tk.Frame(
            self.emailout,
            bg=BG
        )
        heading.pack(
            fill="x",
            pady=(5, 8)
        )

        tk.Label(
            heading,
            text="INVESTIGATION RESULT",
            font=("Segoe UI", 13, "bold"),
            fg=TEXT,
            bg=BG
        ).pack(side="left")

        tk.Label(
            heading,
            text=f"  {severity}",
            font=("Consolas", 9, "bold"),
            fg=RED if severity in ("HIGH", "CRITICAL") else ORANGE,
            bg=RED2 if severity in ("HIGH", "CRITICAL") else ORANGE2,
            padx=8,
            pady=4
        ).pack(side="left", padx=8)

        # --------------------------------------------------
        # METRIC CARDS
        # --------------------------------------------------

        cards = tk.Frame(
            self.emailout,
            bg=BG
        )
        cards.pack(
            fill="x",
            pady=(0, 12)
        )

        self.result_card(
            cards,
            "AI Verdict",
            prediction,
            "Trained email classifier",
            RED if prediction == "SPAM" else GREEN
        )

        self.result_card(
            cards,
            "Ground Truth",
            ground_truth,
            "emails.csv label",
            RED if ground_truth == "SPAM" else GREEN
        )

        self.result_card(
            cards,
            "Confidence",
            f"{confidence:.1f}%",
            "Model decision confidence",
            BLUE
        )

        self.result_card(
            cards,
            "Risk Score",
            f"{risk}/100",
            severity,
            RED if severity in ("HIGH", "CRITICAL") else ORANGE
        )

        # --------------------------------------------------
        # TWO-COLUMN EVIDENCE AREA
        # --------------------------------------------------

        evidence_row = tk.Frame(
            self.emailout,
            bg=BG
        )
        evidence_row.pack(
            fill="both",
            expand=True
        )

        explanation = self.panel(
            evidence_row,
            "AI ANALYST EXPLANATION",
            "WHY THIS EMAIL WAS FLAGGED"
        )

        explanation_text = (
            f"Classification: {prediction}\n\n"
            f"Model confidence: {confidence:.1f}%\n\n"
            f"Behavioral evidence score: {evidence:.1f}/100\n\n"
            f"Observed indicators:\n"
            f" URLs: {features['URLs']}\n"
            f" Urgency terms: {features['Urgency terms']}\n"
            f" Financial terms: {features['Financial terms']}\n"
            f" Action terms: {features['Action terms']}\n"
            f" Uppercase ratio: {features['Uppercase %']:.1f}%\n"
            f" Exclamation marks: {features['Exclamations']}\n\n"
            f"These indicators contribute to the investigation "
            f"risk calculation."
        )

        tk.Label(
            explanation,
            text=explanation_text,
            justify="left",
            anchor="nw",
            font=("Segoe UI", 10),
            fg=MUTED,
            bg=WHITE
        ).pack(
            fill="both",
            expand=True,
            padx=17,
            pady=(7, 18)
        )

        dna = self.panel(
            evidence_row,
            "THREAT DNA PROFILE",
            "OBSERVED EMAIL SIGNATURE"
        )

        dna_text = "\n".join(
            f"{i+1:02d}   {k:<22} {v:.1f}"
            if isinstance(v, float)
            else f"{i+1:02d}   {k:<22} {v}"
            for i, (k, v)
            in enumerate(features.items())
        )

        tk.Label(
            dna,
            text=dna_text,
            justify="left",
            anchor="nw",
            font=("Consolas", 10),
            fg=TEXT,
            bg=WHITE
        ).pack(
            fill="both",
            expand=True,
            padx=17,
            pady=(7, 18)
        )

        # --------------------------------------------------
        # INVESTIGATION SUMMARY
        # --------------------------------------------------

        summary = tk.Frame(
            self.emailout,
            bg=WHITE,
            highlightbackground=BORDER,
            highlightthickness=1
        )
        summary.pack(
            fill="x",
            pady=(12, 10)
        )

        tk.Label(
            summary,
            text="INVESTIGATION SUMMARY",
            font=("Segoe UI", 11, "bold"),
            fg=TEXT,
            bg=WHITE
        ).pack(
            anchor="w",
            padx=17,
            pady=(13, 5)
        )

        summary_text = (
            f"AI verdict: {prediction}    |    "
            f"Dataset ground truth: {ground_truth}    |    "
            f"Risk: {risk}/100    |    "
            f"Severity: {severity}"
        )

        tk.Label(
            summary,
            text=summary_text,
            font=("Consolas", 9),
            fg=MUTED,
            bg=WHITE
        ).pack(
            anchor="w",
            padx=17,
            pady=(0, 14)
        )

    def show_dna(self):
        self.clear();self.page_header("Threat DNA Analyzer","Random malware dataset sample + model-grounded feature signature","DATASET SAMPLE")
        bar=tk.Frame(self.main,bg=WHITE,highlightbackground=BORDER,highlightthickness=1);bar.pack(fill="x",padx=18,pady=5)
        tk.Button(bar,text="[RANDOM] RANDOM MALWARE SAMPLE",command=self.pick_mal,bg=BLUE,fg="white",relief="flat",font=("Segoe UI",9,"bold"),padx=12,pady=8).pack(side="left",padx=10,pady=10)
        tk.Button(bar,text="ANALYZE SAMPLE",command=self.analyze_mal,bg="#0d6f96",fg="white",relief="flat",font=("Segoe UI",9,"bold"),padx=12,pady=8).pack(side="left",padx=3)
        self.maltree=ttk.Treeview(self.main,columns=("feature","value"),show="headings");self.maltree.heading("feature",text="FEATURE");self.maltree.heading("value",text="VALUE");self.maltree.column("feature",width=380);self.maltree.column("value",width=700)
        self.maltree.pack(fill="both",expand=True,padx=18,pady=8)
        if self.mal_row is not None:self.load_mal()

    def pick_mal(self):
        if self.mal_df.empty:
            return

        mode=self.mal_mode.get() if hasattr(self,"mal_mode") else "Random"
        previous=self.mal_row.name if self.mal_row is not None else None

        if mode=="Malicious Only":
            d=self.mal_df[self.mal_df["legitimate"]==0]
        elif mode=="Legitimate Only":
            d=self.mal_df[self.mal_df["legitimate"]==1]
        else:
            d=self.mal_df

        if len(d)==0:
            messagebox.showwarning(
                "ThreatLens X",
                "No samples are available for the selected mode."
            )
            return

        if len(d)>1 and previous in d.index:
            d=d.drop(index=previous)

        self.mal_row=d.sample(1).iloc[0]

        if hasattr(self,"mal_index"):
            self.mal_index.set(str(self.mal_row.name))

        self.load_mal()
    def load_selected_malware(self):
        if self.mal_df.empty:
            messagebox.showerror("ThreatLens X","Malware dataset unavailable.")
            return

        idx=self.mal_index.get().strip()

        if idx:
            try:
                row_number=int(idx)
                if row_number < 0 or row_number >= len(self.mal_df):
                    messagebox.showwarning(
                        "Invalid Dataset Row",
                        f"Enter a row from 0 to {len(self.mal_df)-1:,}."
                    )
                    return
                self.mal_row=self.mal_df.iloc[row_number]
                self.load_mal()
                return
            except ValueError:
                messagebox.showwarning(
                    "Invalid Dataset Row",
                    "Dataset row must be a numeric value."
                )
                return

        self.pick_mal()

    def load_mal(self):
        for x in self.maltree.get_children():self.maltree.delete(x)
        for c,v in self.mal_row.items():self.maltree.insert("","end",values=(c,str(v)))
    def analyze_mal(self):
        if self.mal_row is None:
            self.pick_mal()
        if self.mal_row is None or self.mal_model is None:
            messagebox.showerror("ThreatLens X","Malware model or dataset unavailable.")
            return

        try:
            cols=list(self.mal_features)
            vals=pd.to_numeric(
                pd.Series({c:self.mal_row.get(c,0) for c in cols}),
                errors="coerce"
            ).fillna(0).to_numpy().reshape(1,-1)

            pr=self.mal_model.predict_proba(vals)[0]
            classes=list(self.mal_model.classes_)
            mp=float(pr[classes.index(1)]) if 1 in classes else float(pr[-1])

            gt=int(1-int(self.mal_row["legitimate"])) if "legitimate" in self.mal_row else None
            verdict="MALICIOUS" if mp>=0.5 else "LEGITIMATE"
            ground_truth="MALICIOUS" if gt==1 else "LEGITIMATE"
            risk=mp*100
            severity="CRITICAL" if risk>=85 else "HIGH" if risk>=70 else "MEDIUM" if risk>=40 else "LOW"

            if hasattr(self,"malout"):
                for w in self.malout.winfo_children():
                    w.destroy()
            else:
                self.malout=tk.Frame(self.main,bg=BG)
                self.malout.pack(fill="both",expand=True,padx=18,pady=8)

            panel=tk.Frame(
                self.malout,
                bg=WHITE,
                highlightbackground=BORDER,
                highlightthickness=1
            )
            panel.pack(fill="x")

            tk.Label(
                panel,
                text="MALWARE INVESTIGATION RESULT",
                font=("Segoe UI",13,"bold"),
                fg=TEXT,
                bg=WHITE
            ).pack(anchor="w",padx=16,pady=(14,8))

            summary=tk.Frame(panel,bg=WHITE)
            summary.pack(fill="x",padx=16,pady=(0,12))

            for label,value,fg in [
                ("AI VERDICT",verdict,RED if verdict=="MALICIOUS" else GREEN),
                ("MALICIOUS PROBABILITY",f"{mp*100:.2f}%",RED if mp>=.5 else GREEN),
                ("DATASET GROUND TRUTH",ground_truth,RED if gt==1 else GREEN),
                ("RISK SCORE",f"{risk:.1f}/100",ORANGE if risk>=40 else GREEN),
                ("SEVERITY",severity,RED if severity in ("CRITICAL","HIGH") else ORANGE if severity=="MEDIUM" else GREEN)
            ]:
                card=tk.Frame(summary,bg="#f7f9fc",highlightbackground=BORDER,highlightthickness=1)
                card.pack(side="left",fill="x",expand=True,padx=3)
                tk.Label(card,text=label,font=("Segoe UI",8,"bold"),fg=MUTED,bg="#f7f9fc").pack(pady=(9,2))
                tk.Label(card,text=value,font=("Segoe UI",11,"bold"),fg=fg,bg="#f7f9fc").pack(pady=(0,9))

            evidence=tk.Frame(panel,bg=WHITE)
            evidence.pack(fill="x",padx=16,pady=(0,14))

            tk.Label(
                evidence,
                text="INVESTIGATION EVIDENCE",
                font=("Segoe UI",10,"bold"),
                fg=TEXT,
                bg=WHITE
            ).pack(anchor="w",pady=(0,5))

            tk.Label(
                evidence,
                text=(
                    "Prediction is generated from the trained Random Forest model using "
                    "the malware dataset feature vector. Dataset ground truth comes from "
                    "the 'legitimate' label in malware.csv. "
                    "No external malware sample is used."
                ),
                justify="left",
                wraplength=950,
                font=("Segoe UI",9),
                fg=MUTED,
                bg=WHITE
            ).pack(anchor="w")

        except Exception as e:
            messagebox.showerror("Malware Analysis Error",str(e))

    def show_investigation(self):
        self.clear()

        self.page_header(
            "AI Deep Investigation",
            "Live model-grounded investigation of the selected dataset sample",
            "AI INVESTIGATION"
        )

        if self.mal_row is None:
            self.pick_mal()

        if self.mal_row is None:
            tk.Label(
                self.main,
                text="No malware dataset sample is available.",
                font=("Segoe UI",12,"bold"),
                fg=RED,
                bg=BG
            ).pack(padx=25,pady=30)
            return

        # Build model prediction
        try:
            cols=list(self.mal_features)
            vals=pd.to_numeric(
                pd.Series({
                    c:self.mal_row.get(c,0)
                    for c in cols
                }),
                errors="coerce"
            ).fillna(0).to_numpy().reshape(1,-1)

            pr=self.mal_model.predict_proba(vals)[0]
            classes=list(self.mal_model.classes_)
            malicious_probability=(
                float(pr[classes.index(1)])
                if 1 in classes
                else float(pr[-1])
            )

            verdict=(
                "MALICIOUS"
                if malicious_probability >= 0.5
                else "LEGITIMATE"
            )

            gt=(
                int(1-int(self.mal_row["legitimate"]))
                if "legitimate" in self.mal_row
                else None
            )

            ground_truth=(
                "MALICIOUS"
                if gt==1
                else "LEGITIMATE"
                if gt==0
                else "UNKNOWN"
            )

            risk=malicious_probability*100

            severity=(
                "CRITICAL" if risk>=85 else
                "HIGH" if risk>=70 else
                "MEDIUM" if risk>=40 else
                "LOW"
            )

        except Exception as e:
            messagebox.showerror(
                "Investigation Error",
                str(e)
            )
            return

        # Top status cards
        cards=tk.Frame(self.main,bg=BG)
        cards.pack(fill="x",padx=18,pady=4)

        self.metric(
            cards,
            "AI Verdict",
            verdict,
            "Random Forest classification",
            RED if verdict=="MALICIOUS" else GREEN
        )

        self.metric(
            cards,
            "Malicious Probability",
            f"{malicious_probability*100:.1f}%",
            "Model probability",
            RED if malicious_probability>=0.5 else GREEN
        )

        self.metric(
            cards,
            "Ground Truth",
            ground_truth,
            "malware.csv label",
            RED if gt==1 else GREEN
        )

        self.metric(
            cards,
            "Risk Score",
            f"{risk:.1f}/100",
            "Investigation score",
            ORANGE
        )

        # Scrollable report
        outer=tk.Frame(self.main,bg=BG)
        outer.pack(fill="both",expand=True,padx=18,pady=8)

        canvas=tk.Canvas(
            outer,
            bg=BG,
            highlightthickness=0
        )
        scroll=ttk.Scrollbar(
            outer,
            orient="vertical",
            command=canvas.yview
        )

        report=tk.Frame(canvas,bg=BG)

        window=canvas.create_window(
            (0,0),
            window=report,
            anchor="nw"
        )

        report.bind(
            "<Configure>",
            lambda e:canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.bind(
            "<Configure>",
            lambda e:canvas.itemconfigure(
                window,
                width=e.width
            )
        )

        canvas.configure(
            yscrollcommand=scroll.set
        )

        canvas.pack(
            side="left",
            fill="both",
            expand=True
        )
        scroll.pack(
            side="right",
            fill="y"
        )

        # Sample identity
        identity=tk.Frame(
            report,
            bg=WHITE,
            highlightbackground=BORDER,
            highlightthickness=1
        )
        identity.pack(fill="x",pady=(0,8))

        tk.Label(
            identity,
            text="SAMPLE IDENTITY",
            font=("Segoe UI",12,"bold"),
            fg=TEXT,
            bg=WHITE
        ).pack(anchor="w",padx=16,pady=(14,8))

        sample_name=str(
            self.mal_row.get("Name","Unknown")
        )
        sample_md5=str(
            self.mal_row.get("md5","Not available")
        )

        identity_text=(
            f"Dataset row: {self.mal_row.name}\n"
            f"Name: {sample_name}\n"
            f"MD5: {sample_md5}\n"
            f"Source: data/malware/malware.csv"
        )

        tk.Label(
            identity,
            text=identity_text,
            justify="left",
            font=("Consolas",9),
            fg=MUTED,
            bg=WHITE
        ).pack(anchor="w",padx=16,pady=(0,14))

        # Detection analysis
        detection=tk.Frame(
            report,
            bg=WHITE,
            highlightbackground=BORDER,
            highlightthickness=1
        )
        detection.pack(fill="x",pady=8)

        tk.Label(
            detection,
            text="DETECTION ANALYSIS",
            font=("Segoe UI",12,"bold"),
            fg=TEXT,
            bg=WHITE
        ).pack(anchor="w",padx=16,pady=(14,8))

        detection_text=(
            f"Model verdict: {verdict}\n"
            f"Malicious probability: {malicious_probability*100:.2f}%\n"
            f"Dataset ground truth: {ground_truth}\n"
            f"Risk score: {risk:.1f}/100\n"
            f"Severity: {severity}\n"
            f"Model: Random Forest"
        )

        tk.Label(
            detection,
            text=detection_text,
            justify="left",
            font=("Consolas",9),
            fg=MUTED,
            bg=WHITE
        ).pack(anchor="w",padx=16,pady=(0,14))

        # Feature evidence
        evidence=tk.Frame(
            report,
            bg=WHITE,
            highlightbackground=BORDER,
            highlightthickness=1
        )
        evidence.pack(fill="x",pady=8)

        tk.Label(
            evidence,
            text="FEATURE EVIDENCE",
            font=("Segoe UI",12,"bold"),
            fg=TEXT,
            bg=WHITE
        ).pack(anchor="w",padx=16,pady=(14,8))

        feature_box=tk.Frame(evidence,bg="#f7f9fc")
        feature_box.pack(fill="x",padx=16,pady=(0,15))

        shown=0
        for feature in cols:
            value=self.mal_row.get(feature,0)

            if str(feature).lower() in {
                "name","md5","legitimate"
            }:
                continue

            try:
                numeric=float(value)
            except:
                continue

            row=tk.Frame(feature_box,bg="#f7f9fc")
            row.pack(fill="x",padx=10,pady=2)

            tk.Label(
                row,
                text=str(feature),
                font=("Consolas",8),
                fg=TEXT,
                bg="#f7f9fc",
                width=32,
                anchor="w"
            ).pack(side="left")

            tk.Label(
                row,
                text=f"{numeric:.4g}",
                font=("Consolas",8,"bold"),
                fg="#315a78",
                bg="#f7f9fc",
                anchor="e"
            ).pack(side="right")

            shown+=1

            if shown>=18:
                break

        # Investigation conclusion
        conclusion=tk.Frame(
            report,
            bg=WHITE,
            highlightbackground=BORDER,
            highlightthickness=1
        )
        conclusion.pack(fill="x",pady=8)

        tk.Label(
            conclusion,
            text="INVESTIGATION CONCLUSION",
            font=("Segoe UI",12,"bold"),
            fg=TEXT,
            bg=WHITE
        ).pack(anchor="w",padx=16,pady=(14,8))

        conclusion_text=(
            f"The trained Random Forest classified this dataset sample as "
            f"{verdict} with a malicious probability of "
            f"{malicious_probability*100:.2f}%.\n\n"
            f"The malware.csv ground-truth label for this sample is "
            f"{ground_truth}.\n\n"
            "This investigation is grounded in the supplied malware dataset "
            "and trained model. It does not claim that this sample was "
            "delivered by a particular email."
        )

        tk.Label(
            conclusion,
            text=conclusion_text,
            justify="left",
            wraplength=1000,
            font=("Segoe UI",9),
            fg=MUTED,
            bg=WHITE
        ).pack(anchor="w",padx=16,pady=(0,16))

        # Actions
        actions=tk.Frame(report,bg=BG)
        actions.pack(fill="x",pady=(4,18))

        tk.Button(
            actions,
            text="REANALYZE SAMPLE",
            command=self.analyze_mal,
            bg=BLUE,
            fg="white",
            relief="flat",
            font=("Segoe UI",9,"bold"),
            padx=16,
            pady=9
        ).pack(side="right")

        tk.Button(
            actions,
            text="NEW RANDOM SAMPLE",
            command=lambda:self.show_dna(),
            bg="#edf5fa",
            fg="#1e5f7e",
            relief="flat",
            font=("Segoe UI",9,"bold"),
            padx=16,
            pady=9
        ).pack(side="right",padx=6)

    def show_playbooks(self):
        self.clear();self.page_header("Rule Engine & Playbooks","Safe demo controls for investigation workflow","SOC WORKBENCH")
        for title,desc in [
            ("Quarantine Investigation","Mark the current dataset sample for analyst review."),
            ("Export Investigation","Prepare the current verdict, evidence and ground truth for reporting."),
            ("Refresh Dataset","Reload the existing email and malware datasets."),
            ("Model Status","Inspect whether trained model artifacts are available.")]:
            f=tk.Frame(self.main,bg=WHITE,highlightbackground=BORDER,highlightthickness=1);f.pack(fill="x",padx=23,pady=6)
            tk.Label(f,text=title,font=("Segoe UI",11,"bold"),fg=TEXT,bg=WHITE).pack(anchor="w",padx=16,pady=(13,3))
            tk.Label(f,text=desc,font=("Segoe UI",9),fg=MUTED,bg=WHITE).pack(anchor="w",padx=16,pady=(0,10))
            tk.Button(f,text="OPEN",command=lambda:messagebox.showinfo("ThreatLens X","Demo workflow ready for analyst action."),bg="#edf5fa",fg="#1e5f7e",relief="flat",font=("Segoe UI",8,"bold"),padx=12,pady=6).pack(anchor="e",padx=16,pady=(0,12))

if __name__=="__main__":
    ThreatLensDesktop().mainloop()







