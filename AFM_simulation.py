import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import grey_dilation, gaussian_filter
from scipy.stats import skew, kurtosis
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from dataclasses import dataclass
import threading
import queue
import random

# Set matplotlib style
plt.style.use('default')
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.facecolor'] = '#2b2b2b'
plt.rcParams['figure.facecolor'] = '#1a1a1a'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['grid.color'] = '#444444'

# ------------------------------------------------------
# VALIDATED HYBRID PHYSICS ENGINE
# ------------------------------------------------------
def generate_analytical_fd_curve(
    z_range,
    tip_radius=10e-9,
    epsilon_scaler=1.0,
    E_reduced=100e9,        
    hamaker=1e-19,      
):
    """
    Returns: z_nm, approach_nN, retract_nN
    Model: Hybrid vdW (Sphere-Plane) + Hertzian Contact
    - z > sigma:  vdW attraction F = -AR / 6z^2
    - z <= sigma: Hertz repulsion F = 4/3 E* sqrt(R) d^1.5
    """
    z = np.linspace(z_range[0], z_range[1], 1200)  # meters
    R = max(float(tip_radius), 1e-12)
    
    # Physics Constants
    A_eff = float(hamaker) * float(epsilon_scaler)
    sigma = 0.34e-9  # Interatomic distance (hard wall start)
    
    # Initialize Force Array
    F = np.zeros_like(z)
    
    # 1. NON-CONTACT REGIME (z > sigma)
    # Sphere-Plane van der Waals
    mask_nc = z > sigma
    # Clamp z to avoid singularity if sigma is crossed slightly in mask
    z_nc = np.maximum(z[mask_nc], sigma) 
    F[mask_nc] = - (A_eff * R) / (6.0 * z_nc**2)
    
    # 2. CONTACT REGIME (z <= sigma)
    # Hertzian Contact + Adhesion offset (continuity)
    mask_c = z <= sigma
    delta = sigma - z[mask_c] # Indentation depth
    
    # Force at the handover point (z=sigma) to ensure continuity
    F_adhesion = - (A_eff * R) / (6.0 * sigma**2)
    
    # Hertzian Repulsion: F = 4/3 * E* * sqrt(R) * delta^1.5
    F_hertz = (4.0 / 3.0) * float(E_reduced) * np.sqrt(R) * (delta ** 1.5)
    
    # Total contact force = Repulsion + Constant Adhesion Baseline
    F[mask_c] = F_hertz + F_adhesion

    # Convert to nN and z to nm
    approach_nN = F * 1e9
    retract_nN  = F * 1e9 # Elastic model (no hysteresis added for clarity)
    z_nm = z * 1e9

    return z_nm, approach_nN, retract_nN


# ------------------------------------------------------
# DATA CLASSES
# ------------------------------------------------------
@dataclass
class CantileverParams:
    spring_constant: float = 1.0 
    resonance_freq: float = 75e3 
    quality_factor: float = 200 
    tip_radius: float = 10e-9 
    mass: float = 1e-11 
    damping: float = 0.0 
    def __post_init__(self):
        omega_0 = 2 * np.pi * self.resonance_freq
        self.damping = self.mass * omega_0 / self.quality_factor

@dataclass
class FeedbackParams:
    setpoint: float = 0.8 
    kp: float = 0.1 
    ki: float = 0.01 
    kd: float = 0.001 
    integral_limit: float = 10.0 
     
@dataclass
class EnvironmentParams:
    temperature: float = 295 
    humidity: float = 40 
    vibration_amplitude: float = 1e-10 
    acoustic_noise: float = 60 

# ------------------------------------------------------
# MATERIAL DATABASE
# ------------------------------------------------------
MAT_COLORS = {
    'Gold':   [1.0, 0.84, 0.0], 
    'Silicon': [0.4, 0.4, 0.45], 
    'SiO2':   [0.6, 0.8, 0.9],   
    'Cobalt': [0.2, 0.2, 0.8],   
    'PMMA':   [0.9, 0.3, 0.4],   
    'Graphene':[0.1, 0.1, 0.1]    
}

MATERIAL_DB = {
    'Gold': {'conductivity': 4.1e7, 'work_function': 5.1, 'thermal_conductivity': 317, 
             'elastic_modulus': 79e9, 'poisson': 0.44, 'hamaker': 4e-19},
    'Silicon': {'conductivity': 1.56e-3, 'work_function': 4.85, 'thermal_conductivity': 148, 
                'elastic_modulus': 150e9, 'poisson': 0.28, 'hamaker': 1.865e-19},
    'SiO2': {'conductivity': 1e-14, 'work_function': 5.0, 'thermal_conductivity': 1.4, 
             'elastic_modulus': 70e9, 'poisson': 0.17, 'hamaker': 0.65e-19},
    'Cobalt': {'conductivity': 1.6e7, 'work_function': 5.0, 'thermal_conductivity': 100, 
               'elastic_modulus': 209e9, 'poisson': 0.31, 'hamaker': 2.5e-19},
    'PMMA': {'conductivity': 1e-15, 'work_function': 4.0, 'thermal_conductivity': 0.2, 
             'elastic_modulus': 3e9, 'poisson': 0.35, 'hamaker': 0.7e-19},
    'Graphene': {'conductivity': 1e8, 'work_function': 4.5, 'thermal_conductivity': 3000, 
                 'elastic_modulus': 1000e9, 'poisson': 0.16, 'hamaker': 4e-19}
}

def generate_surface(size, roughness, surface_type='random', **kwargs):
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y) 
    
    surface = np.zeros((size, size))
    bg_mat = 'SiO2'
    material_map = np.full((size, size), bg_mat, dtype=object) 
    
    prop_maps = {
        'conductivity': np.full((size, size), MATERIAL_DB[bg_mat]['conductivity']),
        'work_function': np.full((size, size), MATERIAL_DB[bg_mat]['work_function']),
        'thermal_conductivity': np.full((size, size), MATERIAL_DB[bg_mat]['thermal_conductivity']),
        'elastic_modulus': np.full((size, size), MATERIAL_DB[bg_mat]['elastic_modulus']),
        'poisson': np.full((size, size), MATERIAL_DB[bg_mat]['poisson']),
        'hamaker': np.full((size, size), MATERIAL_DB[bg_mat]['hamaker']),
    }

    def apply_material(mask, mat_name):
        if mat_name not in MATERIAL_DB: return
        props = MATERIAL_DB[mat_name]
        noise = 1.0 + 0.05 * np.random.randn(size, size) 
        prop_maps['conductivity'][mask] = props['conductivity'] * noise[mask]
        prop_maps['work_function'][mask] = props['work_function'] * noise[mask]
        prop_maps['thermal_conductivity'][mask] = props['thermal_conductivity'] * noise[mask]
        prop_maps['elastic_modulus'][mask] = props['elastic_modulus'] * noise[mask]
        prop_maps['poisson'][mask] = props['poisson']
        prop_maps['hamaker'][mask] = props['hamaker']
        material_map[mask] = mat_name 

    mat_list = list(MATERIAL_DB.keys())
    
    if surface_type == 'spheres':
        n_spheres = 20
        for _ in range(n_spheres):
            cx, cy = np.random.randint(size//10, 9*size//10, 2)
            r = np.random.randint(size//25, size//8)
            Yg, Xg = np.ogrid[:size, :size]
            dist_sq = (Xg - cx)**2 + (Yg - cy)**2
            mask = dist_sq <= r**2
            if np.any(mask):
                h = roughness * np.random.uniform(0.8, 1.2)
                sph = np.sqrt(np.maximum(0, r**2 - dist_sq[mask])) / r * h
                surface[mask] = np.maximum(surface[mask], sph)
                apply_material(mask, random.choice(mat_list))
    
    # OPTIMIZED VECTORIZED PYRAMIDS
    elif surface_type == 'pyramids':
        n_pyramids = 12
        cx = np.random.randint(size//8, 7*size//8, n_pyramids)
        cy = np.random.randint(size//8, 7*size//8, n_pyramids)
        base_sizes = np.random.randint(size//15, size//8, n_pyramids)
        heights = roughness * np.random.uniform(0.8, 1.2, n_pyramids)
        
        grid_y, grid_x = np.indices((size, size))
        
        for i in range(n_pyramids):
            # Manhattan distance for pyramid shape: max(|x-cx|, |y-cy|)
            dist = np.maximum(np.abs(grid_x - cx[i]), np.abs(grid_y - cy[i]))
            mask = dist < base_sizes[i]
            if np.any(mask):
                pyr = heights[i] * (1 - dist[mask] / base_sizes[i])
                surface[mask] = np.maximum(surface[mask], pyr)
                apply_material(mask, random.choice(mat_list))

    elif surface_type == 'sine':
        freq = 4
        surface = roughness * (0.5 + 0.5 * np.sin(2 * np.pi * freq * X))
        mask_peaks = surface > (0.5 * roughness)
        apply_material(mask_peaks, 'Cobalt')
        apply_material(~mask_peaks, 'PMMA')
    elif surface_type == 'step':
        surface[size//2:, :] = roughness
        mask_top = np.zeros((size, size), dtype=bool)
        mask_top[size//2:, :] = True
        apply_material(mask_top, 'Gold')
        apply_material(~mask_top, 'Silicon')
    elif surface_type == 'grating':
        period = size // 8
        for i in range(size):
            mask_line = np.zeros((size, size), dtype=bool)
            if (i % period) < (period * 0.5):
                surface[i, :] = roughness
                mask_line[i, :] = True
                apply_material(mask_line, 'Graphene')
            else:
                surface[i, :] = 0
                mask_line[i, :] = True
                apply_material(mask_line, 'SiO2')
    elif surface_type == 'random':
        surface = gaussian_filter(np.random.randn(size, size), sigma=size/40)
        surface = roughness * (surface - surface.min()) / (surface.max() - surface.min())
        noise_map = gaussian_filter(np.random.randn(size, size), sigma=size/10)
        mask_1 = noise_map > 0.4
        mask_2 = (noise_map <= 0.4) & (noise_map > -0.4)
        mask_3 = noise_map <= -0.4
        apply_material(mask_1, random.choice(mat_list))
        apply_material(mask_2, random.choice(mat_list))
        apply_material(mask_3, random.choice(mat_list))

    for key in prop_maps:
        prop_maps[key] = gaussian_filter(prop_maps[key], sigma=1.0)

    props = prop_maps
    props['cpd'] = props['work_function'] - 4.5
    props['material_map'] = material_map 
    
    return surface, props

def create_tip_kernel(radius_px, tip_shape='parabolic', aspect_ratio=1.0):
    r = int(np.ceil(radius_px))
    L = 2 * r + 1
    X, Y = np.meshgrid(np.arange(L) - r, np.arange(L) - r)
    Y = Y / aspect_ratio
    R = np.sqrt(X**2 + Y**2)
    mask = (R <= radius_px)
    kernel = np.full((L, L), np.nan, dtype=float)
    
    if tip_shape == 'parabolic':
        kernel[mask] = (R[mask] ** 2) / (2.0 * max(radius_px, 0.1))
    elif tip_shape == 'conical':
        slope = 0.5
        kernel[mask] = slope * R[mask]
    elif tip_shape == 'spherical':
        rt2 = max(radius_px, 0.1) ** 2
        kernel[mask] = max(radius_px, 0.1) - np.sqrt(np.maximum(rt2 - R[mask] ** 2, 0.0))
    else: 
        kernel[mask] = 0.0
        
    kmin = np.nanmin(kernel)
    kernel = kernel - kmin
    return kernel

# ------------------------------------------------------
# GUI APPLICATION
# ------------------------------------------------------

class AFMSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('🔬 AFM Simulator - Validated Physics Edition')
        self.root.geometry('780x600')
        self.root.configure(bg='#1a1a1a')
        self.root.minsize(780, 600)
        
        self.cantilever_params = CantileverParams()
        self.feedback_params = FeedbackParams()
        self.env_params = EnvironmentParams()
        self.colorbars = {}

        self.setup_styles()
        
        self.paned = ttk.PanedWindow(root, orient='horizontal')
        self.paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        control_container = ttk.Frame(self.paned, width=260)
        self.paned.add(control_container, weight=0)
        
        self.control_notebook = ttk.Notebook(control_container)
        self.control_notebook.pack(fill='both', expand=True, padx=2, pady=2)
        
        self.tab_surface = ttk.Frame(self.control_notebook, padding=8)
        self.tab_cantilever = ttk.Frame(self.control_notebook, padding=8)
        self.tab_environment = ttk.Frame(self.control_notebook, padding=8)
        self.tab_multimodal = ttk.Frame(self.control_notebook, padding=8)
        
        self.control_notebook.add(self.tab_surface, text=' Surface ')
        self.control_notebook.add(self.tab_cantilever, text=' Cantilever ')
        self.control_notebook.add(self.tab_environment, text=' Environment ')
        self.control_notebook.add(self.tab_multimodal, text=' Multimodal ')
        
        plot_container = ttk.Frame(self.paned)
        self.paned.add(plot_container, weight=1)
        
        self.plot_notebook = ttk.Notebook(plot_container)
        self.plot_notebook.pack(fill='both', expand=True, padx=2, pady=2)
        
        self.plot_tab_main = ttk.Frame(self.plot_notebook)
        self.plot_tab_dynamics = ttk.Frame(self.plot_notebook)
        self.plot_tab_analysis = ttk.Frame(self.plot_notebook)
        self.plot_tab_3d = ttk.Frame(self.plot_notebook)
        
        self.plot_notebook.add(self.plot_tab_main, text=' 📊 Images ')
        self.plot_notebook.add(self.plot_tab_dynamics, text=' 📈 Dynamics ')
        self.plot_notebook.add(self.plot_tab_analysis, text=' 🔬 Analysis ')
        self.plot_notebook.add(self.plot_tab_3d, text=' 🧊 3D View ')
        
        self.init_variables()
        self.build_surface_tab()
        self.build_cantilever_tab()
        self.build_environment_tab()
        self.build_multimodal_tab()
        self.setup_plot_figures()
        
        self.surface = None
        self.props = None
        self.simulation_queue = queue.Queue()
        self.simulation_thread = None
        
        self.simulate()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        bg_dark = '#1a1a1a'
        bg_medium = '#2b2b2b'
        fg_color = '#ffffff'
        accent = '#00bfa5'
        
        style.configure('TNotebook', background=bg_dark, borderwidth=0)
        style.configure('TNotebook.Tab', background=bg_medium, foreground=fg_color, padding=[15, 8], font=('Segoe UI', 9, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', accent)], foreground=[('selected', '#000000')])
        style.configure('TFrame', background=bg_dark)
        style.configure('TLabel', background=bg_dark, foreground=fg_color, font=('Segoe UI', 9))
        style.configure('TLabelframe', background=bg_dark, foreground=fg_color, borderwidth=2, relief='groove')
        style.configure('TLabelframe.Label', background=bg_dark, foreground=accent, font=('Segoe UI', 10, 'bold'))
        style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=10)
        style.configure('TScale', background=bg_dark, troughcolor=bg_medium)
        style.configure('TCombobox', font=('Segoe UI', 9))
        style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'), foreground=accent, background=bg_dark)
        style.configure('Info.TLabel', font=('Segoe UI', 9), foreground='#80cbc4', background=bg_dark)
        style.configure('Result.TLabel', font=('Consolas', 10), foreground='#80ff80', background=bg_medium)
        
    def init_variables(self):
        self.grid_var = tk.IntVar(value=128)
        self.roughness_var = tk.DoubleVar(value=5.0)
        self.surface_type_var = tk.StringVar(value='spheres')
        self.scan_size_var = tk.DoubleVar(value=500)
        
        self.spring_const_var = tk.DoubleVar(value=40.0) 
        self.resonance_freq_var = tk.DoubleVar(value=300.0)
        self.q_factor_var = tk.DoubleVar(value=400)
        self.tip_radius_var = tk.DoubleVar(value=10)
        self.tip_modulus_var = tk.DoubleVar(value=170.0) 
        self.tip_poisson_var = tk.DoubleVar(value=0.27)  

        self.setpoint_var = tk.DoubleVar(value=0.8)
        self.scan_speed_var = tk.DoubleVar(value=1.0)
        self.temperature_var = tk.DoubleVar(value=295)
        self.humidity_var = tk.DoubleVar(value=40)
        self.vibration_var = tk.DoubleVar(value=0.1)
        self.acoustic_var = tk.DoubleVar(value=60)
        
        self.mode_var = tk.StringVar(value='tapping')
        self.multimodal_var = tk.StringVar(value='None')
        self.bias_var = tk.DoubleVar(value=1.0)
        
        self.tip_shape_var = tk.StringVar(value='parabolic')
        self.tip_aspect_var = tk.DoubleVar(value=1.0)
        
        self.fd_model_var = tk.StringVar(value='hybrid-vdw-hertz')
        self.z_min_var = tk.DoubleVar(value=-5)
        self.z_max_var = tk.DoubleVar(value=10)
        self.epsilon_scaler_var = tk.DoubleVar(value=1.0)

    def update_mechanical_presets(self, event=None):
        mode = self.mode_var.get()
        if mode == 'contact':
            self.spring_const_var.set(0.2)
            self.resonance_freq_var.set(15.0)
            self.q_factor_var.set(50)
            self.setpoint_var.set(2.0) 
        elif mode == 'tapping':
            self.spring_const_var.set(40.0)
            self.resonance_freq_var.set(300.0)
            self.q_factor_var.set(400)
            self.setpoint_var.set(0.8) 
        elif mode == 'non-contact':
            self.spring_const_var.set(45.0)
            self.resonance_freq_var.set(330.0)
            self.q_factor_var.set(500)
            self.setpoint_var.set(0.9)
        self.simulate()
        
    def build_surface_tab(self):
        ttk.Label(self.tab_surface, text='Surface Configuration', style='Header.TLabel').pack(anchor='w', pady=(0, 15))
        frame = ttk.LabelFrame(self.tab_surface, text='Surface Type', padding=10)
        frame.pack(fill='x', pady=5)
        self.surface_combo = ttk.Combobox(frame, textvariable=self.surface_type_var, 
                                          values=['random', 'sine', 'step', 'spheres', 'pyramids', 'grating'], 
                                          state='readonly')
        self.surface_combo.pack(fill='x')
        self.create_labeled_scale(self.tab_surface, 'Grid Points', self.grid_var, 64, 256, '{:.0f}')
        self.create_labeled_scale(self.tab_surface, 'RMS Roughness', self.roughness_var, 0.1, 20, '{:.1f}', ' nm')
        self.create_labeled_scale(self.tab_surface, 'Scan Size', self.scan_size_var, 100, 2000, '{:.0f}', ' nm')
        ttk.Button(self.tab_surface, text='🔄 Generate Surface', command=self.generate_surface).pack(fill='x', pady=15)
        
    def build_cantilever_tab(self):
        ttk.Label(self.tab_cantilever, text='Cantilever Dynamics', style='Header.TLabel').pack(anchor='w', pady=(0, 15))
        mech_frame = ttk.LabelFrame(self.tab_cantilever, text='Mechanical Properties (Auto-set by Mode)', padding=10)
        mech_frame.pack(fill='x', pady=5)
        self.create_labeled_scale(mech_frame, 'Spring Constant', self.spring_const_var, 0.01, 60, '{:.2f}', ' N/m')
        self.create_labeled_scale(mech_frame, 'Resonance Frequency', self.resonance_freq_var, 5, 500, '{:.0f}', ' kHz')
        self.create_labeled_scale(mech_frame, 'Quality Factor', self.q_factor_var, 10, 1000, '{:.0f}')
        
        tip_frame = ttk.LabelFrame(self.tab_cantilever, text='Tip Properties', padding=10)
        tip_frame.pack(fill='x', pady=5)
        self.tip_shape_combo = ttk.Combobox(tip_frame, textvariable=self.tip_shape_var,
                                            values=['parabolic', 'conical', 'spherical', 'flat'], state='readonly')
        self.tip_shape_combo.pack(fill='x', pady=5)
        self.create_labeled_scale(tip_frame, 'Tip Radius', self.tip_radius_var, 1, 50, '{:.0f}', ' nm')
        self.create_labeled_scale(tip_frame, 'Tip Modulus (Et)', self.tip_modulus_var, 50, 400, '{:.0f}', ' GPa')
        self.create_labeled_scale(tip_frame, 'Tip Poisson Ratio', self.tip_poisson_var, 0.1, 0.5, '{:.2f}', '')

    def build_environment_tab(self):
        ttk.Label(self.tab_environment, text='Environment & Physics', style='Header.TLabel').pack(anchor='w', pady=15)
        fd_frame = ttk.LabelFrame(self.tab_environment, text='Interaction (Leonard - Jones Model)', padding=10)
        fd_frame.pack(fill='x', pady=5)
        self.create_labeled_scale(fd_frame, 'Interaction Strength', self.epsilon_scaler_var, 0.1, 5.0, '{:.1f}', 'x')
        self.create_labeled_scale(fd_frame, 'Z min (nm)', self.z_min_var, -10, 0, '{:.1f}')
        self.create_labeled_scale(fd_frame, 'Z max (nm)', self.z_max_var, 1, 20, '{:.1f}')
        env_frame = ttk.LabelFrame(self.tab_environment, text='Ambient Conditions', padding=10)
        env_frame.pack(fill='x', pady=5)
        self.create_labeled_scale(env_frame, 'Temperature', self.temperature_var, 273, 323, '{:.0f}', ' K')
        self.create_labeled_scale(env_frame, 'Humidity', self.humidity_var, 0, 100, '{:.0f}', ' %')
        
    def build_multimodal_tab(self):
        ttk.Label(self.tab_multimodal, text='Multimodal Operation', style='Header.TLabel').pack(anchor='w', pady=15)
        mode_frame = ttk.LabelFrame(self.tab_multimodal, text='Primary Mode', padding=10)
        mode_frame.pack(fill='x', pady=5)
        self.mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var, values=['contact', 'tapping', 'non-contact'], state='readonly')
        self.mode_combo.pack(fill='x')
        self.mode_combo.bind('<<ComboboxSelected>>', self.update_mechanical_presets)
        
        multi_frame = ttk.LabelFrame(self.tab_multimodal, text='Secondary Channel', padding=10)
        multi_frame.pack(fill='x', pady=5)
        self.multi_combo = ttk.Combobox(multi_frame, textvariable=self.multimodal_var, 
                                        values=['None', 'C-AFM', 'KPFM', 'SThM'], state='readonly')
        self.multi_combo.pack(fill='x', pady=10)
        self.multi_combo.bind('<<ComboboxSelected>>', lambda e: self.simulate())
        
        self.mm_params_frame = ttk.Frame(multi_frame)
        self.mm_params_frame.pack(fill='x')
        self.cafm_frame = ttk.Frame(self.mm_params_frame)
        self.create_labeled_scale(self.cafm_frame, 'Sample Bias', self.bias_var, -5, 5, '{:.1f}', ' V')
        self.cafm_frame.pack(fill='x', pady=5)
        
        ttk.Button(self.tab_multimodal, text='▶ SIMULATE', command=self.simulate).pack(fill='x', pady=15)

    def create_labeled_scale(self, parent, text, variable, from_, to_, label_format='{:.1f}', suffix=''):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=5)
        header = ttk.Frame(frame)
        header.pack(fill='x')
        ttk.Label(header, text=text).pack(side='left')
        value_label = ttk.Label(header, text=label_format.format(variable.get()) + suffix, foreground='#00bfa5')
        value_label.pack(side='right')
        def update_label(*args):
            try: value_label.config(text=label_format.format(variable.get()) + suffix)
            except: pass
        variable.trace_add('write', update_label)
        ttk.Scale(frame, from_=from_, to=to_, variable=variable, orient='horizontal').pack(fill='x')

    def setup_plot_figures(self):
        fig_bg, ax_bg, text_color = '#1a1a1a', '#2b2b2b', 'white'
        self.fig_main = Figure(figsize=(7.6, 5.6), dpi=100, facecolor=fig_bg)
        gs = gridspec.GridSpec(2, 2, figure=self.fig_main, wspace=0.25, hspace=0.3)
        self.ax_surface = self.fig_main.add_subplot(gs[0, 0])
        self.ax_afm = self.fig_main.add_subplot(gs[0, 1])
        self.ax_multimodal = self.fig_main.add_subplot(gs[1, 0])
        self.ax_error = self.fig_main.add_subplot(gs[1, 1])
        
        for ax in [self.ax_surface, self.ax_afm, self.ax_multimodal, self.ax_error]:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors=text_color)
            for spine in ax.spines.values(): spine.set_color(text_color)
        
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=self.plot_tab_main)
        self.canvas_main.get_tk_widget().pack(fill='both', expand=True)
        
        self.fig_dynamics = Figure(figsize=(7.6, 5.6), dpi=100, facecolor=fig_bg)
        self.fig_dynamics.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.1)
        self.ax_force = self.fig_dynamics.add_subplot(111)
        self.ax_force.set_facecolor(ax_bg)
        self.ax_force.tick_params(colors=text_color)
        for spine in self.ax_force.spines.values(): spine.set_color(text_color)
        self.canvas_dynamics = FigureCanvasTkAgg(self.fig_dynamics, master=self.plot_tab_dynamics)
        self.canvas_dynamics.get_tk_widget().pack(fill='both', expand=True)
        
        self.fig_analysis = Figure(figsize=(7.6, 5.6), dpi=100, facecolor=fig_bg)
        gsA = gridspec.GridSpec(2, 2, figure=self.fig_analysis, wspace=0.25, hspace=0.35)
        self.ax_metrics = self.fig_analysis.add_subplot(gsA[0, 0])
        self.ax_psd = self.fig_analysis.add_subplot(gsA[0, 1])
        self.ax_line = self.fig_analysis.add_subplot(gsA[1, 0])
        self.ax_hist = self.fig_analysis.add_subplot(gsA[1, 1])
        for ax in [self.ax_metrics, self.ax_psd, self.ax_line, self.ax_hist]:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors=text_color)
            for spine in ax.spines.values(): spine.set_color(text_color)
        self.canvas_analysis = FigureCanvasTkAgg(self.fig_analysis, master=self.plot_tab_analysis)
        self.canvas_analysis.get_tk_widget().pack(fill='both', expand=True)
        
        self.fig_3d = Figure(figsize=(7.6, 5.6), dpi=100, facecolor=fig_bg)
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.plot_tab_3d)
        self.canvas_3d.get_tk_widget().pack(fill='both', expand=True)
        self.ax_3d_surface = self.fig_3d.add_subplot(221, projection='3d')
        self.ax_3d_afm = self.fig_3d.add_subplot(222, projection='3d')
        self.ax_3d_multi = self.fig_3d.add_subplot(223, projection='3d')
        self.ax_3d_tip = self.fig_3d.add_subplot(224, projection='3d')
        for ax in [self.ax_3d_surface, self.ax_3d_afm, self.ax_3d_multi, self.ax_3d_tip]:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color); ax.yaxis.label.set_color(text_color); ax.zaxis.label.set_color(text_color)

    def generate_surface(self):
        self.surface = None
        self.simulate()
    
    def simulate(self):
        if self.simulation_thread and self.simulation_thread.is_alive(): return
        params = {
            'grid_size': int(self.grid_var.get()),
            'scan_size_nm': self.scan_size_var.get(), 
            'roughness': self.roughness_var.get(),
            'surface_type': self.surface_type_var.get(),
            'spring_constant': self.spring_const_var.get(),
            'resonance_freq': self.resonance_freq_var.get() * 1e3,
            'quality_factor': self.q_factor_var.get(),
            'tip_radius': self.tip_radius_var.get() * 1e-9,
            'setpoint': self.setpoint_var.get(),
            'vibration_amplitude': self.vibration_var.get() * 1e-9,
            'acoustic_noise': self.acoustic_var.get(),
            'mode': self.mode_var.get(),
            'multimodal': self.multimodal_var.get() if self.multimodal_var.get() != 'None' else None,
            'bias': self.bias_var.get(),
            'scan_speed': self.scan_speed_var.get(),
            'tip_shape': self.tip_shape_var.get(),
            'tip_aspect': self.tip_aspect_var.get(),
            'z_min': self.z_min_var.get(),
            'z_max': self.z_max_var.get(),
            'epsilon_scaler': self.epsilon_scaler_var.get(),
            'tip_modulus': self.tip_modulus_var.get() * 1e9, 
            'tip_poisson': self.tip_poisson_var.get()
        }
        self.simulation_thread = threading.Thread(target=self._run_simulation, args=(params,))
        self.simulation_thread.start()
        self.root.after(100, self._check_simulation)
    
    def _run_simulation(self, params):
        if self.surface is None or self.surface.shape[0] != params['grid_size']:
            self.surface, self.props = generate_surface(params['grid_size'], params['roughness'], params['surface_type'])
        
        self.afm_image, self.amplitude_image, self.phase_image, self.multimodal_image = \
            self.quick_simulate(self.surface, self.props, params['mode'], params['multimodal'], params)
        
        self.simulation_queue.put('done')
    
    def _compute_operating_point_nm(self, z_nm, F_nN, mode):
        if mode == 'contact':
            idx_rep = np.where(F_nN > 0.05)[0] 
            if len(idx_rep) > 0: return float(z_nm[idx_rep[0]] - 0.5) 
            return -0.5
        idx_min = int(np.argmin(F_nN)) 
        min_force = F_nN[idx_min]
        if min_force >= -0.01: return 2.0
        if mode == 'tapping': return float(z_nm[idx_min] + 1.0)
        else: return float(z_nm[idx_min] + 3.0)
    
    def _compute_kint_N_per_m(self, z_nm, F_nN, z_op_nm):
        i0 = int(np.argmin(np.abs(z_nm - z_op_nm)))
        i0 = max(2, min(len(z_nm) - 3, i0))
        dz = (z_nm[i0 + 2] - z_nm[i0 - 2])
        dF = (F_nN[i0 + 2] - F_nN[i0 - 2])
        if abs(dz) < 1e-12: return 0.0
        return float(-dF / dz)

    def quick_simulate(self, surface, props, mode, multimodal, params):
        # 1. GEOMETRY: Tip Convolution (Correctly Scaled)
        px_nm = params['scan_size_nm'] / params['grid_size']
        tip_radius_nm = params['tip_radius'] * 1e9
        
        # Guardrail: Limit tip kernel size to prevent freezing
        tip_radius_px = min(tip_radius_nm / px_nm, 64.0)
        
        # Kernel values must be scaled by px_nm to represent height in nm
        tip_kernel_px = create_tip_kernel(tip_radius_px, params['tip_shape'], params['tip_aspect'])
        tip_kernel_nm = tip_kernel_px * px_nm 
        
        tip_structure = -tip_kernel_nm.copy()
        tip_structure[~np.isfinite(tip_structure)] = -np.inf 

        # Efficient footprint
        foot = np.isfinite(tip_structure)
        afm_image = grey_dilation(surface, footprint=foot, structure=tip_structure)
        
        # 2. PHYSICS: Virtual Indentation
        k_c = max(float(params['spring_constant']), 0.1) # Safety clamp
        interaction_strength = float(params['epsilon_scaler'])
        hamaker_map = props.get('hamaker', np.ones_like(surface) * 1e-19)
        tip_R_meters = params['tip_radius']
        
        sigma = 0.34e-9
        # F_adh approximation based on Sphere-Plane vdW
        F_adh_map = (hamaker_map * interaction_strength * tip_R_meters) / (6.0 * sigma**2)
        
        # Indentation in nm
        indentation_map_nm = (F_adh_map / k_c) * 1e9
        
        # SAFETY CLAMP: Prevent explosion in contact mode
        if mode == 'contact':
            indentation_map_nm *= 0.1 
            
        indentation_map_nm = np.clip(indentation_map_nm, -5.0, 5.0) 
        
        # Apply physics indentation
        afm_image = afm_image - indentation_map_nm
        
        # 3. ARTIFACTS: Scan Lag & Noise
        scan_speed = float(params.get('scan_speed', 1.0))
        lag_strength = 0.9 if mode == 'contact' else (0.5 if mode == 'tapping' else 0.2)
        alpha = 1.0 - np.clip(lag_strength * 0.12 * scan_speed, 0.05, 0.80)
        for i in range(1, afm_image.shape[1]):
            afm_image[:, i] = alpha * afm_image[:, i] + (1.0 - alpha) * afm_image[:, i - 1]
        
        noise_level = 0.1 * (params['vibration_amplitude'] * 1e9 + params['acoustic_noise'] / 100)
        afm_image += np.random.randn(*afm_image.shape) * noise_level
        
        # 4. REST OF CALCULATIONS
        zmin_m = float(params['z_min']) * 1e-9
        zmax_m = float(params['z_max']) * 1e-9
        
        E_s = float(np.mean(props.get('elastic_modulus', 100e9)))
        nu_s = float(np.mean(props.get('poisson', 0.3)))
        E_t = params.get('tip_modulus', 170e9)
        nu_t = params.get('tip_poisson', 0.27)
        term_sample = (1 - nu_s**2) / E_s
        term_tip = (1 - nu_t**2) / E_t
        E_reduced = 1.0 / (term_sample + term_tip)
        
        hamaker = float(np.mean(props.get('hamaker', 1e-19)))
        
        z_nm, F_nN, _ = generate_analytical_fd_curve(
            (zmin_m, zmax_m),
            tip_radius=params['tip_radius'], epsilon_scaler=params['epsilon_scaler'],
            E_reduced=E_reduced, hamaker=hamaker
        )
        
        z_op = self._compute_operating_point_nm(z_nm, F_nN, mode)
        k_int_base = self._compute_kint_N_per_m(z_nm, F_nN, z_op) 
        
        norm = (afm_image - afm_image.min()) / (afm_image.max() - afm_image.min() + 1e-9)
        k_int_map = k_int_base * (1.0 + 0.20 * (norm - 0.5))
        
        # Resonance Model (Steady-State Approximation)
        f0 = float(params['resonance_freq'])
        f_drive = f0 
        Q = float(params['quality_factor'])
        stiff_ratio = np.maximum(1.0 + (k_int_map / max(k_c, 1e-9)), 0.01)
        f_eff = f0 * np.sqrt(stiff_ratio)
        r = f_drive / f_eff
        denom = np.sqrt((1.0 - r**2)**2 + (r / Q)**2)
        amp = 1.0 / denom
        phase = np.degrees(np.arctan2((r / Q), (1.0 - r**2)))
        
        if mode == 'tapping':
            amplitude_image = amp * params['setpoint']
            phase_image = phase
        elif mode == 'non-contact':
            amplitude_image = np.ones_like(surface) * params['setpoint'] + np.random.randn(*surface.shape) * 0.01
            phase_image = phase
        else: 
            amplitude_image = k_int_map
            phase_image = np.zeros_like(surface)
            
        amplitude_image += np.random.randn(*surface.shape) * 0.02
        phase_image += np.random.randn(*surface.shape) * 0.4
        phase_image = gaussian_filter(phase_image, sigma=1.0)
        
        multimodal_image = np.zeros_like(surface)
        if multimodal == 'C-AFM': multimodal_image = params['bias'] * props['conductivity'] * 1e12
        elif multimodal == 'KPFM': multimodal_image = props['cpd'] * 1000
        elif multimodal == 'SThM': multimodal_image = props['thermal_conductivity'] / 10
        
        return afm_image, amplitude_image, phase_image, multimodal_image

    def _radial_psd(self, img2d, scan_size_nm):
        img = np.array(img2d, dtype=float)
        n = img.shape[0]
        if n < 8: return np.array([1.0]), np.array([1.0])
        img = img - np.mean(img)
        dx = float(scan_size_nm) / float(n) 
        F = np.fft.fftshift(np.fft.fft2(img))
        P = (np.abs(F) ** 2) / (n * n)
        fx = np.fft.fftshift(np.fft.fftfreq(n, d=dx)) 
        fy = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
        FX, FY = np.meshgrid(fx, fy)
        FR = np.sqrt(FX ** 2 + FY ** 2)
        fr = FR.ravel()
        p = P.ravel()
        mask = fr >= 0
        fr = fr[mask]; p = p[mask]
        nbins = max(25, n // 3)
        fmax = float(np.max(fr))
        if fmax <= 0: return np.array([0.0]), np.array([np.mean(p)])
        bins = np.linspace(0.0, fmax, nbins + 1)
        which = np.digitize(fr, bins) - 1
        psd = np.zeros(nbins, dtype=float)
        cnt = np.zeros(nbins, dtype=float)
        for i in range(nbins):
            m = (which == i)
            if np.any(m):
                psd[i] = np.mean(p[m])
                cnt[i] = 1.0
        centers = 0.5 * (bins[:-1] + bins[1:])
        good = cnt > 0
        return centers[good], psd[good]

    def _check_simulation(self):
        try:
            self.simulation_queue.get_nowait()
            self.update_plots()
        except queue.Empty:
            self.root.after(100, self._check_simulation)
    
    def update_plots(self):
        if self.surface is None: return
        for ax in [self.ax_surface, self.ax_afm, self.ax_multimodal, self.ax_error]: ax.clear()
        
        extent = [0, self.scan_size_var.get(), 0, self.scan_size_var.get()]
        def safe_colorbar(im, ax, key):
            cb = self.colorbars.get(key)
            if cb:
                try: cb.ax.clear(); cb = self.fig_main.colorbar(im, cax=cb.ax); self.colorbars[key] = cb
                except: cb = self.fig_main.colorbar(im, ax=ax, fraction=0.046); self.colorbars[key] = cb
            else: cb = self.fig_main.colorbar(im, ax=ax, fraction=0.046); self.colorbars[key] = cb
            cb.ax.yaxis.set_tick_params(color='white', labelcolor='white')
            if cb.outline: cb.outline.set_edgecolor('white')

        if 'material_map' in self.props:
            mat_map = self.props['material_map']
            h, w = mat_map.shape
            rgb_vis = np.zeros((h, w, 3))
            used_materials = np.unique(mat_map)
            for mat_name in used_materials:
                if mat_name in MAT_COLORS:
                    mask = (mat_map == mat_name)
                    color = MAT_COLORS[mat_name]
                    rgb_vis[mask] = color
            norm_height = (self.surface - self.surface.min()) / (self.surface.max() - self.surface.min() + 1e-9)
            norm_height = 0.6 + 0.4 * norm_height
            rgb_vis = rgb_vis * norm_height[:,:,None]
            self.ax_surface.imshow(rgb_vis, extent=extent)
            self.ax_surface.set_title('Sample Material Map', color='white')
            patches = [mpatches.Patch(color=MAT_COLORS[m], label=m) for m in used_materials if m in MAT_COLORS]
            self.ax_surface.legend(handles=patches, loc='upper right', fontsize=7, facecolor='#2b2b2b', labelcolor='white')
            if 'surface' in self.colorbars: 
                try: self.colorbars['surface'].remove()
                except: pass
        else:
            im1 = self.ax_surface.imshow(self.surface, cmap='viridis', extent=extent)
            self.ax_surface.set_title('Surface Topography', color='white')
            safe_colorbar(im1, self.ax_surface, 'surface')
        
        im2 = self.ax_afm.imshow(self.afm_image, cmap='viridis', extent=extent)
        self.ax_afm.set_title(f'AFM ({self.mode_var.get()})', color='white')
        safe_colorbar(im2, self.ax_afm, 'afm')
        
        if self.multimodal_var.get() != 'None':
            cmap = {'C-AFM': 'hot', 'KPFM': 'seismic', 'SThM': 'coolwarm'}
            im3 = self.ax_multimodal.imshow(self.multimodal_image, cmap=cmap.get(self.multimodal_var.get(), 'viridis'), extent=extent)
            self.ax_multimodal.set_title(self.multimodal_var.get(), color='white')
            safe_colorbar(im3, self.ax_multimodal, 'multi')
        else:
            if 'multi' in self.colorbars: 
                try: self.colorbars['multi'].remove(); del self.colorbars['multi']
                except: pass
            self.ax_multimodal.text(0.5, 0.5, "No Channel", color="gray", ha="center")
            self.ax_multimodal.set_title("Multimodal", color="white")
        
        error = self.afm_image - self.surface
        im6 = self.ax_error.imshow(error, cmap='RdBu_r', extent=extent)
        self.ax_error.set_title('Error Signal', color='white')
        safe_colorbar(im6, self.ax_error, 'error')
        
        self.canvas_main.draw()
        self.update_dynamics_plots()
        self.update_analysis_plots()
        self.update_3d_plots()
    
    def update_dynamics_plots(self):
        self.ax_force.clear()
        
        E_s = 100e9
        nu_s = 0.3
        if getattr(self, 'props', None) is not None:
            try:
                E_s = float(np.mean(self.props.get('elastic_modulus', E_s)))
                nu_s = float(np.mean(self.props.get('poisson', nu_s)))
            except: pass
            
        E_t = self.tip_modulus_var.get() * 1e9
        nu_t = self.tip_poisson_var.get()
        
        term_sample = (1 - nu_s**2) / E_s
        term_tip = (1 - nu_t**2) / E_t
        E_reduced = 1.0 / (term_sample + term_tip)
        
        hamaker = 1e-19
        if getattr(self, 'props', None) is not None:
             hamaker = float(np.mean(self.props.get('hamaker', hamaker)))

        tip_R = self.tip_radius_var.get() * 1e-9
        
        z_nm, approach, retract = generate_analytical_fd_curve(
            (self.z_min_var.get() * 1e-9, self.z_max_var.get() * 1e-9),
            tip_radius=tip_R, epsilon_scaler=self.epsilon_scaler_var.get(),
            E_reduced=E_reduced, hamaker=hamaker
        )
        
        sigma_nm = 0.34 
        self.ax_force.plot(z_nm, approach, 'w-', linewidth=2.5, label='Approach')
        self.ax_force.plot(z_nm, retract, 'm--', linewidth=1.5, label='Retract')
        
        f_min = np.min(approach)
        well_depth = abs(f_min) if abs(f_min) > 1e-12 else 1.0
        y_lower = f_min * 1.5  
        y_upper = well_depth * 4.0 
        self.ax_force.set_ylim(y_lower, y_upper)
        self.ax_force.set_xlim(min(z_nm), max(z_nm))
             
        ylim = self.ax_force.get_ylim()
        cutoff = sigma_nm
        self.ax_force.axvspan(np.min(z_nm), cutoff, color='#448aff', alpha=0.3, label='Contact/Repulsive')
        self.ax_force.axvspan(cutoff, np.max(z_nm), color='#69f0ae', alpha=0.2, label='Non-contact')
        
        self.ax_force.axvline(sigma_nm, color='white', linestyle='--', linewidth=1)
        self.ax_force.text(sigma_nm, ylim[0] + (ylim[1]-ylim[0])*0.1, ' σ', color='white', fontsize=12)
        self.ax_force.axhline(0, color='gray', linestyle=':', linewidth=1)

        mode = self.mode_var.get()
        z_op = self._compute_operating_point_nm(z_nm, approach, mode)
        op_y = np.interp(z_op, z_nm, approach)
        self.ax_force.plot(z_op, op_y, 'ro', markersize=8, markeredgecolor='white', label='Setpoint')

        self.ax_force.set_title(f'Leonard-Jones Model', color='white')
        self.ax_force.set_xlabel('Distance (nm)', color='white')
        self.ax_force.set_ylabel('Force (nN)', color='white')
        self.ax_force.legend(facecolor='#2b2b2b', labelcolor='white', fontsize=9, loc='upper right')
        self.ax_force.grid(True, alpha=0.2)
        self.canvas_dynamics.draw()

    def update_analysis_plots(self):
        self.ax_metrics.clear(); self.ax_psd.clear(); self.ax_line.clear(); self.ax_hist.clear()
        if self.surface is None: self.canvas_analysis.draw(); return
        
        scan_nm = float(self.scan_size_var.get())
        mode = self.mode_var.get()
        surf = np.array(self.surface, dtype=float).ravel()
        afm = np.array(self.afm_image, dtype=float).ravel()
        err = afm - surf
        
        rms_s = np.std(surf); rms_a = np.std(afm); rms_e = np.std(err)
        ra_s = np.mean(np.abs(surf - np.mean(surf)))
        ra_a = np.mean(np.abs(afm - np.mean(afm)))
        ra_e = np.mean(np.abs(err - np.mean(err)))
        sk_s = skew(surf); sk_a = skew(afm)
        ku_s = kurtosis(surf); ku_a = kurtosis(afm)

        self.ax_metrics.set_title(f"Surface Metrology (Mode: {mode})", color='white', fontsize=10, pad=10)
        self.ax_metrics.axis('off')
        
        columns = ('Metric', 'Ideal Surface', 'AFM Image', 'Error/Diff')
        cell_text = [
            ['RMS (nm)', f'{rms_s:.3f}', f'{rms_a:.3f}', f'{rms_e:.3f}'],
            ['Ra (nm)',  f'{ra_s:.3f}', f'{ra_a:.3f}', f'{ra_e:.3f}'],
            ['Skew (Rsk)', f'{sk_s:.2f}', f'{sk_a:.2f}', f'{sk_a-sk_s:.2f}'],
            ['Kurt (Rku)', f'{ku_s:.2f}', f'{ku_a:.2f}', f'{ku_a-ku_s:.2f}'],
            ['P-V (nm)',   f'{np.ptp(surf):.1f}', f'{np.ptp(afm):.1f}', f'{np.ptp(err):.1f}']
        ]
        
        table = self.ax_metrics.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.8) 
        
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#444444') 
            if row == 0:
                cell.set_facecolor('#00bfa5'); cell.set_text_props(color='black', weight='bold')
            else:
                cell.set_facecolor('#2b2b2b'); cell.set_text_props(color='white')
                if col == 0: cell.set_facecolor('#333333'); cell.set_text_props(weight='bold', color='#80cbc4')

        fr_s, psd_s = self._radial_psd(self.surface, scan_nm)
        fr_a, psd_a = self._radial_psd(self.afm_image, scan_nm)
        self.ax_psd.set_title("Power Spectral Density", color='white')
        self.ax_psd.loglog(fr_s, psd_s, color='#00e5ff', linewidth=1.5, label='Ideal Surface')
        self.ax_psd.loglog(fr_a, psd_a, color='#ff4081', linestyle='--', linewidth=1.5, label='AFM Scan')
        self.ax_psd.set_xlabel("Frequency (cycles/nm)", color='white')
        self.ax_psd.set_ylabel("Power ($nm^3$)", color='white')
        self.ax_psd.grid(True, which="both", ls="-", alpha=0.15)
        self.ax_psd.legend(facecolor='#2b2b2b', labelcolor='white', fontsize=8)

        mid = self.surface.shape[0] // 2
        xp = np.linspace(0, scan_nm, self.surface.shape[1])
        self.ax_line.set_title(f"Line Profile (y = {mid})", color='white')
        self.ax_line.plot(xp, self.surface[mid, :], color='#00e5ff', linewidth=2, alpha=0.7, label='Surface')
        self.ax_line.plot(xp, self.afm_image[mid, :], color='#ff4081', linestyle='--', linewidth=1.5, label='AFM')
        self.ax_line.fill_between(xp, self.surface[mid, :], self.afm_image[mid, :], color='red', alpha=0.3, label='Error')
        self.ax_line.set_xlabel("Position (nm)", color='white')
        self.ax_line.set_ylabel("Height (nm)", color='white')
        self.ax_line.grid(True, alpha=0.2)
        self.ax_line.legend(facecolor='#2b2b2b', labelcolor='white', fontsize=8)

        self.ax_hist.set_title("Height Distribution", color='white')
        self.ax_hist.hist(surf, bins=50, density=True, color='#00e5ff', alpha=0.4, label='Surface')
        self.ax_hist.hist(afm, bins=50, density=True, color='#ff4081', alpha=0.4, label='AFM')
        ax2 = self.ax_hist.twinx()
        err_hist, err_bins = np.histogram(err, bins=50, density=True)
        err_centers = (err_bins[:-1] + err_bins[1:]) / 2
        ax2.plot(err_centers, err_hist, 'w:', linewidth=1, label='Error Dist')
        ax2.set_yticks([]) 
        self.ax_hist.set_xlabel("Height (nm)", color='white')
        self.ax_hist.grid(True, alpha=0.2)
        self.ax_hist.legend(loc='upper left', facecolor='#2b2b2b', labelcolor='white', fontsize=7)
        ax2.legend(loc='upper right', facecolor='#2b2b2b', labelcolor='white', fontsize=7)
        self.canvas_analysis.draw()

    def update_3d_plots(self):
        for ax in [self.ax_3d_surface, self.ax_3d_afm, self.ax_3d_multi, self.ax_3d_tip]: ax.clear()
        stride = max(1, self.surface.shape[0] // 64) 
        x = np.linspace(0, self.scan_size_var.get(), self.surface.shape[0])
        y = np.linspace(0, self.scan_size_var.get(), self.surface.shape[1])
        X, Y = np.meshgrid(x, y)
        self.ax_3d_surface.plot_surface(X, Y, self.surface, cmap='viridis', rstride=stride, cstride=stride, linewidth=0, antialiased=False)
        self.ax_3d_surface.set_title('Original Surface', color='white')
        self.ax_3d_afm.plot_surface(X, Y, self.afm_image, cmap='viridis', rstride=stride, cstride=stride, linewidth=0, antialiased=False)
        self.ax_3d_afm.set_title('AFM Topography', color='white')
        
        if self.multimodal_var.get() != 'None':
            multi_data = self.multimodal_image.copy()
            data_min, data_max = multi_data.min(), multi_data.max()
            if data_max - data_min > 0:
                multi_norm = (multi_data - data_min) / (data_max - data_min)
                plot_data = multi_norm * np.abs(self.surface.max() - self.surface.min())
                if self.multimodal_var.get() in ['KPFM']: plot_data = self.surface * 0.3 + plot_data * 0.7
            else: plot_data = self.surface * 0.3
            self.ax_3d_multi.plot_surface(X, Y, plot_data, cmap='inferno', rstride=stride, cstride=stride, linewidth=0, antialiased=False, alpha=0.9)
            self.ax_3d_multi.set_title(f'{self.multimodal_var.get()} Signal', color='white')
        else:
            self.ax_3d_multi.text2D(0.5, 0.5, "No Signal", transform=self.ax_3d_multi.transAxes, color='white', ha='center')
            self.ax_3d_multi.set_title('Multimodal (None)', color='white')

        px_nm = self.scan_size_var.get() / self.grid_var.get()
        tip_radius_nm = self.tip_radius_var.get()
        tip_radius_px = min(tip_radius_nm / px_nm, 64.0)
        
        tip_k_px = create_tip_kernel(tip_radius_px, self.tip_shape_var.get(), self.tip_aspect_var.get())
        tip_k_nm = tip_k_px * px_nm 
        tip_k_vis = np.nan_to_num(tip_k_nm, nan=np.nanmax(tip_k_nm)) 
        
        tx = np.arange(tip_k_nm.shape[0]) - tip_k_nm.shape[0]//2
        ty = np.arange(tip_k_nm.shape[1]) - tip_k_nm.shape[1]//2
        TX, TY = np.meshgrid(tx, ty)
        self.ax_3d_tip.plot_surface(TX, TY, tip_k_vis.max() - tip_k_vis, cmap='copper', rstride=1, cstride=1, linewidth=0, antialiased=True)
        self.ax_3d_tip.set_title('Tip Geometry (NM Scaled)', color='white')
        self.canvas_3d.draw()

if __name__ == '__main__':
    root = tk.Tk()
    app = AFMSimulatorGUI(root)
    root.mainloop()
