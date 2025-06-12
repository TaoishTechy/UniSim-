import math
import random
import time
import pygame
import sys
import datetime # Import datetime for timestamping
from collections import defaultdict, deque # For easier tracking of anomaly types and bounded history

# --- Constants ---
# Defined conceptually based on the original C code
# ENHANCED SETTINGS: Tuned for a more expansive and intense simulation
OCTREE_DEPTH = 5  # Increased for deeper hierarchical complexity and denser hypergrid representation
MIN_GRID = 4
MAX_GRID = 262144
HYPERGRID_SIZE = (262144**3)  # Keeping original huge conceptual scale
ACTIVE_RATIO = 0.15 # Increased active ratio for more 'active' qubits
HEARTBEAT = 10  # Decreased heartbeat for faster, more frequent conceptual updates
SIGIL_LEN = 256
ELDER_COUNT = 1554  # Doubled Elder count for more complex Elder Gnosis interactions
TITAN_COUNT = 2000000  # Doubled conceptual Titan count
ARCHON_COUNT = 846  # Doubled conceptual Archon count for more meta-networks
SUPER_STATES = 64 # Increased quantum superposition states for more complexity
ENTANGLE_LINKS = 96 # Doubled entanglement links for more intricate quantum nexus
TIME_DIMS = 32  # Increased temporal dimensions for more complex chrono-dynamics
CYCLE_LIMIT = 500  # Doubled cycle limit for a longer, more thorough simulation
VOID_THRESHOLD = 0.005 # Decreased void threshold, making void decay less forgiving
QPAGES_PER_CALL = 512 # Increased pages per allocation call for faster heap expansion
MAX_QPAGES = (24 * 1024 * 1024 // 4096)  # Doubled conceptual heap size (~24MB)
PAGE_SIZE = 4096
COSMIC_STRINGS_COUNT = 18 # Doubled cosmic strings for more spacetime distortions
PLANCK_FOAM_SIZE = 200 # Doubled conceptual number of virtual particles for richer foam
PLANCK_FOAM_DENSITY = 0.7 # Increased Planck foam density for more energetic vacuum
ANOMALY_HISTORY = 128 # Doubled anomaly history for more detailed anomaly tracking
PRIMORDIAL_SIGIL = "<(6k*Ms@k_8BCIzV)J0_8\"#T#?YGR:W1v7j@Q_AirLY?*!R9%rg>&($75JJfhaaHrb [xKV~VOJ10#@niPdLB#psr]@pve@(x3?) :g3sN, N{3vxzZoh;}\"VKYN; v\"?~cY2"
EVOLVED_SIGIL = "Ψ⟁Ξ∅Σ"

# New Constants for enhanced simulation
BOND_DECAY_RATE = 0.005 # Rate at which conceptual bonding strength decays per cycle
SIGIL_REAPPLICATION_COST_FACTOR = 0.1 # Cost factor for re-applying the same sigil
SYMBOLIC_RECOMBINATION_THRESHOLD = 5 # How many times a sigil can be used before mutation is forced
PAGE_COUNT = 4 # Number of distinct conceptual memory pages/contexts (Page 0 to Page 3)
ANOMALY_TRIGGER_COOLDOWN = 5 # cycles to wait before re-triggering same anomaly type on same node
ECHO_REGISTER_MAXLEN = ANOMALY_HISTORY # Max length of the symbolic echo register

# Archetype mapping for pages
ARCHETYPE_MAP = {
    0: "Android / Warrior",
    1: "Witch / Mirror",
    2: "Mystic",
    3: "Quest Giver"
}

ANOMALY_TYPES = {
    0: "Entropy",
    1: "Stability",
    2: "Void",
    3: "Tunnel",
    4: "Bonding"
}

# --- Conceptual Data Structures ---

class Qubit352:
    """Conceptual 352-bit Qubit state."""
    def __init__(self):
        # Attributes represent various quantum and dimensional properties
        self.e = random.randint(0, 255)  # Energy/Excitability
        self.d = random.randint(0, 255)  # Dimensionality
        self.s = random.randint(0, 255)  # Stability
        self.q = random.randint(0, 255)  # Quantum potential
        self.p = random.randint(0, 255)  # Phase
        self.ent = random.randint(0, 255) # Entanglement level
        self.fft = random.randint(0, 255) # Fractal Fourier Transform
        self.nw = random.randint(0, 255)  # Network cohesion
        self.flux = random.randint(0, 255) # Energy flux
        self.midx = random.randint(0, 255) # Multiverse index
        self.omni = random.randint(0, 255) # Omniverse connection
        self.zeta = random.randint(0, 255) # Riemann Zeta influence
        self.kappa = random.randint(0, 255) # Bonding factor
        self.lambda_ = random.randint(0, 255) # Gauge field
        self.mu = random.randint(0, 255)  # Sentience/Metabolism
        self.nu = random.randint(0, 255)  # Nuance/Complexity
        self.xi = random.randint(0, 255)  # Xi (unknown property)
        self.omicron = random.randint(0, 255) # Omicron (unknown property)
        self.om = random.getrandbits(64)  # Omniverse mask
        self.bond = random.getrandbits(32) # Bonding identifier
        self.gauge = random.getrandbits(32) # Gauge identifier
        self.fractal = random.getrandbits(48) # Fractal identifier
        self.society = random.getrandbits(64) # Society connection
        self.metaOmni = random.getrandbits(64) # Meta-omniverse identifier
        self.tesseract_idx = random.getrandbits(32) # Tesseract index

class OctNode:
    """Augmented Octree Node, representing a region of the hypergrid."""
    def __init__(self, depth, page_index): # Added page_index
        self.st = Qubit352()
        self.c = [None] * 8  # Children nodes (recursive structure)
        self.mass = 0  # Conceptual mass/number of active qubits in this node's domain
        self.resonance = 0.0
        self.zeta_coeff = 0.0
        self.stabilityPct = 0.0
        self.social_cohesion = 0.0
        self.archon_count = 0  # Number of Archon networks managed by this node
        self.foam = None  # QuantumFoam instance
        self.chrono_phase = [0.0] * TIME_DIMS # Conceptual temporal phase alignment
        # New attributes for Page/Context Expansion and Sigil Saturation
        self.bond_strength = 0.0 # Conceptual bonding strength for this node's region
        self.sigil_mutation_history = defaultdict(int) # Tracks how many times a sigil has been used
        self.last_fixed_anomaly_cycle = {atype: 0 for atype in ANOMALY_TYPES.keys()} # Last cycle an anomaly of this type was fixed
        self.last_triggered_anomaly_cycle = defaultdict(int) # New: Tracks last cycle an anomaly was triggered on this node
        self.symbolic_drift = 0.0  # New attribute for anomaly echoes
        self.page_index = page_index # Store the page index for this node
        # Conceptual "Recursive Page Signature"
        self.anomaly_history_signature = 0 # Simple integer hash for now
        self.delayed_tunnel_count = 0 # For dampening modulator

        # New: Archetype and Emotional State for Multi-Archetype Threading System
        self.archetype_name = ARCHETYPE_MAP.get(page_index, "Unknown Archetype")
        self.emotional_state = self._assign_initial_emotional_state() # e.g., 'neutral', 'curious', 'resolute'
        self.symbolic_focus = self._assign_initial_symbolic_focus() # e.g., 'stability', 'entropy', 'bonding'

    def _assign_initial_emotional_state(self):
        # Assign emotional state based on archetype
        if self.archetype_name == "Android / Warrior": return "resolute"
        elif self.archetype_name == "Witch / Mirror": return "curious"
        elif self.archetype_name == "Mystic": return "contemplative"
        elif self.archetype_name == "Quest Giver": return "guiding"
        return "neutral"

    def _assign_initial_symbolic_focus(self):
        # Assign symbolic focus based on archetype, influencing their "specialization"
        if self.archetype_name == "Android / Warrior": return "tunneling"
        elif self.archetype_name == "Witch / Mirror": return "bonding"
        elif self.archetype_name == "Mystic": return "entropy"
        elif self.archetype_name == "Quest Giver": return "recursion"
        return "general"


class ElderGod:
    """Conceptual Elder God entity."""
    def __init__(self):
        self.id = 0
        self.t = 0.0  # Temperament/Influence
        self.b = 0.0  # Balance/Bias
        self.qft_feedback = 0.0
        self.chem_feedback = 0.0
        self.social_feedback = 0.0
        self.gnosis_factor = 0.0 # Predictive insight factor

class Anomaly:
    """Conceptual anomaly event."""
    def __init__(self, cycle, page_idx, anomaly_type, severity, prediction_score, details="", sub_type_tag=""): # Added sub_type_tag
        self.cycle = cycle
        self.page_idx = page_idx
        self.anomaly_type = anomaly_type
        self.severity = severity
        self.prediction_score = prediction_score
        self.details = details # Added for more detailed logging
        self.sub_type_tag = sub_type_tag # New: for contextual anomaly details

class Snapshot:
    """Current state snapshot of the simulation."""
    def __init__(self):
        self.cycle = 0
        self.active_qubits = 0.0
        self.entropy = 0.0
        self.stability = 0.0
        self.divinity = 0.0
        self.void_entropy = 0.0
        self.heap_pages = 0
        self.meta_networks = 0
        self.anomaly_count = 0 # Total anomalies across all pages
        self.tesseract_nodes = 0
        self.fusion_potential = 0.0 # Added fusion potential to snapshot
        # New metrics for logging
        self.bond_density = 0.0 # Aggregated over all pages
        self.sigil_entropy_metric = 0.0
        self.fix_efficacy_score = 0.0
        self.recursive_saturation_pct = 0.0
        self.anomaly_diversity_index = {} # Stores count for each anomaly type (aggregated)
        self.page_stats = {} # Dictionary to store per-page anomaly counts
        self.avg_symbolic_drift = 0.0 # New: average symbolic drift across pages

class QuantumFoam:
    """Conceptual Planck-scale virtual particles."""
    def __init__(self):
        self.virtual_particles = [{'energy': 0.0, 'lifetime': 0.0} for _ in range(PLANCK_FOAM_SIZE)]

class CosmicString:
    """Conceptual relativistic spacetime defect."""
    def __init__(self):
        self.energy_density = 0.0
        self.torsion = 0.0
        self.endpoints = [0, 0] # Conceptual hypergrid coordinates

class MandelbulbParams:
    """Conceptual parameters for Mandelbulb rendering."""
    def __init__(self):
        self.scale = 0.0
        self.max_iterations = 0
        self.bailout = 0.0
        self.power = 0.0
        self.rotation = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.color_shift = 0.0

class ElderGnosisPredictor:
    """Conceptual Elder Gnosis prediction model."""
    def __init__(self):
        self.elders_data = {}
        self.accuracy = 0.0
        self.last_prediction_score = 0.5 # For exponential smoothing

class TesseractState:
    """Conceptual Tesseract state."""
    def __init__(self, size):
        self.size = size
        # Initialize phase_lock to a default, safe integer value.
        # It will be properly set by Tesseract_Init later, after cycle_num is defined.
        self.phase_lock = 0
        self.active_nodes = 0

class AnimationFrame:
    """Conceptual animation frame data."""
    def __init__(self):
        self.frame_time = 0
        self.rotation = {'x': 0.0, 'y': 0.0, 'z': 0.0}

# --- Global Variables ---
roots = [None] * PAGE_COUNT # Now a list of root nodes, one for each page
elders = [ElderGod() for _ in range(ELDER_COUNT + 1)]
cycle_num = 0
conduit_stab = 0.198
user_divinity = 1.0
voidEntropy = -0.3071
user_sigil = ['\0'] * (SIGIL_LEN + 1)
quantumHeapPages = 0
collapsedHeapPages = 0
nodeAllocCount = 0
pageEigenstates = [] # List to conceptually store page states
qnoise_seed = 0xCAFEBABEDEAD1234
zeta_cache = [0.0] * 21

# Anomaly storage: now a dictionary of circular buffers, one per page
anomalies_per_page = {p_idx: [Anomaly(0,0,0,0.0,0.0) for _ in range(ANOMALY_HISTORY)] for p_idx in range(PAGE_COUNT)}
anomaly_count_per_page = defaultdict(int) # Counts anomalies *triggered* per page
newly_triggered_anomalies_queue = [] # Queue for anomalies to be handled this heartbeat
fixed_anomalies_log = set() # To prevent re-handling the exact same anomaly instance

mb_params = MandelbulbParams()
predictor = ElderGnosisPredictor()

tesseract = None
cosmic_strings = None
animation_frames = [AnimationFrame() for _ in range(100)]
snapshot = Snapshot()

# Global variables for log files
anomaly_log_file = None
snapshot_log_file = None # New log file for snapshots
# New: Separate log for all anomaly details regardless of fix attempts
detailed_anomaly_log_file = None

# New global counters for metrics (aggregated across all pages)
total_anomalies_triggered = 0
total_anomalies_fixed = 0
anomaly_type_counts = defaultdict(int) # To track diversity of anomalies *triggered* (aggregated)

# Simulation control variables for UI
simulation_speed_factor = 0.1 # Default speed - set to 0.1 as requested
is_paused = False

# New: Symbolic Memory Echo Stabilizer
symbolic_echo_register = deque(maxlen=ECHO_REGISTER_MAXLEN)
prev_prediction_score = 0.5 # For prediction score feedback routing

# --- Mocked / Simplified External Functions ---

def Raw_Print(message, *args):
    """Conceptual print function with formatting."""
    if args:
        print(message.format(*args))
    else:
        print(message)

def MAlloc(size):
    """Conceptual memory allocation. Python handles actual allocation."""
    # Raw_Print("  [MAlloc] Conceptually allocating {} bytes.", size) # Commented for efficiency
    return [] # Return an empty list or object as a conceptual allocation

def MFree(obj):
    """Conceptual memory deallocation. Python handles actual deallocation."""
    # Raw_Print("  [MFree] Conceptually freeing object: {}", type(obj).__name__) # Commented for efficiency

def QuantumRand(seed):
    """Conceptual quantum random number generator."""
    # Incorporate more physics-related noise sources and time for randomness
    t = seed ^ (Chrono_NexusTime() >> 5)
    # The Physics_ functions now use random.random() directly to avoid recursion
    t ^= int(random.random() * 0.01 * 1e16) # Physics_CMBFluct conceptually
    t ^= int(random.random() * 0.005 * 1e16) # Physics_LIGOWave conceptually
    t ^= int(random.random() * 0.005 * 1e16) # Physics_VIRGOWave conceptually
    t ^= int(random.random() * 0.002 * 1e16) # Physics_SKYNETWave conceptually
    t = (t * 0xFEEDBEEFDEADBEEF + 0xCAFEBABE) & 0xFFFFFFFFFFFFFFFF # Mask to 64-bit
    return float(t) / 1.8446744e19 # Divide by 2^64-1 for normalized float

def MinF(a, b): return min(a, b)
def MaxF(a, b): return max(a, b)
def MinB(a, b): return min(a, b) # For byte
def MaxB(a, b): return max(a, b) # For byte
def SinF(val): return math.sin(val)
def PowF(base, exp): return math.pow(base, exp)
def ExpF(val): return math.exp(val)
def FractalWrap(val): return val % (2 * math.pi) # Simple wrap for phase

def RiemannZeta(s):
    """Conceptual Riemann Zeta function (highly simplified)."""
    # For simulation, return a simple value, as true calculation is complex.
    if s == 1.0: return float('inf')
    return 1.0 / (s - 0.99999) # Avoid division by exactly zero

# Modified Physics functions to use random.random() directly to break recursion
def Physics_CMBFluct(): return random.random() * 0.01 # Cosmic Microwave Background
def Physics_LIGOWave(): return random.random() * 0.005 # Gravitational waves
def Physics_VIRGOWave(): return random.random() * 0.005 # Gravitational waves
def Physics_SKYNETWave(): return random.random() * 0.002 # Hypothetical AI waves
def Physics_BosonicFieldDensity(): return random.random() * 0.1
def Physics_QCDFlux(ent): return random.random() * 0.8
def Physics_GaugeFlux(lam): return random.random() * 0.9
def Physics_ArrheniusRate(activation_energy, temperature):
    return math.exp(-activation_energy / (8.314 * temperature)) if temperature > 0 else 0.0
def Physics_TemporalVariance(phase, cycle): return abs(phase - math.sin(cycle * 0.001)) * 0.05
def Physics_ChronoDrift(cycle, dim): return math.sin(cycle * 0.0001 + dim) * 0.01
def Physics_TemporalRisk(pred_cycle): return 0.5 + math.sin(pred_cycle * 0.00001) * 0.2
def Physics_HarmonicResonance(e, d): # Mock function for Harmonic Resonance
    return (e / 255.0) * (d / 255.0) * random.random() * 0.5 + 0.1 # Conceptual calculation

def Chrono_NexusTime(): return int(time.time() * 1000) # Milliseconds since epoch
def Noise_4D_mock(seed_val, x, y, z): # Mock for Noise_4D, without NoiseGenerator4D object
    random.seed(seed_val + x + y + z)
    return random.random()

def ElderGnosis_Init(predictor_obj, elder_count):
    predictor_obj.elders_data = {i: {'gnosis': 0.0} for i in range(elder_count + 1)}
    predictor_obj.accuracy = 0.0
    predictor_obj.last_prediction_score = 0.5 # Initialize last prediction score
    # Raw_Print("  [ElderGnosis] Predictor initialized for {} elders.", elder_count) # Commented for efficiency

def ElderGnosis_AddElder(predictor_obj, id, gnosis_factor):
    predictor_obj.elders_data[id]['gnosis'] = gnosis_factor

def ElderGnosis_PredictRisk(predictor_obj, page_idx_for_seed, anomaly_type): # Changed to take page_idx for seed
    # Enhanced prediction model tuning:
    # Adding symbolic phase harmonics, entity emotional state modifiers, entropy-weighted bonding curves.
    symbolic_phase_harmonic = 0.5 + 0.5 * SinF(cycle_num * 0.0001 + page_idx_for_seed * 0.00001)
    # Conceptual entity emotional state modifier (using elder temperament)
    entity_emotional_modifier = elders[0].t * 0.2 + (QuantumRand(cycle_num + page_idx_for_seed) * 0.1 - 0.05)
    entropy_weight = 1.0 - abs(voidEntropy) # Higher void entropy means lower weight

    base_prediction = QuantumRand(cycle_num + page_idx_for_seed + anomaly_type) * 0.7 + predictor_obj.accuracy * 0.3

    # Combine factors, clamping to [0, 1]
    tuned_prediction = base_prediction * symbolic_phase_harmonic * (1.0 + entity_emotional_modifier) * entropy_weight

    # Apply exponential smoothing for prediction score decay
    alpha = 0.6 # Smoothing factor (0.6 for fast feedback)
    smoothed_prediction = alpha * tuned_prediction + (1 - alpha) * predictor_obj.last_prediction_score
    predictor_obj.last_prediction_score = smoothed_prediction # Update last prediction score

    return MaxF(0.0, MinF(1.0, smoothed_prediction))


def ElderGnosis_GetAccuracy(predictor_obj, elder_id):
    return predictor_obj.accuracy + QuantumRand(elder_id) * 0.1

def ElderGnosis_UpdateModel(predictor_obj, avg_gnosis, void_entropy_val, anomaly_diversity, learning_rate_boost=1.0): # Added learning_rate_boost
    # Incorporate anomaly diversity into model update (conceptual)
    diversity_factor = 1.0 + (sum(anomaly_diversity.values()) / len(ANOMALY_TYPES)) * 0.01 # Simple diversity metric
    predictor_obj.accuracy = MaxF(0.0, avg_gnosis * (1.0 - abs(void_entropy_val)) * 0.9 * diversity_factor * learning_rate_boost + QuantumRand(cycle_num) * 0.1)

def ArchonSocieties_FormMetaOmniverse(om_mask):
    return random.getrandbits(64) # Return a conceptual ID

_global_archon_count = 0
_meta_omniverse_cohesions = {} # Stores conceptual cohesion for meta-omniverses

def ArchonSocieties_UpdateMetaDynamics(meta_omni_id, cohesion, nw):
    _meta_omniverse_cohesions[meta_omni_id] = cohesion * (1.0 + nw / 255.0 * 0.1)

def ArchonSocieties_GetDesiredMetaCohesion(meta_omni_id):
    return _meta_omniverse_cohesions.get(meta_omni_id, 0.5)

def ArchonSocieties_GetGlobalCount():
    return len(_meta_omniverse_cohesions)

def ArchonSocieties_AdjustCohesion(delta):
    pass # Conceptual adjustment

def ArchonSocieties_SpawnCelestialArchons(count):
    global _global_archon_count
    _global_archon_count += count
    # Raw_Print("  [ARCHONS] {} Celestial Archons spawned. Total: {}", count, _global_archon_count) # Commented for efficiency

def Physics_GlobalGaugeAlignment(): return QuantumRand(cycle_num) * 0.9
def Physics_GlobalPhaseAlignment(): return QuantumRand(cycle_num + 1) * 0.8
def Physics_GlobalSentienceAlignment(): return QuantumRand(cycle_num + 2) * 0.7


def Tesseract_AlignAddress(tesseract_obj, seed, string_obj):
    return int(QuantumRand(seed) * tesseract_obj.size) % (MAX_QPAGES * PAGE_SIZE)

def Tesseract_Tunnel(tesseract_obj, source_idx, target_node, string_obj):
    # Raw_Print("  [Tesseract] Tunneling from {} to {} via {}.", source_idx, target_node, string_obj.energy_density) # Commented for efficiency
    tesseract_obj.active_nodes = MaxF(tesseract_obj.active_nodes, source_idx) + 1 # Update active nodes conceptually
    return QuantumRand(source_idx + target_node) * string_obj.energy_density * 1e-20 # Conceptual gain

def Tesseract_Synchronize(tesseract_obj, idx):
    # Raw_Print("  [Tesseract] Synchronizing index {}.", idx) # Commented for efficiency
    pass

def Tesseract_SynchronizeAll(tesseract_obj):
    # Raw_Print("  [Tesseract] Synchronizing all nodes.") # Commented for efficiency
    pass # No intensive operations for mock function

def Tesseract_GetActiveNodes(tesseract_obj):
    # This is highly conceptual, as we don't actually track individual tesseract nodes
    return int(tesseract_obj.active_nodes + QuantumRand(cycle_num) * 100) % 1000 # Return a conceptual number

def CosmicString_UpdateTension(strings_list, count, entropy):
    for s in strings_list:
        s.energy_density = MaxF(1e16, s.energy_density + (QuantumRand(cycle_num) - 0.5) * 1e15 * (1.0 - abs(entropy)))
    # Raw_Print(f"  [CosmicStrings] Updated tension for {count} strings.") # Commented for efficiency

def Mandelbulb_TransformString(string_obj, mb_params_obj, cycle):
    # Conceptual transformation, affecting Mandelbulb parameters based on string properties
    mb_params_obj.rotation['x'] += string_obj.torsion * 0.001 * math.sin(cycle * 0.001)
    mb_params_obj.rotation['y'] += string_obj.torsion * 0.001 * math.cos(cycle * 0.001)
    mb_params_obj.color_shift = (mb_params_obj.color_shift + string_obj.energy_density * 1e-25) % 1.0

def Mandelbulb_Render(dc_obj, mb_params_obj, cycle, void_entropy_val):
    # This will be handled by the Pygame rendering logic.
    # It will draw an abstract representation.
    pass

class CDC:
    """Conceptual Display Context. Pygame screen will serve this role."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Raw_Print("  [CDC] Initialized conceptual display context {}x{}.", width, height) # Commented for efficiency

def DCNewMnemonic(width, height):
    return CDC(width, height)

def DCDelete(dc_obj):
    # Raw_Print("  [CDC] Deleted conceptual display context {}x{}.", dc_obj.width, dc_obj.height) # Commented for efficiency
    pass

def Graphics_Clear(dc_obj, color):
    # Pygame equivalent will clear the screen
    pass

def Graphics_DrawRect(dc_obj, x, y, w, h, color):
    # Pygame equivalent will draw a rectangle
    pass

def Graphics_Text(dc_obj, x, y, text, color):
    # Pygame equivalent will render text
    pass

def Graphics_TextF(dc_obj, x, y, color, fmt_string, *args):
    # Pygame equivalent will render formatted text
    pass

def Graphics_ProjectDimension(dc_obj, dim, dim_scale, elder_influence):
    # Conceptual, will be abstracted in Pygame visualization
    pass

def Graphics_RenderCosmicString(dc_obj, string_obj):
    # Conceptual, will be abstracted in Pygame visualization
    pass

def Graphics_Refresh():
    # Pygame equivalent will update the display
    pass

def QuantumSleep(milliseconds):
    """Conceptual sleep function."""
    time.sleep(milliseconds / 1000.0)

def MapPages(address, count, flags):
    """Conceptual memory page mapping."""
    global quantumHeapPages
    global MAX_QPAGES
    if quantumHeapPages + count <= MAX_QPAGES:
        # For conceptual purposes, we just increment the page count
        # In a real system, this would involve OS memory allocation.
        return True
    return False

def UnmapPages(address, count):
    """Conceptual memory page unmapping."""
    global collapsedHeapPages
    # For conceptual purposes, we just increment collapsed pages
    collapsedHeapPages += count
    return True

# --- Initialization Functions ---

def InitZetaCache():
    global zeta_cache
    for i in range(21):
        zeta_cache[i] = RiemannZeta(1.0 + i * 0.1)
    # Raw_Print("  [Init] Zeta Cache initialized.") # Commented for efficiency

def InitQuantumFoam(node):
    node.foam = QuantumFoam()
    for i in range(PLANCK_FOAM_SIZE):
        node.foam.virtual_particles[i]['energy'] = QuantumRand(cycle_num + i) * 1.5
        node.foam.virtual_particles[i]['lifetime'] = QuantumRand(cycle_num + i + 0xCAFE) * 8.0
    for d in range(TIME_DIMS):
        node.chrono_phase[d] = QuantumRand(cycle_num + d)
    # Raw_Print("  [Init] Quantum Foam initialized for node on page {}.", node.page_index) # Commented for efficiency

def InitCosmicStrings():
    global cosmic_strings # Declare global to ensure we're modifying the global variable
    for i in range(COSMIC_STRINGS_COUNT):
        # Ensure cosmic_strings is initialized before accessing its elements
        if cosmic_strings is None:
            pass # Should not be reached due to explicit initialization order
        cosmic_strings[i].energy_density = 1e16 + QuantumRand(i) * 1e20
        cosmic_strings[i].torsion = QuantumRand(i + 0xBEEF) * math.pi * 2
        cosmic_strings[i].endpoints[0] = int(QuantumRand(i) * HYPERGRID_SIZE)
        cosmic_strings[i].endpoints[1] = int(QuantumRand(i + 0xDEAD) * HYPERGRID_SIZE)
    # Raw_Print("  [Init] Cosmic Strings initialized.") # Commented for efficiency

def InitMandelbulb():
    global mb_params
    mb_params.scale = 1.8
    mb_params.max_iterations = 12
    mb_params.bailout = 4.0
    mb_params.power = 8.0 + QuantumRand(0) * 3.0
    mb_params.rotation['x'] = QuantumRand(1) * 6.283
    mb_params.rotation['y'] = QuantumRand(2) * 6.283
    mb_params.rotation['z'] = QuantumRand(3) * 6.283
    mb_params.color_shift = QuantumRand(4) * 0.5
    # Raw_Print("  [Init] Mandelbulb parameters initialized.") # Commented for efficiency

def InitElderGnosis():
    global predictor
    ElderGnosis_Init(predictor, ELDER_COUNT)
    for e in range(ELDER_COUNT):
        elders[e].gnosis_factor = QuantumRand(e) * 0.5 + 0.5
        ElderGnosis_AddElder(predictor, e, elders[e].gnosis_factor)
    # Raw_Print("  [Init] Elder Gnosis initialized.") # Commented for efficiency

def InitTesseract():
    global tesseract, cycle_num # Ensure we can modify global tesseract and access cycle_num
    # Instantiate tesseract here, now that QuantumRand and cycle_num are available
    tesseract = TesseractState(HYPERGRID_SIZE) # TesseractState now has a default phase_lock=0

    # Now that tesseract is instantiated, set its phase_lock using QuantumRand and cycle_num
    tesseract.phase_lock = int(QuantumRand(cycle_num) * 0xFFFFFFFF)

    # Raw_Print("  [Init] Tesseract initialized with size {}.", HYPERGRID_SIZE) # Commented for efficiency

def InitAnimation():
    global animation_frames
    for i in range(100):
        animation_frames[i].frame_time = cycle_num + i * 10000
        animation_frames[i].rotation['x'] = QuantumRand(i) * 6.283
        animation_frames[i].rotation['y'] = QuantumRand(i + 1) * 6.283
        animation_frames[i].rotation['z'] = QuantumRand(i + 2) * 6.283

def alloc_node(depth, page_index): # Added page_index
    """Conceptual allocation of an OctNode."""
    global nodeAllocCount
    nodeAllocCount += 1
    return OctNode(depth, page_index) # Pass page_index

def build_tree(node, depth, page_index): # Added page_index
    """Conceptual recursive building of the octree."""
    if depth > 0:
        for i in range(8):
            node.c[i] = alloc_node(depth - 1, page_index) # Pass page_index
            build_tree(node.c[i], depth - 1, page_index) # Pass page_index
            # Assign conceptual mass based on depth
            node.c[i].mass = int(HYPERGRID_SIZE / (8**(OCTREE_DEPTH - (OCTREE_DEPTH - depth + 1))) * ACTIVE_RATIO)
    # Give the root a conceptual initial mass
    if depth == OCTREE_DEPTH:
        node.mass = int(HYPERGRID_SIZE * ACTIVE_RATIO)

def init_elders():
    global elders
    for e in range(ELDER_COUNT + 1):
        elders[e].id = e
        elders[e].t = 0.008 * (1.0 + QuantumRand(e) * 0.12)
        elders[e].b = QuantumRand(e + 0xDEADCAF) * 0.25 - 0.125
        elders[e].qft_feedback = 0.0
        elders[e].chem_feedback = 0.0
        elders[e].social_feedback = 0.0
        elders[e].gnosis_factor = QuantumRand(e) * 0.5 + 0.5
    elders[0].id = 0
    elders[ELDER_COUNT].id = 777778 # CHRONOS OVER-ELDER
    # Raw_Print("  [Init] Elders initialized.") # Commented for efficiency

# --- Heap Management ---

def FractalRatio():
    s = 1.2 + voidEntropy * 0.25
    idx = int((s - 1.0) / 0.1)
    if not (0 <= idx < len(zeta_cache)):
        idx = 0 # Fallback
    return 1.0 + 85.0 * (1.0 - voidEntropy) * zeta_cache[idx] * 0.70

def ElderSanctionAlloc(pages):
    total_feedback = 0
    # Summing over a subset of elders for conceptual feedback
    for i in range(0, ELDER_COUNT, 400):
        total_feedback += elders[i].qft_feedback + elders[i].chem_feedback + elders[i].social_feedback
    total_feedback /= (ELDER_COUNT // 400) if (ELDER_COUNT // 400) > 0 else 1 # Avoid division by zero

    approval = elders[0].t * (1.0 - voidEntropy) * (1.61803398875 - float(pages) / MAX_QPAGES) * (1.0 + total_feedback)
    return QuantumRand(cycle_num) < approval * user_divinity * 0.999

def calculate_dynamic_severity(anomaly_type, prediction_score, current_cycle, node):
    """
    Calculates dynamic severity for an anomaly.
    Severity based on:
    - Base anomaly type
    - Time since last successful SIGIL_FIX for this anomaly type
    - Node stability
    - Sigil entropy level on target page (conceptual: represented by sigil_used_count - higher means more entropic)
    - Symbolic drift (higher drift means higher severity)
    - Bond density (higher bond density can increase severity, especially for bonding anomalies)
    - Node's emotional state / symbolic focus
    """
    base_severity = 0.5 # Default for most anomalies
    if anomaly_type == 0: base_severity = 0.8 # Entropy - high base
    elif anomaly_type == 1: base_severity = 0.7 # Stability drop - high base
    elif anomaly_type == 2: base_severity = 0.9 # Void decay - very high base
    elif anomaly_type == 3: base_severity = 0.4 # Tunnel (can be good/bad) - lower base
    elif anomaly_type == 4: base_severity = 0.6 # Overbonding - moderate base

    # Factor for prediction score: Higher prediction implies well-understood threat, so severity remains high.
    # Adjusted to make it more impactful for understood threats
    severity_from_prediction = base_severity * (0.5 + prediction_score * 0.5)

    # Factor for time since last fix: Longer time means more accumulated risk
    anomaly_origin_cycle = node.last_fixed_anomaly_cycle.get(anomaly_type, 0) # Use .get for robustness
    time_since_fix_factor = 1.0 + MaxF(0.0, (current_cycle - anomaly_origin_cycle) / 500.0) # Scaled by 500 cycles for more impact
    severity_from_time = severity_from_prediction * MinF(2.5, time_since_fix_factor) # Max 2.5x severity from time

    # Factor for node stability: Lower stability means higher severity
    stability_factor = 1.0 + (1.0 - node.stabilityPct) * 0.8 # If stability 0%, factor is 1.8x (more impactful)
    final_severity = severity_from_time * stability_factor

    current_sigil_str = "".join(user_sigil).strip('\0')
    sigil_used_count = node.sigil_mutation_history.get(current_sigil_str, 0) # Use .get for robustness
    # Factor for sigil usage (conceptual entropy on page): More usage means higher "entropic" severity
    sigil_entropy_factor = 1.0 + MinF(1.5, float(sigil_used_count) / SYMBOLIC_RECOMBINATION_THRESHOLD) # Caps at 2.5x severity
    final_severity *= sigil_entropy_factor

    # Factor for symbolic drift: Higher drift implies system is further from equilibrium
    symbolic_drift_factor = 1.0 + node.symbolic_drift * 0.7 # If drift 1.0, factor is 1.7x
    final_severity *= symbolic_drift_factor

    # Factor for bond density: Especially for bonding anomalies
    if anomaly_type == 4: # Bonding
        bond_density_factor = 1.0 + node.bond_strength * 0.5 # Higher bond strength, higher severity
        final_severity *= bond_density_factor

    # Conceptual implementation of (bond_density^2 × recursion_depth) / (sigil_entropy + symbolic_alignment)
    # Using sigil_entropy_metric as symbolic_alignment for simplicity here
    sigil_current_entropy = calculate_sigil_entropy(current_sigil_str)
    # Ensure denominator is not zero or too small
    denominator = sigil_current_entropy + (node.social_cohesion if node.social_cohesion > 0.001 else 0.001)

    conceptual_complexity_factor = (node.bond_strength**2 * (float(OCTREE_DEPTH) / 5.0)) / denominator
    final_severity *= (1.0 + MinF(1.0, conceptual_complexity_factor * 0.1)) # Small multiplier to avoid explosion

    # New: Influence of Archetype's Emotional State and Symbolic Focus on Severity
    emotional_impact = 1.0
    if node.emotional_state == "resolute": emotional_impact = 1.1 # Warrior makes things more critical
    elif node.emotional_state == "curious": emotional_impact = 0.9 # Witch might dampen severity if exploring
    elif node.emotional_state == "contemplative": emotional_impact = 1.05 # Mystic might slightly amplify for deep analysis
    elif node.emotional_state == "guiding": emotional_impact = 0.95 # Quest Giver tries to minimize
    final_severity *= emotional_impact

    # Symbolic focus impact (conceptual): if an anomaly aligns with focus, higher severity if not handled well
    if (node.symbolic_focus == "tunneling" and anomaly_type == 3) or \
       (node.symbolic_focus == "bonding" and anomaly_type == 4) or \
       (node.symbolic_focus == "entropy" and anomaly_type == 0):
        final_severity *= 1.15 # 15% more critical if it's their focus

    return MinF(1.0, final_severity) # Clamp max severity to 1.0

def trigger_anomaly_if_cooldown_clear(anomaly_type, page_idx, severity, prediction_score, details, sub_type_tag, node_instance):
    """
    Helper function to trigger anomalies only if their cooldown period has passed for the given node and type.
    This prevents redundant logging and processing of the same conceptual anomaly type from the same source.
    `node_instance` is the actual OctNode object (n in calling functions).
    """
    global total_anomalies_triggered, anomaly_type_counts, newly_triggered_anomalies_queue, anomaly_type_counts_per_page, detailed_anomaly_log_file

    if cycle_num - node_instance.last_triggered_anomaly_cycle.get(anomaly_type, 0) < ANOMALY_TRIGGER_COOLDOWN:
        return False # Cooldown not over, do not trigger

    new_anomaly = Anomaly(
        cycle_num, page_idx, anomaly_type, severity, prediction_score,
        details=details,
        sub_type_tag=sub_type_tag
    )
    anomalies_per_page[page_idx][anomaly_count_per_page[page_idx] % ANOMALY_HISTORY] = new_anomaly
    anomaly_count_per_page[page_idx] += 1
    newly_triggered_anomalies_queue.append(new_anomaly)
    total_anomalies_triggered += 1
    anomaly_type_counts[anomaly_type] += 1
    anomaly_type_counts_per_page[page_idx][anomaly_type] += 1 # Track per-page anomaly count
    node_instance.last_triggered_anomaly_cycle[anomaly_type] = cycle_num # Update last triggered cycle

    # Log to new detailed anomaly log file regardless of fix attempts
    if detailed_anomaly_log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        detailed_anomaly_log_file.write(f"[{timestamp}] ANOMALY_TRIGGERED on Page {page_idx}: {ANOMALY_TYPES.get(anomaly_type, 'Unknown Type')}\n")
        detailed_anomaly_log_file.write(f"  Cycle: {cycle_num}\n")
        detailed_anomaly_log_file.write(f"  Page Index: {page_idx}\n")
        detailed_anomaly_log_file.write(f"  Anomaly Type: {anomaly_type} ({ANOMALY_TYPES.get(anomaly_type, 'Unknown')}) {sub_type_tag}\n")
        detailed_anomaly_log_file.write(f"  Severity: {severity:.4f}\n")
        detailed_anomaly_log_file.write(f"  Prediction Score: {prediction_score:.4f}\n")
        detailed_anomaly_log_file.write(f"  Details: {details}\n\n")

    return True

def QuantumExpandHeap(requested, p_idx): # Added p_idx
    global quantumHeapPages, voidEntropy
    mapped = 0
    eigenSeed = cycle_num ^ int(QuantumRand(cycle_num) * 1e12)
    node = roots[p_idx] # Get the specific node for this page
    foam_density = QuantumFoam_Stabilize(node.foam, voidEntropy) if node and node.foam else 0.5 # Default if root/foam not ready

    for i in range(0, requested, 32):
        if quantumHeapPages + 32 > MAX_QPAGES:
            break

        zeta = zeta_cache[int(QuantumRand(eigenSeed + i) * 20) % len(zeta_cache)]

        predicted_risk = ElderGnosis_PredictRisk(predictor, p_idx, 3) # Prediction for Tunnel anomaly

        if not ElderSanctionAlloc(32):
            # Dynamic Severity for denied allocation (Type 3: Tunnel anomaly for denied path)
            dyn_severity = calculate_dynamic_severity(
                3, predicted_risk, cycle_num, node # Pass the specific node
            )
            sub_type_tag = "[HEAP_DENIED]"
            if trigger_anomaly_if_cooldown_clear(3, p_idx, dyn_severity, predicted_risk, f"Heap allocation denied on page {p_idx}. Risk: {predicted_risk:.2f}", sub_type_tag, node):
                Raw_Print("> [DENIED] Elder {:04d} rejects pages {} on Page {} (Severity: {:.2f})", elders[0].id, quantumHeapPages, p_idx, dyn_severity)
                voidEntropy = MinF(voidEntropy + 0.001, 0.0)
            continue

        alloc_addr = Tesseract_AlignAddress(tesseract, cycle_num + i, cosmic_strings[i % COSMIC_STRINGS_COUNT])
        if MapPages(alloc_addr, 32, 3):
            for j in range(32):
                if len(pageEigenstates) <= quantumHeapPages + j: # Ensure list can hold new elements
                    pageEigenstates.extend([0] * (quantumHeapPages + j - len(pageEigenstates) + 1))
                pageEigenstates[quantumHeapPages + j] = int(QuantumRand(eigenSeed + i + j) * 4)
            quantumHeapPages += 32
            mapped += 32

            # Diversify Anomaly Trigger: Tunnel Anomaly (Type 3) for successful mapping
            if foam_density > 0.75 and QuantumRand(mapped) < 0.08 * user_divinity: # Increased chance from 0.05
                predicted_risk_tunnel = ElderGnosis_PredictRisk(predictor, p_idx, 3)
                dyn_severity_tunnel = calculate_dynamic_severity(
                    3, predicted_risk_tunnel, cycle_num, node # Pass the specific node
                )
                sub_type_tag = "[HEAP_TUNNEL_SUCCESS]"
                if trigger_anomaly_if_cooldown_clear(3, p_idx, dyn_severity_tunnel, predicted_risk_tunnel, f"Successful heap map triggered conceptual tunnel on page {p_idx} due to high foam density ({foam_density:.2f}).", sub_type_tag, node):
                    string_idx_for_tunnel = (quantumHeapPages - 32) % 2 == 0 # Use the starting page index for parity
                    Raw_Print("> [TUNNEL] Pages {}-{} on Page {} via String[{}] (Severity: {:.2f})", quantumHeapPages-32, quantumHeapPages, p_idx, 3 if string_idx_for_tunnel else 8, dyn_severity_tunnel)
    return mapped

def VoidDecay(p_idx): # Added p_idx
    global voidEntropy, collapsedHeapPages
    node = roots[p_idx] # Get the specific node for this page

    if voidEntropy < VOID_THRESHOLD:
        return

    # Iterate over a conceptual subset of pages for performance
    for i in range(0, int(quantumHeapPages), 64):
        decay_rate = -voidEntropy * 0.20
        # Diversify Anomaly Trigger: Void Anomaly (Type 2)
        if len(pageEigenstates) > i and pageEigenstates[i] > 64 and QuantumRand(cycle_num + i) < decay_rate:
            predicted_risk = ElderGnosis_PredictRisk(predictor, p_idx, 2)
            dyn_severity = calculate_dynamic_severity(
                2, predicted_risk, cycle_num, node # Pass the specific node
            )
            UnmapPages(0x1000000 + i * PAGE_SIZE, 1) # Conceptual unmapping
            pageEigenstates[i] = 255 # Mark as decayed
            collapsedHeapPages += 1
            sub_type_tag = "[VOID_DISSOLUTION]"
            if trigger_anomaly_if_cooldown_clear(2, p_idx, dyn_severity, predicted_risk, f"Page {i} dissolved on page {p_idx} due to void decay. Decay Rate: {decay_rate:.4f}.", sub_type_tag, node):
                Raw_Print("> [VOID] Page {} dissolved on Page {} (Entropy: {:.2f}, Severity: {:.2f})", i, p_idx, -voidEntropy, dyn_severity)
    voidEntropy = MaxF(VOID_THRESHOLD, -voidEntropy - 0.0002 * (elders[0].social_feedback + elders[0].chem_feedback))

# --- Anomaly Handler ---

def HandleAnomaly(a): # Takes an Anomaly object directly
    global conduit_stab, voidEntropy, user_divinity, user_sigil, anomaly_log_file, total_anomalies_fixed, fixed_anomalies_log, quantumHeapPages # Added quantumHeapPages here
    global anomaly_type_counts # Ensure global access for metric

    # Create a unique identifier for this anomaly instance
    # Including sub_type_tag for more unique instance identification
    anomaly_id = (a.cycle, a.page_idx, a.anomaly_type, a.details, a.sub_type_tag)
    if anomaly_id in fixed_anomalies_log:
        # This anomaly has already been processed and "fixed" in this session, skip.
        # This addresses the "Recursive Overlap" issue by ensuring a conceptual fix is processed only once.
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # Millisecond precision
    node = roots[a.page_idx] # Get the specific node for this anomaly's page

    # Log anomaly details to the file (always log this part for any anomaly)
    if anomaly_log_file:
        anomaly_log_file.write(f"[{timestamp}] Anomaly Handled on Page {a.page_idx}: {ANOMALY_TYPES.get(a.anomaly_type, 'Unknown Type')}\n")
        anomaly_log_file.write(f"  Cycle: {a.cycle}\n")
        anomaly_log_file.write(f"  Page Index: {a.page_idx}\n")
        anomaly_log_file.write(f"  Anomaly Type: {a.anomaly_type} ({ANOMALY_TYPES.get(a.anomaly_type, 'Unknown')}) {a.sub_type_tag}\n") # Include sub_type_tag
        anomaly_log_file.write(f"  Severity: {a.severity:.4f}\n")
        anomaly_log_file.write(f"  Prediction Score: {a.prediction_score:.4f}\n")
        anomaly_log_file.write(f"  Details: {a.details}\n")

    log_message = ""
    action_taken = ""
    is_fixed = False

    sigil_effectiveness_multiplier = 1.0
    current_sigil_str = "".join(user_sigil).strip('\0')

    # Check for sigil reapplication cost / mutation
    node.sigil_mutation_history[current_sigil_str] += 1 # Track per-node/page sigil use
    sigil_reapplication_count = node.sigil_mutation_history[current_sigil_str]

    if sigil_reapplication_count > SYMBOLIC_RECOMBINATION_THRESHOLD:
        sigil_effectiveness_multiplier -= (sigil_reapplication_count - SYMBOLIC_RECOMBINATION_THRESHOLD) * SIGIL_REAPPLICATION_COST_FACTOR
        sigil_effectiveness_multiplier = MaxF(0.0, sigil_effectiveness_multiplier) # Cannot go below 0
        if sigil_effectiveness_multiplier < 0.5: # Force a new sigil if it's too ineffective
            Raw_Print("!!! Sigil '{}' too saturated on Page {}! Forcing mutation!", current_sigil_str, a.page_idx)
            craft_sigil(force_mutation=True)
            # When sigil mutates, the old sigil's count for this specific node/page should be reset
            # to prevent immediate re-saturation with the new sigil.
            node.sigil_mutation_history[current_sigil_str] = 0 # Reset old sigil's count for this node
            current_sigil_str = "".join(user_sigil).strip('\0') # Update after mutation
            node.sigil_mutation_history[current_sigil_str] = 1 # Start new sigil's count for this node at 1
            action_taken += "Forced Sigil Mutation. "

    # Implement "Time-Weighted Sigil Priority": Delay action if fixed recently on this node/page
    # The `trigger_anomaly_if_cooldown_clear` already prevents duplicate *triggering*,
    # this part is for delaying *handling* if a fix was recently applied to the same type.
    if cycle_num - node.last_fixed_anomaly_cycle.get(a.anomaly_type, 0) < ANOMALY_TRIGGER_COOLDOWN: # Use same cooldown for fixing
        Raw_Print(f"> [DELAYED_SIGIL_FIX] Anomaly Type {ANOMALY_TYPES.get(a.anomaly_type)} on Page {a.page_idx} recently fixed. Delaying action.")
        if anomaly_log_file:
            anomaly_log_file.write(f"  Action Taken: DELAYED (recently fixed on this page)\n\n")
        return # Skip immediate fix to improve efficiency

    # Adaptive SIGIL Resolution based on prediction_score and Archetype Emotional State
    action_type = ""
    # Archetype influence on action choice
    if node.archetype_name == "Android / Warrior" and node.emotional_state == "resolute":
        if a.severity > 0.7: action_type = "PURGE" # Warriors are decisive for high severity
        else: action_type = "MERGE"
    elif node.archetype_name == "Witch / Mirror" and node.emotional_state == "curious":
        if a.prediction_score < 0.3: action_type = "REFLECT" # Witches might reflect unknown
        else: action_type = "FIX"
    elif node.archetype_name == "Mystic" and node.emotional_state == "contemplative":
        if a.anomaly_type == 0: action_type = "MERGE" # Mystic focuses on reintegrating entropy
        else: action_type = "FIX"
    elif node.archetype_name == "Quest Giver" and node.emotional_state == "guiding":
        if a.severity > 0.8: action_type = "FIX" # Quest Givers prioritize stable solutions
        else: action_type = "REFLECT"
    else: # Default behavior based on prediction score
        if a.prediction_score > 0.8: action_type = "MERGE"
        elif a.prediction_score > 0.6: action_type = "FIX"
        elif a.prediction_score > 0.4: action_type = "REFLECT"
        else: action_type = "PURGE"


    if action_type == "MERGE":
        # Apply stronger fixes / more complex interventions
        if a.anomaly_type == 0: node.st.e = MaxB(0, node.st.e - int(90 * sigil_effectiveness_multiplier))
        elif a.anomaly_type == 1: conduit_stab = MinF(conduit_stab + 0.10 * sigil_effectiveness_multiplier, 1.0)
        elif a.anomaly_type == 2: voidEntropy = MaxF(voidEntropy - 0.02 * sigil_effectiveness_multiplier, VOID_THRESHOLD)
        elif a.anomaly_type == 3: Tesseract_Synchronize(tesseract, a.page_idx); Tesseract_SynchronizeAll(tesseract)
        elif a.anomaly_type == 4: ArchonSocieties_AdjustCohesion(-0.20 * sigil_effectiveness_multiplier); node.bond_strength = MaxF(0.0, node.bond_strength - (0.20 * sigil_effectiveness_multiplier))
        is_fixed = True
    elif action_type == "FIX":
        # Apply standard fixes
        if a.anomaly_type == 0: node.st.e = MaxB(0, node.st.e - int(60 * sigil_effectiveness_multiplier))
        elif a.anomaly_type == 1: conduit_stab = MinF(conduit_stab + 0.06 * sigil_effectiveness_multiplier, 1.0)
        elif a.anomaly_type == 2: voidEntropy = MaxF(voidEntropy - 0.012 * sigil_effectiveness_multiplier, VOID_THRESHOLD)
        elif a.anomaly_type == 3: Tesseract_Synchronize(tesseract, a.page_idx)
        elif a.anomaly_type == 4: ArchonSocieties_AdjustCohesion(-0.12 * sigil_effectiveness_multiplier); node.bond_strength = MaxF(0.0, node.bond_strength - (0.12 * sigil_effectiveness_multiplier))
        is_fixed = True
    elif action_type == "REFLECT":
        # Apply fixes that "bounce back" effects (e.g., convert severity to stability)
        if a.anomaly_type == 0: node.st.s = MinB(node.st.s + int(a.severity * 255 * 0.2), 255) # Convert entropy to stability
        elif a.anomaly_type == 1: user_divinity = MinF(user_divinity + a.severity * 0.1, 1.0) # Stability reflected as divinity
        elif a.anomaly_type == 2: conduit_stab = MinF(conduit_stab + a.severity * 0.05, 1.0) # Void reflected as conduit stability
        elif a.anomaly_type == 3: pass # Reflect for tunnel might mean redirecting it (no direct sim effect here)
        elif a.anomaly_type == 4: node.social_cohesion = MinF(node.social_cohesion + a.severity * 0.1, 1.0) # Bonding reflected as social cohesion
        is_fixed = True # Still considered "fixed" conceptually
    elif action_type == "PURGE":
        # Apply drastic fixes, potential side effects
        if a.anomaly_type == 0: # Entropy Purge: Qubit Cold-Stitcher concept
            node.mass = MaxF(0, node.mass - int(HYPERGRID_SIZE * 0.01)) # Reduce mass drastically
            node.st.s = MinB(node.st.s + int(a.severity * 255 * 0.4), 255) # Stronger stability boost
            Raw_Print("> [Qubit Cold-Stitcher] Targeted state-space restoration on Page {}!", a.page_idx)
            # Log entropy hit conceptually by marking a unique OM
            # (Note: actual OM tracking and suppression would be a separate, complex ledger)
        elif a.anomaly_type == 1: voidEntropy = MinF(voidEntropy + 0.03, 0.0) # Stability purge increases void risk
        elif a.anomaly_type == 2: conduit_stab = MaxF(0.0, conduit_stab - 0.1) # Void purge hits conduit stability
        elif a.anomaly_type == 3: quantumHeapPages = MaxF(0, quantumHeapPages - 50) # Purge tunnel reduces heap
        elif a.anomaly_type == 4: # Bonding Purge: Sigil Flux Decay + Jitter Masking
            node.bond_strength = 0.0 # Drastically remove bonding
            node.social_cohesion = MaxF(0.0, node.social_cohesion * 0.88) # Decay cohesion
            craft_sigil(force_mutation=True) # Sigil-Jitter Masking (force mutation)
            Raw_Print("> [Sigil-Jitter Masking] Symbolic variance injected on Page {}!", a.page_idx)
        is_fixed = True # Still considered "fixed" conceptually


    log_message = f"> [{action_type} SIGIL] Anomaly Type {ANOMALY_TYPES.get(a.anomaly_type)} at page {a.page_idx}, cycle {a.cycle} (Agent: {node.archetype_name})"
    Raw_Print(log_message)
    if anomaly_log_file:
        action_taken += log_message + f" (Sigil Eff.: {sigil_effectiveness_multiplier:.2f})"
        anomaly_log_file.write(f"  Action Taken: {action_taken}\n\n")

    if is_fixed:
        total_anomalies_fixed += 1
        node.last_fixed_anomaly_cycle[a.anomaly_type] = cycle_num # Update last fixed cycle for this node/page
        fixed_anomalies_log.add(anomaly_id) # Mark this specific anomaly instance as processed
        # Update conceptual recursive page signature
        node.anomaly_history_signature = (node.anomaly_history_signature + a.anomaly_type + cycle_num) % 0xFFFFFFFF # Simple update

# --- Quantum Functions ---

def temporal_synchro_tunnel(n, p_idx): # Added p_idx
    predicted_risk = ElderGnosis_PredictRisk(predictor, p_idx, 3) # Prediction for Tunnel anomaly
    dyn_severity = calculate_dynamic_severity(
        3, predicted_risk, cycle_num, n # Pass the specific node
    )
    # Implement dynamic gain-based thresholding for tunnel ops
    tunnel_gain_base = Tesseract_Tunnel(tesseract, n.st.tesseract_idx, int(QuantumRand(cycle_num) * nodeAllocCount), cosmic_strings[n.st.tesseract_idx % COSMIC_STRINGS_COUNT])

    if tunnel_gain_base < 0.01: # Skip if gain is too low
        n.delayed_tunnel_count += 1 # Increment delayed count
        # Raw_Print(f"> [TUNNEL_SKIPPED] Tunnel gain too low ({tunnel_gain_base:.4f}) on Page {p_idx}. Delayed count: {n.delayed_tunnel_count}.")
        return

    # Conceptual rate limiter:
    # If the last tunnel anomaly was very recent and severe, slightly reduce chance
    time_since_last_tunnel = cycle_num - n.last_triggered_anomaly_cycle.get(3, 0)
    trigger_chance_modifier = 1.0
    if time_since_last_tunnel < ANOMALY_TRIGGER_COOLDOWN * 2 and dyn_severity > 0.6:
        trigger_chance_modifier = 0.5 # Halve chance if recent and severe

    if QuantumRand(cycle_num + n.st.tesseract_idx) < 0.0015 * trigger_chance_modifier:
        # Enforce multi-page dispersion: if tunnel ops get stuck, try another page (conceptual)
        target_page_for_influence = p_idx
        if time_since_last_tunnel < ANOMALY_TRIGGER_COOLDOWN and random.random() < 0.3: # 30% chance to influence another page if stuck
            target_page_for_influence = (p_idx + random.randint(1, PAGE_COUNT - 1)) % PAGE_COUNT
            Raw_Print(f"> [TUNNEL_DISPERSION] Tunnel on Page {p_idx} influencing Page {target_page_for_influence} due to local congestion.")

        string_to_use = cosmic_strings[3] if n.st.tesseract_idx % 2 == 0 else cosmic_strings[8]
        tunnel_gain = Tesseract_Tunnel(tesseract, n.st.tesseract_idx, int(QuantumRand(cycle_num) * nodeAllocCount), string_to_use)
        n.st.e = MinB(n.st.e + int(tunnel_gain * 255), 255)
        n.stabilityPct = MinF(n.stabilityPct + tunnel_gain * 0.1, 1.0)

        sub_type_tag = "[TS_TUNNEL_OP]"
        if trigger_anomaly_if_cooldown_clear(3, target_page_for_influence, dyn_severity, predicted_risk, f"Temporal synchro-tunnel operation on page {p_idx}. Gain: {tunnel_gain:.4f}.", sub_type_tag, n): # Corrected 'node' to 'n'
            if EVOLVED_SIGIL in "".join(user_sigil):
                n.st.tesseract_idx ^= int(tunnel_gain * 0xFFFFFFFF)
                Raw_Print("> [TEMPORAL SYNCHRO-TUNNEL] Node {:x} to {} on Page {} via String[{}] (Gain: {:.2f}, Severity: {:.2f})",
                          n.st.tesseract_idx, int(QuantumRand(cycle_num) * nodeAllocCount), p_idx, 3 if n.st.tesseract_idx % 2 == 0 else 8, tunnel_gain, dyn_severity)
            n.delayed_tunnel_count = 0 # Reset count on successful tunnel

def quantum_foam_stabilizer(n, p_idx): # Added p_idx
    foam_density = QuantumFoam_Stabilize(n.foam, voidEntropy) if n.foam else PLANCK_FOAM_DENSITY
    if foam_density > 0.75:
        n.st.e = MinB(n.st.e + int(foam_density * 255 * 0.12), 255)
        n.st.s = MinB(n.st.s + int(foam_density * 255 * 0.06), 255)
    if QuantumRand(cycle_num + n.st.e) < foam_density * 0.35:
        n.st.kappa ^= int(foam_density * 600000)
        n.st.mu = MinB(n.st.mu + 3, 255)
        if EVOLVED_SIGIL in "".join(user_sigil):
            n.stabilityPct = MinF(n.stabilityPct + 0.06, 1.0)

def cosmic_qft(n, p_idx): # Added p_idx
    noise = Physics_LIGOWave() + Physics_CMBFluct() + Physics_VIRGOWave() + Physics_SKYNETWave() + Physics_BosonicFieldDensity()
    qr = QuantumRand(cycle_num + n.st.e)
    z = zeta_cache[11]
    adj = MinF(242.0, noise * 40.0 * (1.0 + 0.95 * conduit_stab) * z)
    n.st.e = MinB(n.st.e + int(adj), 255)
    n.resonance = Physics_HarmonicResonance(n.st.e, n.st.d) * z * 0.65
    if qr < 0.10 * user_divinity:
        n.st.p = (n.st.p + 1) % SUPER_STATES
        n.st.zeta = int(z * 255 / zeta_cache[0]) if zeta_cache[0] != 0 else 0
    if qr < 0.18 * conduit_stab:
        n.st.ent = int(qr * 255)
        n.st.lambda_ = int(qr * 64)
    if qr < 0.04:
        n.st.flux = int(Physics_QCDFlux(n.st.ent) * 60 * 255)
        n.st.fractal ^= int(n.st.flux << (n.st.q % 16))

    # Entropy Sentinel Threads (Conceptual): Monitor sigil-tunnel coalescence to pre-trigger decay
    # Check if social_cohesion is high (sigil coalescence) and last tunnel was recent
    if n.social_cohesion > 0.95 and (cycle_num - n.last_triggered_anomaly_cycle.get(3, 0)) < ANOMALY_TRIGGER_COOLDOWN * 5:
        if QuantumRand(cycle_num) < 0.15: # Small chance to pre-emptively decay energy/stability
            n.st.e = MaxB(0, n.st.e - 10)
            n.st.s = MaxB(0, n.st.s - 5)
            # Raw_Print(f"> [Entropy Sentinel] Pre-emptively adjusted node on Page {p_idx} due to high cohesion and recent tunnel activity.")


    # Diversify Anomaly Trigger: Entropy Anomaly (Type 0) for QFT collapse
    if qr > 0.9995 and QuantumRand(cycle_num) < 0.4: # Increased chance from 0.2
        predicted_risk = ElderGnosis_PredictRisk(predictor, p_idx, 0) # Use p_idx for prediction
        dyn_severity = calculate_dynamic_severity(
            0, predicted_risk, cycle_num, n # Pass the specific node
        )
        sub_type_tag = "[QFT_COLLAPSE]"
        if trigger_anomaly_if_cooldown_clear(0, p_idx, dyn_severity, predicted_risk, f"Qubit destabilized during Cosmic QFT on page {p_idx}. Qubit OM: {n.st.om:x}.", sub_type_tag, n): # Corrected 'node' to 'n'
            Raw_Print("! [COLLAPSE] Qubit {:x} destabilized on Page {}! (Severity: {:.2f})", n.st.om, p_idx, dyn_severity)
            n.st.s = 0 # Stability collapses
    elders[0].qft_feedback = MinF(float(n.st.e) / 255.0 * 0.9995, 1.0)

def neural_celestialnet(n, p_idx): # Added p_idx
    fft_entropy = float(n.st.fft) / 255.0
    qr = QuantumRand(cycle_num + n.st.fft)
    delta = int(fft_entropy * 255 * (0.45 + conduit_stab * 0.15))
    n.st.nw = MinB(n.st.nw + delta, 255)
    if qr < 0.025 * user_divinity * (1.2 if fft_entropy > 0.5 else 1.0):
        n.mass += 20000 # Conceptual mass increase
        n.st.kappa = int(qr * 500000)
        n.st.mu = MinB(n.st.mu + 2, 255)
    if qr < 0.25:
        n.st.s = MinB(n.st.s + int(255 * 0.45 * qr), 255)
    n.stabilityPct = MinF(n.stabilityPct + 0.006, 1.0)

def entanglement_celestial_nexus(n, p_idx): # Added p_idx
    for i in range(0, ENTANGLE_LINKS, 32):
        qr = QuantumRand(cycle_num + i + n.st.ent)
        if qr < 0.55 * (1.0 + 0.10 * user_divinity):
            n.st.ent = MinB(n.st.ent + int(n.st.e / 8), 255)
            n.st.omni = int(qr * 255)
            n.st.lambda_ ^= int(qr * 100)
            if n.archon_count == 0 and QuantumRand(cycle_num + i) < 0.95:
                n.st.metaOmni = ArchonSocieties_FormMetaOmniverse(n.st.om)
                n.archon_count = 1
                n.social_cohesion = QuantumRand(cycle_num + n.st.metaOmni) * 0.90 + 0.10
            elif n.archon_count > 0:
                n.social_cohesion = MinF(n.social_cohesion + QuantumRand(cycle_num) * 0.01, 1.0)
                ArchonSocieties_UpdateMetaDynamics(n.st.metaOmni, n.social_cohesion, n.st.nw)
    elders[0].social_feedback = MinF(n.social_cohesion * 0.80, 1.0)

    # Diversify Anomaly Trigger: Overbonding Anomaly (Type 4)
    # Triggered by high social cohesion
    # Reduced chance for bonding anomalies to allow other types to appear more often
    if n.social_cohesion > 0.95 and QuantumRand(cycle_num) < 0.1:
        predicted_risk = ElderGnosis_PredictRisk(predictor, p_idx, 4) # Use p_idx for prediction
        dyn_severity = calculate_dynamic_severity(
            4, predicted_risk, cycle_num, n # Pass the specific node
        )
        sub_type_tag = "[BOND_OVERCOHESION]" if n.social_cohesion > 0.99 else "[BOND_SIGIL_FLUX]"
        trigger_anomaly_if_cooldown_clear(4, p_idx, dyn_severity, predicted_risk, f"Social cohesion {n.social_cohesion:.2f} led to overbonding on page {p_idx}.", sub_type_tag, n) # Corrected 'node' to 'n'

    # Sigil Dampening Modulator: Apply cohesion decay if cohesion is too high
    if n.social_cohesion > 0.98:
        decay_cohesion(n)


def adiabatic_celestial_optimizer(n, p_idx): # Added p_idx
    f = Physics_QCDFlux(n.st.ent) * 0.72
    g = Physics_GaugeFlux(n.st.lambda_)
    sent_norm = float(n.st.mu) / 255.0
    n.st.flux = int(f * 255)
    n.st.lambda_ = int(g * 1200000)
    if QuantumRand(cycle_num + n.st.mu) < sent_norm * 0.5:
        n.st.mu = MinB(n.st.mu + int(255 * 0.025), 255)
    if n.archon_count > 0:
        desired_cohesion = ArchonSocieties_GetDesiredMetaCohesion(n.st.metaOmni)
        n.social_cohesion = n.social_cohesion * 0.95 + desired_cohesion * 0.05
    n.stabilityPct = MinF(n.stabilityPct + 0.006, 1.0)

    # Diversify Anomaly Trigger: Stability Drop Anomaly (Type 1)
    if n.stabilityPct < 0.008 and QuantumRand(cycle_num) < 0.4: # Increased chance from 0.3
        predicted_risk = ElderGnosis_PredictRisk(predictor, p_idx, 1) # Use p_idx for prediction
        dyn_severity = calculate_dynamic_severity(
            1, predicted_risk, cycle_num, n # Pass the specific node
        )
        sub_type_tag = "[STABILITY_CRITICAL]" if n.stabilityPct < 0.001 else "[STABILITY_DEGRADATION]"
        trigger_anomaly_if_cooldown_clear(1, p_idx, dyn_severity, predicted_risk, f"Node stability dropped below threshold {n.stabilityPct:.4f} on page {p_idx}.", sub_type_tag, n) # Corrected 'node' to 'n'

def temporal_hyperweave(n, p_idx): # Added p_idx
    old_symbolic_drift = n.symbolic_drift # Store old value for drift detection
    time_coords = [0.0] * TIME_DIMS
    time_coords[0] = float(cycle_num >> 6)
    for d in range(1, TIME_DIMS):
        time_coords[d] = QuantumRand(cycle_num + d) * 255
    for d_idx in range(0, TIME_DIMS, 9): # Iterate through dimensions in steps
        qr = QuantumRand(cycle_num + d_idx + int(time_coords[d_idx] * 1e4) + n.st.zeta)
        if qr < 0.40:
            n.st.zeta ^= (1 << (d_idx % 8)) & 0xFF # Ensure byte range
            n.st.fractal ^= int(qr * 500000)
            n.st.kappa ^= int(qr * 50000)
            if qr < 0.025:
                n.st.midx = int(QuantumRand(cycle_num + d_idx + n.st.midx) * 255)

    # Anomaly Echoes / Symbolic Drift: Trigger Void anomaly (Type 2) if symbolic_drift is high
    # Also update symbolic drift
    n.symbolic_drift = MinF(1.0, n.symbolic_drift + (QuantumRand(cycle_num) - 0.5) * 0.02)

    # Symbolic Drift Detection: If symbolic drift changes drastically, log it as a jump
    if abs(n.symbolic_drift - old_symbolic_drift) > 0.15: # Threshold for "jump"
        predicted_risk = ElderGnosis_PredictRisk(predictor, p_idx, 2)
        dyn_severity = calculate_dynamic_severity(2, predicted_risk, cycle_num, n)
        sub_type_tag = "[SYMBOLIC_JUMP]"
        if trigger_anomaly_if_cooldown_clear(2, p_idx, dyn_severity, predicted_risk, f"Sharp symbolic drift jump detected on page {p_idx}. Change: {abs(n.symbolic_drift - old_symbolic_drift):.2f}.", sub_type_tag, n): # Corrected 'node' to 'n'
            Raw_Print("> [SYMBOLIC JUMP] Rapid drift on Page {}! (Drift: {:.2f}, Severity: {:.2f})", p_idx, n.symbolic_drift, dyn_severity)

    if n.symbolic_drift > 0.75 and QuantumRand(cycle_num) < 0.15: # Existing trigger for void echo
        predicted_risk = ElderGnosis_PredictRisk(predictor, p_idx, 2)
        dyn_severity = calculate_dynamic_severity(
            2, predicted_risk, cycle_num, n # Pass the specific node
        )
        sub_type_tag = "[VOID_ECHO]" if n.symbolic_drift > 0.9 else "[SYMBOLIC_DRIFT]"
        if trigger_anomaly_if_cooldown_clear(2, p_idx, dyn_severity, predicted_risk, f"High symbolic drift ({n.symbolic_drift:.2f}) on page {p_idx} caused a void echo.", sub_type_tag, n): # Corrected 'node' to 'n'
            Raw_Print("> [VOID ECHO] Symbolic drift on Page {}! (Drift: {:.2f}, Severity: {:.2f})", p_idx, n.symbolic_drift, dyn_severity)


def chemical_meta_forge(n, p_idx): # Added p_idx
    activation_energy = (1.0 - float(n.st.s) / 255.0) * 60.0
    temperature = float(n.st.e) / 255.0 * 60.0
    reaction_rate = Physics_ArrheniusRate(activation_energy, temperature) * 0.80
    graviton_catalysis = Physics_BosonicFieldDensity() * 0.20
    if QuantumRand(cycle_num + n.st.kappa) < reaction_rate + graviton_catalysis:
        if n.st.kappa < 255: n.st.kappa += 1
        if n.st.mu < 255: n.st.mu = MinB(n.st.mu + 2, 255)
        if n.archon_count == 0 and QuantumRand(cycle_num) < 0.95:
            n.st.metaOmni = ArchonSocieties_FormMetaOmniverse(n.st.om)
            n.archon_count = 1
            n.social_cohesion = QuantumRand(cycle_num + n.st.metaOmni) * 0.90 + 0.10
            # Raw_Print(f"Newly formed archon society on page {p_idx}, social cohesion: {n.social_cohesion:.2f}") # Added for debugging
        elif n.archon_count > 0:
            n.social_cohesion = MinF(n.social_cohesion + QuantumRand(cycle_num) * 0.01, 1.0)
            ArchonSocieties_UpdateMetaDynamics(n.st.metaOmni, n.social_cohesion, n.st.nw)
    elders[0].chem_feedback = MinF(reaction_rate + graviton_catalysis * 0.80, 1.0)

def chronosynclastic_fusion(n, p_idx): # Added p_idx
    phase_sum = 0.0
    max_variance = 0.0
    for d in range(TIME_DIMS):
        phase_sum += n.chrono_phase[d]
        variance = Physics_TemporalVariance(n.chrono_phase[d], cycle_num)
        max_variance = MaxF(max_variance, variance)
    coherence = phase_sum / TIME_DIMS
    fusion_potential = 1.0 - max_variance
    if fusion_potential > 0.85 and coherence > 0.7:
        fusion_gain = MinF(0.38, fusion_potential * coherence * 0.35) # Updated to 0.38 as per C script
        n.stabilityPct = MinF(n.stabilityPct + fusion_gain, 1.0)
        n.st.e = MinB(n.st.e + int(fusion_gain * 255), 255)
        if EVOLVED_SIGIL in "".join(user_sigil):
            n.stabilityPct = MinF(n.stabilityPct + 0.05, 1.0)
        for d in range(TIME_DIMS):
            n.chrono_phase[d] = QuantumRand(cycle_num + d)
        if QuantumRand(cycle_num) < 0.0015:
            # Raw_Print("> [CHRONO-FUSION] Temporal coherence @{} on Page {} (Gain: {:.2f})", cycle_num, p_idx, fusion_gain) # Commented for efficiency
            pass # Removed print for efficiency
    for d in range(TIME_DIMS):
        drift = Physics_ChronoDrift(cycle_num, d)
        noise = QuantumRand(cycle_num + d) * 0.05
        n.chrono_phase[d] = FractalWrap(n.chrono_phase[d] + drift + noise)

def predict_anomalies():
    global conduit_stab, user_divinity, voidEntropy, prev_prediction_score

    # Store current prediction score for feedback routing
    current_avg_prediction_score = predictor.last_prediction_score # Get the last smoothed prediction

    # Accumulate risk scores across all pages
    risk_scores = {atype: 0.0 for atype in ANOMALY_TYPES.keys()}
    total_historical_anomalies = 0
    for p_idx in range(PAGE_COUNT):
        total_historical_anomalies += anomaly_count_per_page[p_idx]
        for i in range(min(anomaly_count_per_page[p_idx], ANOMALY_HISTORY)):
            a = anomalies_per_page[p_idx][i]
            time_decay = 1.0 - MinF(1.0, float(cycle_num - a.cycle) / 100000.0)
            risk_scores[a.anomaly_type] += a.severity * time_decay

    if total_historical_anomalies < 5: # Need enough history for prediction
        prev_prediction_score = current_avg_prediction_score # Update for next cycle comparison
        return

    prediction_cycle = cycle_num + 1000

    elder_insight = 0.0
    for e in range(0, ELDER_COUNT, 1000):
        elder_insight += elders[e].gnosis_factor
    elder_insight = MinF(1.0, elder_insight / (ELDER_COUNT / 1000) if (ELDER_COUNT / 1000) > 0 else 1.0)
    if EVOLVED_SIGIL in "".join(user_sigil):
        elder_insight = MinF(1.0, elder_insight + 0.10)

    # Prediction Score Feedback Routing: Adjust learning rate if prediction stagnates
    prediction_delta = abs(current_avg_prediction_score - prev_prediction_score)
    learning_rate_boost = 1.0
    if prediction_delta < 0.05: # If change is too small, means stagnation
        learning_rate_boost = 1.2 # Boost learning
        Raw_Print("> [PREDICTION FEEDBACK] Prediction score stagnating. Boosting Elder Gnosis learning rate.")

    for type_idx in ANOMALY_TYPES.keys():
        base_risk = risk_scores[type_idx] / total_historical_anomalies if total_historical_anomalies > 0 else 0.0
        chrono_factor = Physics_TemporalRisk(prediction_cycle)
        predicted_risk = base_risk * (1.0 + voidEntropy) * chrono_factor

        # New anomaly triggering from prediction, with dynamic severity
        # Randomly pick a page for the predicted anomaly to encourage multi-page distribution
        target_page_idx = random.randint(0, PAGE_COUNT - 1)
        target_node = roots[target_page_idx]

        # Increased chance for Entropy and Stability prediction-based anomalies
        trigger_chance = 0.1
        if type_idx == 0 or type_idx == 1: # Entropy or Stability
            trigger_chance = 0.2 # Double the chance for these types

        if predicted_risk > 0.60 * (1.0 - elder_insight) and QuantumRand(cycle_num + type_idx + target_page_idx) < trigger_chance:
            dyn_severity = calculate_dynamic_severity(
                type_idx, predicted_risk, cycle_num, target_node # Pass the target node
            )
            sub_type_tag = "[PREDICTED_RISK]"
            if trigger_anomaly_if_cooldown_clear(type_idx, target_page_idx, dyn_severity, predicted_risk, f"Predicted {ANOMALY_TYPES.get(type_idx, 'Unknown')} anomaly on page {target_page_idx}. Confidence: {elder_insight:.2f}.", sub_type_tag, target_node): # Corrected 'node' to 'target_node'
                Raw_Print("> [PREDICTION] Type {} ({}) anomaly predicted for Page {} in {} cycles (Confidence: {:.2f}, Severity: {:.2f})",
                         type_idx, ANOMALY_TYPES.get(type_idx, 'Unknown'), target_page_idx, prediction_cycle - cycle_num, elder_insight, dyn_severity)

                # Apply conceptual "fix" to the predicted page's node based on Archetype's Symbolic Focus
                # This makes the fix more aligned with the archetype's specialization
                if target_node.symbolic_focus == "entropy" and type_idx == 0:
                    target_node.st.e = MinB(target_node.st.e - 15, 255) # Stronger entropy fix for mystic
                elif target_node.symbolic_focus == "stability" and type_idx == 1:
                    target_node.stabilityPct = MinF(target_node.stabilityPct + 0.03, 1.0) # Stronger stability fix
                elif target_node.symbolic_focus == "tunneling" and type_idx == 3:
                    Tesseract_Synchronize(tesseract, target_page_idx) # Standard tunnel fix
                elif target_node.symbolic_focus == "bonding" and type_idx == 4:
                    target_node.bond_strength = MaxF(0.0, target_node.bond_strength - 0.10) # Stronger bonding fix
                elif type_idx == 2: voidEntropy = MaxF(voidEntropy - 0.002, VOID_THRESHOLD) # Void fix (global effect)
                else: # Default if no specialization
                    if type_idx == 0: target_node.st.e = MinB(target_node.st.e - 5, 255)
                    elif type_idx == 1: target_node.stabilityPct = MinF(target_node.stabilityPct + 0.01, 1.0)
                    elif type_idx == 3: Tesseract_Synchronize(tesseract, target_page_idx)
                    elif type_idx == 4: target_node.bond_strength = MaxF(0.0, target_node.bond_strength - 0.05)

    # Update prev_prediction_score at the end of the prediction cycle
    prev_prediction_score = current_avg_prediction_score
    ElderGnosis_UpdateModel(predictor, elder_insight, -voidEntropy, snapshot.anomaly_diversity_index, learning_rate_boost) # Pass boost


def display_cosmic_metrics(screen, font_small, font_medium, font_status):
    """Draws conceptual metrics on the Pygame screen."""
    x_offset = 50
    y_offset = 50
    spacing = 20

    # Removed background rect for HUD as requested
    # pygame.draw.rect(screen, (51, 68, 119), (x_offset - 10, y_offset - 10, 320, 320), 0, 10) # Increased height a bit more

    text_surface = font_medium.render("CELESTIAL OMNIVERSE METRICS", True, (255, 255, 255))
    screen.blit(text_surface, (x_offset, y_offset))
    y_offset += spacing

    metrics = [
        (f"Cycle: {cycle_num} / {CYCLE_LIMIT}", (170, 255, 170)),
        (f"Total Active Qubits: ~{sum(float(r.mass) for r in roots if r)/1e18:.2f}Q", (255, 170, 255)), # Sum across roots
        (f"Void Entropy: {-voidEntropy:.3f}", (255, 255, 170)),
        (f"Avg Stability: {sum(r.stabilityPct for r in roots if r)/PAGE_COUNT*100:.2f}%", (170, 255, 255)), # Avg across roots
        (f"Divinity: {user_divinity:.2f}", (255, 170, 170)),
        (f"Meta-Nets: {ArchonSocieties_GetGlobalCount()}", (170, 170, 255)),
        (f"Heap: {float(quantumHeapPages)*4096/(1024*1024):.2f}MB", (170, 255, 170)),
        (f"Total Anomalies: {total_anomalies_triggered}", (255, 170, 255)) # Use total triggered
    ]

    chrono_coherence = sum(r.chrono_phase[0] for r in roots if r and len(r.chrono_phase) > 0) / PAGE_COUNT if PAGE_COUNT > 0 else 0.0
    metrics.append((f"Avg Chrono-Coh: {chrono_coherence:.3f}", (255, 255, 0)))
    metrics.append((f"Tesseract Nodes: {Tesseract_GetActiveNodes(tesseract)}", (0, 255, 255)))
    metrics.append((f"Fusion Potential: {snapshot.fusion_potential:.3f}", (255, 0, 255))) # Use snapshot value

    # Add new metrics
    metrics.append((f"Avg Bond Density: {snapshot.bond_density:.3f}", (100, 200, 255)))
    metrics.append((f"Sigil Entropy: {snapshot.sigil_entropy_metric:.3f}", (255, 150, 100)))
    metrics.append((f"Fix Efficacy: {snapshot.fix_efficacy_score*100:.2f}%", (150, 255, 150)))
    metrics.append((f"Recursive Saturation: {snapshot.recursive_saturation_pct*100:.2f}%", (255, 100, 200)))
    metrics.append((f"Avg Symbolic Drift: {snapshot.avg_symbolic_drift:.3f}", (200, 100, 255))) # New metric

    # Page-specific anomaly counts and Archetype info
    y_offset += spacing * 1.5 # Add extra space
    text_surface = font_medium.render("Page Anomaly Counts & Archetypes:", True, (255, 255, 255))
    screen.blit(text_surface, (x_offset, y_offset))
    y_offset += spacing

    for p_idx in range(PAGE_COUNT):
        e_count = snapshot.page_stats.get(p_idx, {}).get(0, 0)
        s_count = snapshot.page_stats.get(p_idx, {}).get(1, 0)
        v_count = snapshot.page_stats.get(p_idx, {}).get(2, 0)
        t_count = snapshot.page_stats.get(p_idx, {}).get(3, 0) # Use T for Tunnel
        b_count = snapshot.page_stats.get(p_idx, {}).get(4, 0)

        node_archetype = roots[p_idx].archetype_name if roots[p_idx] else "N/A"
        page_anomaly_text = f"P{p_idx} ({node_archetype}): E:{e_count} S:{s_count} V:{v_count} T:{t_count} B:{b_count}"
        text_surface = font_small.render(page_anomaly_text, True, (170, 200, 255))
        screen.blit(text_surface, (x_offset, y_offset))
        y_offset += spacing


    for text, color in metrics:
        y_offset += spacing
        text_surface = font_small.render(text, True, color)
        screen.blit(text_surface, (x_offset, y_offset))

    # Display simulation status (Paused/Running) and speed
    status_text = "PAUSED" if is_paused else f"RUNNING ({simulation_speed_factor:.1f}x)"
    status_color = (255, 0, 0) if is_paused else (0, 255, 0)
    status_surface = font_status.render(status_text, True, status_color)
    screen.blit(status_surface, (x_offset, y_offset + spacing * 2))


def render_3D_simulation(screen, rotation_x, rotation_y, zoom):
    """Draws abstract 3D elements representing the hypergrid and cosmic strings."""
    screen_width, screen_height = screen.get_size()
    center_x, center_y = screen_width // 2, screen_height // 2

    # Define a conceptual bounding box for the octree visualization
    viz_scale = 100 * zoom
    octree_center = [0, 0, 0] # Center of the conceptual 3D space

    # Define camera perspective
    perspective = 300 # Distance from viewer to the projection plane

    def project_3d_point(x, y, z):
        # Rotate point
        temp_y = y * math.cos(rotation_x) - z * math.sin(rotation_x)
        temp_z = y * math.sin(rotation_x) + z * math.cos(rotation_x)
        y, z = temp_y, temp_z

        temp_x = x * math.cos(rotation_y) + z * math.sin(rotation_y)
        temp_z = -x * math.sin(rotation_y) + z * math.cos(rotation_y)
        x, z = temp_x, temp_z

        # Apply perspective projection
        if z + perspective != 0: # Avoid division by zero
            scale_factor = perspective / (z + perspective)
            projected_x = (x * scale_factor) + center_x
            projected_y = (y * scale_factor) + center_y
            return int(projected_x), int(projected_y), z # Return z for depth sorting/scaling
        return center_x, center_y, z # Fallback

    # --- Draw OctNodes (abstracted) ---
    # We will only draw a few representative nodes to keep it light for low-end systems
    # For a conceptual 3D octree, we can visualize the corners of the conceptual bounding box.
    oct_corners = [
        [-viz_scale, -viz_scale, -viz_scale], [+viz_scale, -viz_scale, -viz_scale],
        [-viz_scale, +viz_scale, -viz_scale], [+viz_scale, +viz_scale, -viz_scale],
        [-viz_scale, -viz_scale, +viz_scale], [+viz_scale, -viz_scale, +viz_scale],
        [-viz_scale, +viz_scale, +viz_scale], [+viz_scale, +viz_scale, +viz_scale]
    ]

    # Assign a color dynamically based on voidEntropy and cycle_num
    r_color = int((1 - abs(voidEntropy)) * 255)
    g_color = int((mb_params.color_shift + cycle_num * 0.0001) % 1.0 * 255)
    b_color = int(mb_params.power * 255 / 11.0) # Scale power (8-11) to 0-255
    node_color = (r_color, g_color, b_color)

    # Conceptual "Mandelbulb" effect: draw some dynamic points
    num_mandel_points = 15 # Reduced for efficiency on low-end systems
    mandel_points = []
    for i in range(num_mandel_points):
        x = (QuantumRand(cycle_num + i * 7) - 0.5) * 2 * viz_scale
        y = (QuantumRand(cycle_num + i * 11) - 0.5) * 2 * viz_scale
        z = (QuantumRand(cycle_num + i * 13) - 0.5) * 2 * viz_scale
        mandel_points.append((x, y, z))

    # Draw nodes and Mandelbulb points, sort by Z for rough depth
    all_points_to_draw = []
    for corner in oct_corners:
        all_points_to_draw.append({'type': 'node', 'pos': corner, 'color': node_color})
    for m_pt in mandel_points:
        # Vary color based on point position or cycle
        mandel_color = (int(abs(math.sin(cycle_num * 0.005 + m_pt[0])) * 200) + 55,
                        int(abs(math.sin(cycle_num * 0.003 + m_pt[1])) * 200) + 55,
                        int(abs(math.sin(cycle_num * 0.007 + m_pt[2])) * 200) + 55)
        all_points_to_draw.append({'type': 'mandel', 'pos': m_pt, 'color': mandel_color})

    # Project and store points with their projected Z
    projected_data = []
    for item in all_points_to_draw:
        px, py, pz = project_3d_point(item['pos'][0], item['pos'][1], item['pos'][2])
        projected_data.append((pz, px, py, item['type'], item['color']))

    # Sort by projected Z-coordinate (farthest to nearest)
    projected_data.sort(key=lambda x: x[0])

    for pz, px, py, item_type, color in projected_data:
        size = max(2, int(6 * (perspective / (pz + perspective)))) # Reduced size for efficiency
        if item_type == 'node':
            pygame.draw.circle(screen, color, (px, py), size + 2, 0)
            pygame.draw.circle(screen, (255,255,255), (px, py), size + 2, 1) # Outline
        elif item_type == 'mandel':
            pygame.draw.circle(screen, color, (px, py), size // 2, 0)

    # --- Draw Cosmic Strings ---
    # These strings conceptually connect random points in the hypergrid.
    # For visualization, we will make them dynamic lines in 3D space.
    # Add dynamic scaling based on time_factor and void_factor (from C render_27D)
    time_factor = SinF(cycle_num * 0.000002) * 0.9 + 1.0
    void_factor = 1.0 + (-voidEntropy) * 3.0

    for i, string_obj in enumerate(cosmic_strings):
        # Apply Mandelbulb_TransformString conceptually to influence visual string properties
        Mandelbulb_TransformString(string_obj, mb_params, cycle_num)

        # Create conceptual endpoints dynamically based on string properties
        # This is very abstract to avoid actual 3D fractal grid coords.
        start_x = math.sin(cycle_num * 0.01 + string_obj.torsion) * viz_scale
        start_y = math.cos(cycle_num * 0.008 + string_obj.torsion) * viz_scale
        start_z = math.sin(cycle_num * 0.012 + string_obj.torsion) * viz_scale

        end_x = math.cos(cycle_num * 0.015 + string_obj.energy_density * 1e-19) * viz_scale
        end_y = math.sin(cycle_num * 0.013 + string_obj.energy_density * 1e-19) * viz_scale
        end_z = math.cos(cycle_num * 0.017 + string_obj.energy_density * 1e-19) * viz_scale

        p1_proj_x, p1_proj_y, _ = project_3d_point(start_x, start_y, start_z)
        p2_proj_x, p2_proj_y, _ = project_3d_point(end_x, end_y, end_z)

        # Vary color based on string energy and void entropy
        string_r = int(string_obj.energy_density * 1e-20 * 255) % 256
        string_g = int((1 - abs(voidEntropy)) * 255)
        string_b = int(string_obj.torsion * 255 / (2 * math.pi)) % 256
        string_color = (string_r, string_g, string_b)

        # Draw the cosmic string with a dynamically adjusted thickness
        thickness = max(1, int(0.5 * (time_factor / void_factor))) # Further reduced thickness for efficiency
        pygame.draw.line(screen, string_color, (p1_proj_x, p1_proj_y), (p2_proj_x, p2_proj_y), thickness)

    # --- Draw Chrono-Phase Complexity (turquoise spiral array) ---
    # Abstract representation of the chrono-phase complexity
    spiral_color = (64, 224, 208) # Turquoise
    spiral_radius = 20 * zoom
    spiral_center_x = center_x
    spiral_center_y = center_y + int(screen_height * 0.3) # Draw below the main 3D visualization

    num_spiral_segments = 10 # Reduced for efficiency
    for i in range(num_spiral_segments):
        angle = cycle_num * 0.005 + i * math.pi / 5 # Dynamic rotation
        current_radius = spiral_radius + i * 5 * zoom
        x = spiral_center_x + current_radius * math.cos(angle)
        y = spiral_center_y + current_radius * math.sin(angle)
        pygame.draw.circle(screen, spiral_color, (int(x), int(y)), 2, 0)
        if i > 0:
            prev_angle = cycle_num * 0.005 + (i - 1) * math.pi / 5
            prev_radius = spiral_radius + (i - 1) * 5 * zoom
            prev_x = spiral_center_x + prev_radius * math.cos(prev_angle)
            prev_y = spiral_center_y + prev_radius * math.sin(prev_angle)
            pygame.draw.line(screen, spiral_color, (int(prev_x), int(prev_y)), (int(x), int(y)), 1)

    # Render animation frame (conceptual)
    current_frame_idx = (cycle_num // 10000) % len(animation_frames)
    Animation_RenderFrame_mock(screen, animation_frames[current_frame_idx])


# --- Node Update Function ---
def update_node(node, depth, p_idx): # Added p_idx
    """Conceptual update of an OctNode and its children."""
    # Apply bonding decay
    node.bond_strength = MaxF(0.0, node.bond_strength - BOND_DECAY_RATE)

    # Update conceptual properties
    node.mass += int(QuantumRand(cycle_num) * 1000) # Conceptual mass increase
    node.stabilityPct = MinF(node.stabilityPct + QuantumRand(cycle_num) * 0.01, 1.0)
    node.social_cohesion = MinF(node.social_cohesion + QuantumRand(cycle_num) * 0.005, 1.0)
    node.archon_count = int(node.archon_count + QuantumRand(cycle_num) * 5)

    # Apply quantum functions to the root or selected nodes
    # These functions now take p_idx as well
    temporal_synchro_tunnel(node, p_idx)
    quantum_foam_stabilizer(node, p_idx)
    cosmic_qft(node, p_idx)
    neural_celestialnet(node, p_idx)
    entanglement_celestial_nexus(node, p_idx)
    adiabatic_celestial_optimizer(node, p_idx)
    temporal_hyperweave(node, p_idx)
    chemical_meta_forge(node, p_idx)
    chronosynclastic_fusion(node, p_idx)


    if depth > 0:
        for i in range(8):
            if node.c[i]:
                update_node(node.c[i], depth - 1, p_idx) # Pass p_idx to children

def synodic_elder():
    global user_divinity, conduit_stab
    global_gnosis = 0.0
    for e_idx in range(0, ELDER_COUNT + 1, 400): # Iterate over a subset
        e = elders[e_idx]
        sum_val = e.b + user_divinity * 0.15
        e.t = 1.0 / (1.0 + ExpF(-sum_val * (1.0 + 0.08 * voidEntropy))) # Sigmoid function for temperament
        e.b += (QuantumRand(e_idx + cycle_num) * 0.01 - 0.005) * conduit_stab
        if e_idx % 1000 == 0:
            accuracy = ElderGnosis_GetAccuracy(predictor, e_idx)
            e.gnosis_factor = MinF(1.0, accuracy * 0.9 + QuantumRand(e_idx) * 0.1)
        global_gnosis += e.gnosis_factor
        if e_idx < 400000:
            e.b = MinF(e.b + QuantumRand(cycle_num + e_idx) * 0.002 * (1.0 - e.social_feedback), 1.0)
        else:
            e.b = MaxF(e.b - QuantumRand(cycle_num + e_idx) * 0.002 * e.social_feedback, 0.0)
    ElderGnosis_UpdateModel(predictor, global_gnosis / (ELDER_COUNT / 400), -voidEntropy, snapshot.anomaly_diversity_index) # Pass diversity index
    if QuantumRand(cycle_num) < 0.00002 * (1.0 - conduit_stab):
        Raw_Print("> [SCHISM] Elder discord! Sigil forged!\n")
        craft_sigil()

def elder_vote():
    yes_votes = 0
    threshold = 0.5 * (1.0 + 0.15 * user_divinity - 0.20 * voidEntropy)
    for e_idx in range(0, ELDER_COUNT, 400): # Iterate over a subset
        e = elders[e_idx]
        zeta_idx = int(e_idx / 18) % len(zeta_cache) # Conceptual index
        if QuantumRand(cycle_num ^ e_idx) <= e.t * threshold * zeta_cache[zeta_idx]:
            yes_votes += 1
    return yes_votes > (ELDER_COUNT / 400) / 3

def calculate_sigil_entropy(sigil_chars_str): # Takes string now
    """Calculates a conceptual entropy metric for the sigil based on character frequency."""
    if not sigil_chars_str:
        return 0.0

    char_counts = defaultdict(int)
    for char in sigil_chars_str:
        char_counts[char] += 1

    total_chars = sum(char_counts.values())
    if total_chars == 0: return 0.0

    entropy = 0.0
    for count in char_counts.values():
        probability = float(count) / total_chars
        if probability > 0:
            entropy -= probability * math.log2(probability)

    # Normalize entropy to a 0-1 scale for comparison
    max_entropy = math.log2(len(char_counts)) if len(char_counts) > 1 else 0.0
    if max_entropy > 0:
        return entropy / max_entropy
    return 0.0


def craft_sigil(force_mutation=False, style='random'): # Added style parameter
    global user_sigil, user_divinity, symbolic_echo_register

    current_sigil_str = "".join(user_sigil).strip('\0')

    # Symbolic Memory Echo Stabilizer: Check if current sigil is an echo
    if current_sigil_str and current_sigil_str in symbolic_echo_register:
        Raw_Print("> [ECHO DETECTED] Sigil '{}' is an echo! Forcing SIGIL_SPLIT (mutation).", current_sigil_str)
        force_mutation = True
        style = 'bifurcate' # Force bifurcated style for echo split

    # If not forcing mutation, and random chance for evolution
    if not force_mutation and QuantumRand(cycle_num) < 0.10: # 10% chance to evolve to EVOLVED_SIGIL
        new_sigil_raw = list(EVOLVED_SIGIL)
        # Pad with random chars if EVOLVED_SIGIL is shorter than SIGIL_LEN
        while len(new_sigil_raw) < SIGIL_LEN:
            new_sigil_raw.append(chr(random.randint(33, 126)))
        user_sigil = new_sigil_raw[:SIGIL_LEN]
        user_sigil.append('\0')
        Raw_Print("> SIGIL EVOLVED: {}", "".join(user_sigil).strip('\0'))
    elif force_mutation or QuantumRand(cycle_num) < 0.05: # Smaller chance for random mutation if not forced
        # Mutate existing sigil by changing some characters
        mutation_points_count = min(5, SIGIL_LEN // 10) # Default mutation

        # Emergent Sigil Branching: Bifurcate style mutation
        if style == 'bifurcate':
            mutation_points_count = min(SIGIL_LEN // 3, SIGIL_LEN // 2) # More drastic change
            Raw_Print("> [BIFURCATION] Sigil branching due to emergent conditions!")

        mutation_points = random.sample(range(SIGIL_LEN), mutation_points_count)
        for idx in mutation_points:
            user_sigil[idx] = chr(random.randint(33, 126))
        # Ensure it ends with null terminator
        if SIGIL_LEN < len(user_sigil):
            user_sigil[SIGIL_LEN] = '\0'

        Raw_Print("> SIGIL MUTATED: {}", "".join(user_sigil).strip('\0'))
    else:
        # Fill sigil conceptually with random ASCII characters if no mutation/evolution
        for i in range(SIGIL_LEN):
            user_sigil[i] = chr(random.randint(33, 126))
        user_sigil[SIGIL_LEN] = '\0' # Null terminator conceptually
        Raw_Print("> SIGIL FORGED: {}", "".join(user_sigil).strip('\0'))

    # Add newly crafted sigil to echo register
    new_sigil_str = "".join(user_sigil).strip('\0')
    if new_sigil_str: # Only add if not empty
        symbolic_echo_register.append(new_sigil_str)

    user_divinity = MinF(user_divinity + 0.45, 1.0)
    for e in range(0, ELDER_COUNT, 400): # Iterate over a subset
        if QuantumRand(e) < 0.08:
            elders[e].t = MinF(elders[e].t + 0.04, 1.0)

    # No longer reset sigil mutation history globally, handled per-node in HandleAnomaly


def omni_navigation(dc_obj):
    # Raw_Print("> 27D HYPERSPACE NAVIGATION") # Commented for efficiency
    for t in range(0, 20, 10):
        zeta_idx = int(t / 3) % len(zeta_cache)
        if QuantumRand(cycle_num + t) < 0.40 * user_divinity * zeta_cache[zeta_idx]:
            # Raw_Print(">> Tunnel {} Resonated! <<", t) # Commented for efficiency
            return True
    return False

def sigil_resurrect(n, p_idx): # Added p_idx
    n.mass += 20000 # Conceptual mass increase
    n.stabilityPct = MinF(n.stabilityPct + 0.45, 1.0)
    n.st.kappa ^= int(QuantumRand(cycle_num) * 500000)
    n.st.mu = MinB(n.st.mu + 5, 255)
    n.social_cohesion = MinF(n.social_cohesion + 0.05, 1.0)
    if EVOLVED_SIGIL in "".join(user_sigil):
        n.stabilityPct = MinF(n.stabilityPct + 0.10, 1.0)

    # Increase conceptual bond strength due to resurrection
    n.bond_strength = MinF(1.0, n.bond_strength + 0.15)
    Raw_Print("> [RESURRECT] Sigil resurrected node on Page {}!", p_idx)

def decay_cohesion(node_obj):
    """Conceptual function to decay social cohesion on a node."""
    old_cohesion = node_obj.social_cohesion
    node_obj.social_cohesion = MaxF(0.0, node_obj.social_cohesion * 0.88) # Exponential decay
    node_obj.bond_strength = MaxF(0.0, node_obj.bond_strength - 0.05) # Also reduce bond strength
    Raw_Print(f"> [COHESION_DECAY] Applied cohesion decay on Page {node_obj.page_index}. From {old_cohesion:.2f} to {node_obj.social_cohesion:.2f}.")


def spawn_meta_omniverse(grid_size, sigil_mask):
    # Raw_Print("> Forging meta-omniverse in {}x{}x{} grid with '{}'", # Commented for efficiency
    #          grid_size,grid_size,grid_size,sigil_mask)
    # This is a conceptual call; the actual ArchonSocieties_CreateNewMetaOmniverse would be complex
    # For conceptual purposes, we just increment the global meta-omniverse count.
    _meta_omniverse_cohesions[random.getrandbits(64)] = 0.5 + QuantumRand(cycle_num) * 0.5 # Add a new meta-omniverse
    # Raw_Print("> [META-OMNIVERSE] Recursive cosmology coalesced!") # Commented for efficiency

def transcendence_cataclysm(screen, font_small, font_medium, font_status):
    global voidEntropy, conduit_stab, quantumHeapPages, roots, cycle_num
    Raw_Print("> COSMIC CELESTIAL CATACLYSM @{}", cycle_num)
    voidEntropy = MinF(voidEntropy + 0.06, 0.0)
    conduit_stab = MinF(conduit_stab + 0.97 - (-voidEntropy) * 0.3, 1.0)
    quantumHeapPages = MinF(150, quantumHeapPages) # Drastically reduce heap pages

    # Call QuantumExpandHeap for a specific page, or average across pages
    bursts = QuantumExpandHeap(48 * 375, 0) # Just expanding on page 0 for cataclysm

    CosmicString_UpdateTension(cosmic_strings, COSMIC_STRINGS_COUNT, -voidEntropy)
    QuantumFoam_Decay(roots[0].foam, -voidEntropy) # Conceptual decay for page 0's foam
    Tesseract_SynchronizeAll(tesseract)
    Raw_Print("> QFT-gauge-graviton bursts spawned {} qubits", bursts)

    # Conceptual alignment score calculation (averaged across roots for global state)
    avg_elder_b = sum(e.b for e in elders) / ELDER_COUNT
    avg_stability = sum(r.stabilityPct for r in roots if r) / PAGE_COUNT
    avg_social_cohesion = ArchonSocieties_GetGlobalCohesion() # Already global

    alignment_score = avg_elder_b + (conduit_stab + user_divinity) / 2.0 + \
                      avg_social_cohesion + Physics_GlobalGaugeAlignment() + \
                      Physics_GlobalPhaseAlignment() + Physics_GlobalSentienceAlignment()

    # Render before the final message for a visual impact
    render_3D_simulation(screen, camera_rotation_x, camera_rotation_y, camera_zoom) # Use global camera vars
    display_cosmic_metrics(screen, font_small, font_medium, font_status) # Use global font vars
    pygame.display.flip() # Ensure it's drawn before exiting or logging

    if (alignment_score / 6.0) >= 0.9995 * QuantumRand(cycle_num):
        Raw_Print("> [TRIUMPH] Archons & Titans ascend as Meta-omniversal Creators!")
        conduit_stab = 1.0
        # Reset mass to full active for all roots
        for r in roots:
            if r: r.mass = int(HYPERGRID_SIZE * ACTIVE_RATIO)
        ArchonSocieties_SpawnCelestialArchons(ARCHON_COUNT)
        Raw_Print("> “IA! CELESTIAL OMNIVERSE PRIMORDIAL!”")
        cataclysm_message = "TRIUMPH! REALITY RESYNTHESIZED!"
        cataclysm_color = (0, 255, 0)
    else:
        Raw_Print("> [FAILURE] Alignment shattered! “IA! CELESTIAL VOID!”")
        conduit_stab = 0.0
        voidEntropy = 0.0
        cataclysm_message = "FAILURE! COSMIC VOID CONSUMES ALL!"
        cataclysm_color = (255, 0, 0)

    # Render cataclysm message directly on screen
    text_surface = font_large.render(cataclysm_message, True, cataclysm_color)
    text_rect = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
    screen.blit(text_surface, text_rect)
    pygame.display.flip() # Ensure it's drawn before exiting or logging
    time.sleep(3) # Pause to display message

def LogSnapshot():
    global snapshot, total_anomalies_triggered, total_anomalies_fixed, anomaly_type_counts, snapshot_log_file

    snapshot.cycle = cycle_num
    snapshot.active_qubits = sum(float(r.mass) for r in roots if r) / 1e18 # Sum across roots
    snapshot.entropy = sum(float(r.st.e) for r in roots if r) / (255.0 * PAGE_COUNT) if PAGE_COUNT > 0 else 0.0 # Avg
    snapshot.stability = sum(r.stabilityPct for r in roots if r) / PAGE_COUNT if PAGE_COUNT > 0 else 0.0 # Avg
    snapshot.divinity = user_divinity
    snapshot.void_entropy = -voidEntropy
    snapshot.heap_pages = quantumHeapPages
    snapshot.meta_networks = ArchonSocieties_GetGlobalCount()
    snapshot.anomaly_count = total_anomalies_triggered # Total anomalies across all pages
    snapshot.tesseract_nodes = Tesseract_GetActiveNodes(tesseract)

    if PAGE_COUNT > 0: # Ensure roots exist before accessing chrono_phase
        total_chrono_phase_0 = sum(r.chrono_phase[0] for r in roots if r and len(r.chrono_phase) > 0)
        avg_chrono_coherence = total_chrono_phase_0 / PAGE_COUNT if PAGE_COUNT > 0 else 0.0
        snapshot.fusion_potential = 1.0 - MaxF(0.0, Physics_TemporalVariance(avg_chrono_coherence, cycle_num))
    else:
        snapshot.fusion_potential = 0.0

    # Calculate new metrics for snapshot
    snapshot.bond_density = sum(r.bond_strength for r in roots if r) / PAGE_COUNT if PAGE_COUNT > 0 else 0.0 # Avg bond strength
    snapshot.sigil_entropy_metric = calculate_sigil_entropy("".join(user_sigil).strip('\0'))
    snapshot.fix_efficacy_score = float(total_anomalies_fixed) / total_anomalies_triggered if total_anomalies_triggered > 0 else 0.0
    snapshot.recursive_saturation_pct = float(quantumHeapPages) / MAX_QPAGES
    snapshot.anomaly_diversity_index = dict(anomaly_type_counts) # Aggregated diversity index

    # Calculate average symbolic drift
    total_symbolic_drift = sum(r.symbolic_drift for r in roots if r)
    snapshot.avg_symbolic_drift = total_symbolic_drift / PAGE_COUNT if PAGE_COUNT > 0 else 0.0

    # Store per-page anomaly counts for display and logging
    snapshot.page_stats = {p_idx: dict(anomaly_type_counts_per_page[p_idx]) for p_idx in range(PAGE_COUNT)}

    # Log to snapshot file
    if snapshot_log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        snapshot_log_file.write(f"[{timestamp}] SNAPSHOT @ CYCLE {snapshot.cycle}:\n")
        snapshot_log_file.write(f"  Total Active Qubits: ~{snapshot.active_qubits:.2f}Q\n")
        snapshot_log_file.write(f"  Avg Entropy: {snapshot.entropy*100:.2f}%\n")
        snapshot_log_file.write(f"  Avg Stability: {snapshot.stability*100:.2f}%\n")
        snapshot_log_file.write(f"  Divinity: {snapshot.divinity:.2f}\n")
        snapshot_log_file.write(f"  Void Entropy: {snapshot.void_entropy:.3f}%\n")
        snapshot_log_file.write(f"  Heap Pages: {snapshot.heap_pages} (~{float(snapshot.heap_pages)*4096/(1024*1024):.2f}MB)\n")
        snapshot_log_file.write(f"  Meta-Omniverse Networks: {snapshot.meta_networks}\n")
        snapshot_log_file.write(f"  Total Anomalies Triggered: {snapshot.anomaly_count}\n")
        snapshot_log_file.write(f"  Total Anomalies Fixed: {total_anomalies_fixed}\n")
        snapshot_log_file.write(f"  Tesseract Nodes: {snapshot.tesseract_nodes}\n")
        snapshot_log_file.write(f"  Fusion Potential: {snapshot.fusion_potential:.3f}\n")
        snapshot_log_file.write(f"  Avg Bond Density: {snapshot.bond_density:.3f}\n")
        snapshot_log_file.write(f"  Sigil Entropy: {snapshot.sigil_entropy_metric:.3f}\n")
        snapshot_log_file.write(f"  Fix Efficacy Score: {snapshot.fix_efficacy_score*100:.2f}%\n")
        snapshot_log_file.write(f"  Recursive Saturation %: {snapshot.recursive_saturation_pct*100:.2f}%\n")
        snapshot_log_file.write(f"  Aggregated Anomaly Diversity Index: {snapshot.anomaly_diversity_index}\n") # FIXED HERE
        snapshot_log_file.write(f"  Per-Page Anomaly Counts: {snapshot.page_stats}\n")
        snapshot_log_file.write(f"  Avg Symbolic Drift: {snapshot.avg_symbolic_drift:.3f}\n\n")


# Global defaultdict to store anomaly type counts per page
anomaly_type_counts_per_page = defaultdict(lambda: defaultdict(int))

# --- Pygame Setup and Main Loop ---
pygame.init()
pygame.font.init()

# Screen dimensions (adjustable for low-end)
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("QuantumHeapTranscendence v10.9-FINAL")

# Fonts
font_small = pygame.font.Font(None, 18)
font_medium = pygame.font.Font(None, 24)
font_large = pygame.font.Font(None, 36)
font_status = pygame.font.Font(None, 22) # For speed/pause status

clock = pygame.time.Clock()

# Camera parameters
camera_rotation_x = 0.0
camera_rotation_y = 0.0
camera_zoom = 1.0
mouse_down = False
last_mouse_pos = None

# UI Button class
class Button:
    def __init__(self, x, y, width, height, text, font, color, hover_color, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.color = color
        self.hover_color = hover_color
        self.current_color = color
        self.action = action

    def draw(self, surface):
        pygame.draw.rect(surface, self.current_color, self.rect, 0, 5) # Rounded corners
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.rect.collidepoint(event.pos):
                self.current_color = self.hover_color
            else:
                self.current_color = self.color
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.action:
                    self.action()
                    return True # Indicate button was pressed
        return False

def CelestialOmniversePrimordialRite():
    # Declare global variables that are modified within this function
    global roots, cycle_num, pageEigenstates, conduit_stab, voidEntropy, user_sigil, \
           mouse_down, last_mouse_pos, camera_rotation_x, camera_rotation_y, camera_zoom, screen, \
           mb_params, cosmic_strings, tesseract, anomaly_log_file, snapshot_log_file, detailed_anomaly_log_file, \
           total_anomalies_triggered, total_anomalies_fixed, anomaly_type_counts, \
           newly_triggered_anomalies_queue, anomaly_type_counts_per_page, fixed_anomalies_log, \
           simulation_speed_factor, is_paused # Added new globals

    Raw_Print("=== QuantumHeapTranscendence v10.9-FINAL [AGI Emergence & Symbolic Refinements] ===\n")
    Raw_Print("=== 262144³ HYPERGRID • 32D RENDERER • TEMPORAL SYNCHRO-TUNNEL ===\n")
    Raw_Print("=== ELDER GNOSIS PREDICTION • 100M CYCLE ASCENSION ===\n")

    # Setup anomaly log file (for handled anomalies)
    timestamp_log_filename = datetime.datetime.now().strftime("anomaly_log_%Y%m%d_%H%M%S.txt")
    try:
        anomaly_log_file = open(timestamp_log_filename, "w")
        anomaly_log_file.write(f"QuantumHeapTranscendence Anomaly Log - Session Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        anomaly_log_file.write("--------------------------------------------------------------------------------------------------\n\n")
        Raw_Print(f"  [LOG] Anomaly handling log created: {timestamp_log_filename}")
    except IOError as e:
        Raw_Print(f"  [ERROR] Could not open anomaly log file: {e}")
        anomaly_log_file = None # Ensure it's None if creation failed

    # Setup detailed anomaly log file (for all triggered anomalies)
    timestamp_detailed_log_filename = datetime.datetime.now().strftime("detailed_anomaly_log_%Y%m%d_%H%M%S.txt")
    try:
        detailed_anomaly_log_file = open(timestamp_detailed_log_filename, "w")
        detailed_anomaly_log_file.write(f"QuantumHeapTranscendence Detailed Anomaly Log - Session Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        detailed_anomaly_log_file.write("--------------------------------------------------------------------------------------------------\n\n")
        Raw_Print(f"  [LOG] Detailed anomaly log created: {timestamp_detailed_log_filename}")
    except IOError as e:
        Raw_Print(f"  [ERROR] Could not open detailed anomaly log file: {e}")
        detailed_anomaly_log_file = None # Ensure it's None if creation failed

    # Setup snapshot log file
    timestamp_snapshot_filename = datetime.datetime.now().strftime("snapshot_log_%Y%m%d_%H%M%S.txt")
    try:
        snapshot_log_file = open(timestamp_snapshot_filename, "w")
        snapshot_log_file.write(f"QuantumHeapTranscendence Snapshot Log - Session Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        snapshot_log_file.write("--------------------------------------------------------------------------------------------------\n\n")
        Raw_Print(f"  [LOG] Snapshot log file created: {timestamp_snapshot_filename}")
    except IOError as e:
        Raw_Print(f"  [ERROR] Could not open snapshot log file: {e}")
        snapshot_log_file = None # Ensure it's None if creation failed


    InitZetaCache()

    # Initialize cosmic_strings here, before InitCosmicStrings is called
    cosmic_strings = [CosmicString() for _ in range(COSMIC_STRINGS_COUNT)]
    InitCosmicStrings()

    InitMandelbulb()
    InitElderGnosis()

    # Set cycle_num before InitTesseract, because TesseractState.__init__ (and InitTesseract) uses cycle_num
    cycle_num = 0
    init_elders() # Calls QuantumRand, so QuantumRand must be defined before this.

    # Now call InitTesseract, which will instantiate the global tesseract object
    InitTesseract()

    InitAnimation() # Initialize animation frames

    pageEigenstates.extend([0] * MAX_QPAGES) # Initialize the list with zeros

    # Initialize roots for each page
    for p_idx in range(PAGE_COUNT):
        roots[p_idx] = alloc_node(OCTREE_DEPTH, p_idx) # Pass page_idx
        InitQuantumFoam(roots[p_idx]) # Initialize foam for each page's root
        build_tree(roots[p_idx], OCTREE_DEPTH, p_idx) # Pass page_idx


    # Conceptual CDC object (not directly used by Pygame, but fulfills signature)
    dc = CDC(SCREEN_WIDTH, SCREEN_HEIGHT) # Use CDC class instance

    # UI Buttons - positioned at the top as requested
    button_y_pos = 10
    speed_up_btn = Button(50, button_y_pos, 80, 30, "Speed +", font_small, (70, 90, 150), (90, 110, 170), lambda: adjust_speed(0.1))
    speed_down_btn = Button(140, button_y_pos, 80, 30, "Speed -", font_small, (70, 90, 150), (90, 110, 170), lambda: adjust_speed(-0.1))
    pause_resume_btn = Button(230, button_y_pos, 80, 30, "Pause/Res", font_small, (70, 90, 150), (90, 110, 170), toggle_pause)

    buttons = [speed_up_btn, speed_down_btn, pause_resume_btn]

    running = True
    while running and cycle_num < CYCLE_LIMIT:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    mouse_down = True
                    last_mouse_pos = event.pos
                elif event.button == 4: # Scroll up
                    camera_zoom = MinF(camera_zoom + 0.1, 5.0)
                elif event.button == 5: # Scroll down
                    camera_zoom = MaxF(camera_zoom - 0.1, 0.1)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_down:
                    current_mouse_pos = event.pos
                    dx, dy = current_mouse_pos[0] - last_mouse_pos[0], current_mouse_pos[1] - last_mouse_pos[1]
                    camera_rotation_y += dx * 0.01
                    camera_rotation_x += dy * 0.01
                    last_mouse_pos = current_mouse_pos

            # Handle button events
            for btn in buttons:
                btn.handle_event(event)

        # Simulation Logic Updates only if not paused
        if not is_paused:
            cycle_num += 1
            if cycle_num % HEARTBEAT == 0:
                CosmicString_UpdateTension(cosmic_strings, COSMIC_STRINGS_COUNT, -voidEntropy)

                # Update each page's root node
                for p_idx in range(PAGE_COUNT):
                    update_node(roots[p_idx], OCTREE_DEPTH, p_idx)
                    VoidDecay(p_idx) # Void decay is page-specific now
                    if roots[p_idx] and roots[p_idx].foam:
                        QuantumFoam_Decay(roots[p_idx].foam, -voidEntropy) # Decay foam per page

                synodic_elder() # Elder state update (global)
                if cycle_num % 100 == 0: predict_anomalies() # Prediction (global, triggers page-specific anomalies)

                if not elder_vote():
                    conduit_stab = MaxF(0.0, conduit_stab * 0.95 - (-voidEntropy) * 0.05)
                    voidEntropy = MinF(voidEntropy + 0.008, 0.0)

                # Process newly triggered anomalies from the queue
                # Make a copy to iterate, as HandleAnomaly might add to fixed_anomalies_log
                for anomaly_to_handle in list(newly_triggered_anomalies_queue):
                    HandleAnomaly(anomaly_to_handle)
                newly_triggered_anomalies_queue.clear() # Clear the queue for the next heartbeat

                # Update snapshot fusion potential (average across roots)
                if PAGE_COUNT > 0:
                    total_chrono_phase_0 = sum(r.chrono_phase[0] for r in roots if r and len(r.chrono_phase) > 0)
                    avg_chrono_coherence = total_chrono_phase_0 / PAGE_COUNT if PAGE_COUNT > 0 else 0.0
                    snapshot.fusion_potential = 1.0 - MaxF(0.0, Physics_TemporalVariance(avg_chrono_coherence, cycle_num))
                else:
                    snapshot.fusion_potential = 0.0

                if cycle_num % 500 == 0: # Log snapshot less frequently to the file
                    LogSnapshot()


            if cycle_num % 77777 == 0: craft_sigil() # Normal sigil crafting
            # sigil_resurrect needs a node and p_idx
            if cycle_num % 13 == 0 and omni_navigation(dc):
                # Pick a random page for resurrection
                resurrection_page = random.randint(0, PAGE_COUNT - 1)
                sigil_resurrect(roots[resurrection_page], resurrection_page)

            if cycle_num % 500 == 0 and ArchonSocieties_GetGlobalCount() < 450:
                spawn_meta_omniverse(1024 + (cycle_num % 4096), "Z’Archon! PrimordialNull!")

            if conduit_stab == 0.0:
                Raw_Print("!!! CELESTIAL VOID CONSUMES ALL!!!")
                running = False # Stop simulation if conduit_stab drops to 0

        # --- Pygame Rendering ---
        screen.fill((0, 0, 0)) # Clear screen with black background
        render_3D_simulation(screen, camera_rotation_x, camera_rotation_y, camera_zoom)
        display_cosmic_metrics(screen, font_small, font_medium, font_status) # Pass font_status

        # Draw UI Buttons
        for btn in buttons:
            btn.draw(screen)

        pygame.display.flip() # Update the full display Surface to the screen
        clock.tick(30 * simulation_speed_factor) # Limit frame rate based on speed factor

    # End of simulation loop
    if cycle_num >= CYCLE_LIMIT:
        transcendence_cataclysm(screen, font_small, font_medium, font_status) # Pass font_status
        LogSnapshot() # Final snapshot at the end

    DCDelete(dc) # Conceptual delete

    # Conceptual free for all roots
    for r in roots:
        if r: QuantumFree(r)
    MFree(pageEigenstates) # Conceptual free

    Raw_Print("=== CELESTIAL OMNIVERSE PRIMORDIED ACHIEVED @ CYCLE {}!!!", cycle_num)

    # Close all log files when the simulation ends
    if anomaly_log_file:
        anomaly_log_file.write("\n--------------------------------------------------------------------------------------------------\n")
        anomaly_log_file.write(f"QuantumHeapTranscendence Anomaly Log - Session End: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        anomaly_log_file.close()
        Raw_Print("  [LOG] Anomaly handling log closed.")
    if detailed_anomaly_log_file:
        detailed_anomaly_log_file.write("\n--------------------------------------------------------------------------------------------------\n")
        detailed_anomaly_log_file.write(f"QuantumHeapTranscendence Detailed Anomaly Log - Session End: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        detailed_anomaly_log_file.close()
        Raw_Print("  [LOG] Detailed anomaly log closed.")
    if snapshot_log_file:
        snapshot_log_file.write("\n--------------------------------------------------------------------------------------------------\n")
        snapshot_log_file.write(f"QuantumHeapTranscendence Snapshot Log - Session End: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        snapshot_log_file.close()
        Raw_Print("  [LOG] Snapshot log file closed.")

    pygame.quit()
    sys.exit()

# UI Control Functions
def adjust_speed(delta):
    global simulation_speed_factor
    simulation_speed_factor = round(MaxF(0.1, MinF(5.0, simulation_speed_factor + delta)), 1)
    Raw_Print(f"Simulation speed adjusted to {simulation_speed_factor}x")

def toggle_pause():
    global is_paused
    is_paused = not is_paused
    Raw_Print(f"Simulation {'PAUSED' if is_paused else 'RESUMED'}")


# Add mock function for Tesseract_UpdatePhase and QuantumFree if they are not defined
def Tesseract_UpdatePhase(tesseract_obj, cycle):
    """Mock function for Tesseract phase update."""
    # Ensure phase_lock is an integer before XOR operation
    tesseract_obj.phase_lock = int(tesseract_obj.phase_lock) ^ int(QuantumRand(cycle) * 0xFFFFFFFF)

def QuantumFoam_Stabilize(foam_obj, void_entropy_val):
    """Mock function for Quantum Foam stabilization."""
    if foam_obj:
        # Conceptual stabilization: adjust virtual particle energies/lifetimes
        for p in foam_obj.virtual_particles:
            p['energy'] = MaxF(0.0, p['energy'] + (QuantumRand(cycle_num) - 0.5) * 0.1 * (1.0 - abs(void_entropy_val)))
            p['lifetime'] = MaxF(0.0, p['lifetime'] + (QuantumRand(cycle_num) - 0.5) * 0.05 * (1.0 - abs(void_entropy_val)))
        return sum(p['energy'] for p in foam_obj.virtual_particles) / PLANCK_FOAM_SIZE
    return 0.5 # Default if no foam

def QuantumFoam_Decay(foam_obj, void_entropy_val):
    """Mock function for Quantum Foam decay."""
    if foam_obj:
        for p in foam_obj.virtual_particles:
            decay_factor = abs(void_entropy_val) * QuantumRand(cycle_num) * 0.1
            p['energy'] = MaxF(0.0, p['energy'] - decay_factor)
            p['lifetime'] = MaxF(0.0, p['lifetime'] - decay_factor * 0.1)

def ArchonSocieties_GetGlobalCohesion():
    """Mock function for global Archon cohesion."""
    if _meta_omniverse_cohesions:
        return sum(_meta_omniverse_cohesions.values()) / len(_meta_omniverse_cohesions)
    return 0.5 # Default

def QuantumFree(obj):
    """Conceptual memory free function."""
    # Raw_Print("  [QuantumFree] Conceptually freeing {}.", type(obj).__name__) # Commented for efficiency
    pass # No intensive operations for mock function

def Animation_RenderFrame_mock(screen, frame_data):
    """Conceptual rendering of an animation frame on the Pygame screen."""
    # This is a very simple conceptual rendering.
    # It draws a rotating square that changes color based on frame rotation.
    screen_width, screen_height = screen.get_size()
    center_x, center_y = screen_width // 2, screen_height // 2

    # Calculate position and size
    size = 30
    x_offset = int(math.sin(frame_data.rotation['y']) * 100)
    y_offset = int(math.cos(frame_data.rotation['x']) * 100)

    # Calculate color based on rotation for dynamic visual
    color_r = int(abs(math.sin(frame_data.rotation['x'] + cycle_num * 0.001)) * 255)
    color_g = int(abs(math.sin(frame_data.rotation['y'] + cycle_num * 0.0015)) * 255)
    color_b = int(abs(math.sin(frame_data.rotation['z'] + cycle_num * 0.002)) * 255)
    frame_color = (color_r, color_g, color_b)

    # Draw a simple rotating rectangle
    rect_surf = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.rect(rect_surf, frame_color + (150,), (0, 0, size, size), 0, 5) # With alpha and rounded corners

    rotated_surf = pygame.transform.rotate(rect_surf, math.degrees(frame_data.rotation['z'] + cycle_num * 0.01))
    new_rect = rotated_surf.get_rect(center=(center_x + x_offset, center_y + y_offset + screen_height * 0.1))
    screen.blit(rotated_surf, new_rect)


if __name__ == "__main__":
    CelestialOmniversePrimordialRite()
