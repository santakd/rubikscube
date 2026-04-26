#   __.-._
#   '-._"7'
#    /'.-c
#    |  //
#   _)_/||
#
# rubiks3x3.py - 3x3 Rubik's Cube 3-D Solver with detailed logging.
# Author: santakd
# Contact: santakd at gmail dot com
# Date: April 22, 2026
# Version: 1.0.8
# License: MIT License 
#
# ========================================================================
# Rubik's Cube Solver — Installation & Run Instructions
# ========================================================================
#
# REQUIREMENTS
#   Python 3.9+   (https://www.python.org/downloads/)
#   pip3 install pygame kociemba numpy
#
# The app is a 3-D interactive Rubik's Cube rendered with PyGame. 
# It has three modes:
#  Mode	        Description
#  Edit	        User clicks stickers to set the scrambled state
#  Solve	    App computes the Kociemba solution
#  Playback     User steps forward/backward through the solution
#
# RUN
#   python3 rubiks3x3.py
#
# CONTROLS
#   Left-click a sticker  → cycle through face colors (W Y R O B G)
#   Right-click a sticker → reset that sticker to its default color
#   [Solve]               → compute Kociemba solution and load step list
#   [Next]                → advance one move in the solution
#   [Back]                → undo one move in the solution
#   [Reset]               → return cube to solved state and clear solution
#   Drag (empty space)    → rotate the 3-D view
#   Scroll wheel (space)  → zoom in/out
#   Scroll wheel (panel)  → scroll the move list in Playback mode
#
# NOTES
#   • The cube must be in a valid, solvable state before clicking Solve.
#     Invalid configurations produce a clear on-screen error message.
#   • Each face must contain exactly 9 stickers; each color must appear
#     exactly 9 times across the whole cube.
#   • The solver uses Kociemba's two-phase algorithm (≤ 20 moves, < 1 s).
#   • The 3-D renderer uses an orthographic projection and is not a true 3-D engine.
#   • The code is organized into 11 sections: Constants, Logger, Exceptions,
#     CubeState, KociembaBridge, MoveExecutor, SolutionPlayer, Cube3DRenderer,
#     NetUI, App, and the __main__ entry point.
#   • This version does have scrollable move list
#
# Kociemba's Algorithm
# Herbert Kociemba’s algorithm is absolutely fascinating. 
# It is the gold standard for software-based Rubik’s Cube solvers and is the mathematical 
# backbone that proved "God's Number" (the maximum number of moves required to solve any 
# Rubik's Cube state is 20). Here is a breakdown of how the algorithm works conceptually, 
# followed by exactly how we integrated it into the Python application.
#
# A Rubik’s Cube has about 43 quintillion possible states. You cannot brute-force 
# a solution. To solve this, Kociemba invented the Two-Phase Algorithm in 1992. 
# Instead of trying to solve the cube in one massive leap, the algorithm breaks 
# the problem down into two smaller, highly optimized steps:
#
# Phase 1: "Restrict the move set"
# Normally, you can turn any face 90 degrees: `U, D, L, R, F, B`. 
# In Phase 1, the algorithm looks for a sequence of moves to get the cube into a 
# specific mathematical subgroup (a restricted state). In this state:
# * All corners are oriented correctly (no twisted corners).
# * All edges are oriented correctly (no flipped edges).
# * The four "middle slice" edges belong to the middle slice.
# Once the cube is in this state, it can be fully solved using only 180-degree turns
# for the vertical faces (`R2, L2, F2, B2`) and standard turns for the top/bottom (`U, D`).
# This drastically reduces the complexity of the remaining solution space, making it
# Phase 2: "Solve the rest"
# Using only that restricted move set (`U, D, R2, L2, F2, B2`), the algorithm searches
# for the sequence that actually solves the cube. Because the state space is now 
# infinitely smaller, this search is incredibly fast.
# 
# The Magic (Iterative Deepening):
# If Phase 1 takes 12 moves and Phase 2 takes 15 moves, the total is 27 moves. 
# Kociemba's algorithm doesn't stop there. It goes back and says, "What if I take 14 moves
# in Phase 1? Does that allow me to do Phase 2 in only 5 moves?" It rapidly searches these
# trade-offs using massive pre-computed lookup tables (pruning tables) until it finds a
# solution that is ≤ 20 moves.
# ========================================================================

import sys                                      # For system exit and command-line arguments
import math                                     # For trigonometric functions in 3D rendering
import copy                                     # For deep copying cube states
import logging                                  # For logging debug/info/warning messages to console and file
import threading                                # For running the solver in a separate thread to avoid freezing the UI
from typing import Optional, Dict, List, Tuple  # For type annotations to improve code clarity and catch errors
from datetime import datetime                   # For timestamping log files and messages

try:
    import pygame                               # For rendering and input handling
    import kociemba                             # For solving the cube
    import numpy as np                          # For 3D geometry calculations
except ImportError as e:
    print("=" * 70)
    print("MISSING DEPENDENCIES DETECTED")
    print(f"Error: {e}")
    print("Please install required packages using:")
    print("    pip3 install pygame kociemba numpy")
    print("=" * 70)
    sys.exit(1)


# ========================================================================
# 1. Constants & color palette, rendering parameters, and other fixed values
# ========================================================================

# check your cube for these colors/orientation and change if needed! 
# The app relies on the centers to determine face colors and the Kociemba string mapping
COLORS: Dict[str, Tuple[int, int, int]] = {
    "W": (255, 255, 255),   # Up    face — white
    "Y": (255, 215,   2),   # Down  face — yellow
    "R": (183,  16,  52),   # Front face — red
    "O": (255,  80,   2),   # Back  face — orange
    "B": (  2,  68, 177),   # Right face — blue
    "G": (  2, 156,  73),   # Left  face — green
}

# Centers define the face's color and must remain unchanged for a valid cube
FACE_DEFAULTS: Dict[str, str] = {
    "U": "W", "D": "Y", "F": "R", "B": "O", "R": "B", "L": "G"
}

# Rendering Constants
STICKER_PX: float = 44.0
GAP_PX: float = 3.0
WINDOW_WIDTH: int = 1024
WINDOW_HEIGHT: int = 768
FPS_LIMIT: int = 60

# UI Constants
BUTTON_H: int = 44
BUTTON_MIN_W: int = 110
BUTTON_PAD_X: int = 18
BUTTON_RADIUS: int = 6
BUTTON_COLOR_NORMAL: Tuple[int, int, int] = (50, 50, 60)
BUTTON_COLOR_HOVER: Tuple[int, int, int] = (80, 80, 95)
BUTTON_COLOR_DISABLED: Tuple[int, int, int] = (35, 35, 40)
STATUS_BAR_H: int = 28
TOAST_DURATION_S: float = 4.0
COLOR_POPUP_RADIUS: int = 15
COLOR_POPUP_GAP: int = 8

# Highlight Overlay Color (R, G, B, A)
HIGHLIGHT_COLOR: Tuple[int, int, int, int] = (40, 100, 255, 80)


# ========================================================================
# 2. Logger setup
# ========================================================================

# Generate the timestamped log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"rubiks3x3_{timestamp}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, mode="w"),
    ],
)
logger = logging.getLogger("rubiks3x3")


# ========================================================================
# Exceptions
# ========================================================================

class RubiksSolverError(Exception):
    """Base exception for the Rubik's Cube Solver."""
    pass

class InvalidCubeError(RubiksSolverError):
    """Raised when the cube state is physically invalid."""
    pass

class SolverEngineError(RubiksSolverError):
    """Raised when the Kociemba library fails to find a solution."""
    pass

class MoveError(RubiksSolverError):
    """Raised when an invalid move string is requested."""
    pass


# ========================================================================
# 3. CubeState
# ========================================================================

class CubeState:
    """Pure data model representing the 3x3 Rubik's Cube facelet colors."""

    def __init__(self) -> None:
        """Initialize the cube in a solved state."""
        self.state: Dict[str, List[str]] = {
            face: [color] * 9 for face, color in FACE_DEFAULTS.items()
        }
        logger.debug("CubeState initialized to solved state.")

    def solved(self) -> bool:
        """Return True if all stickers match their face's default color."""
        for face, default_color in FACE_DEFAULTS.items():
            if any(sticker != default_color for sticker in self.state[face]):
                return False
        return True

    def is_valid(self) -> Tuple[bool, str]:
        """Validate exact 9 of each color, 54 total, and centers unchanged."""
        color_counts: Dict[str, int] = {c: 0 for c in COLORS.keys()}
        total = 0

        for face, stickers in self.state.items():
            if len(stickers) != 9:
                return False, f"Face {face} does not have exactly 9 stickers."
            if stickers[4] != FACE_DEFAULTS[face]:
                return False, f"Center sticker of face {face} was changed."
            for color in stickers:
                if color not in color_counts:
                    return False, f"Invalid color code '{color}' found."
                color_counts[color] += 1
                total += 1

        if total != 54:
            return False, f"Cube has {total} stickers, expected 54."

        for color, count in color_counts.items():
            if count != 9:
                return False, f"Color {color} appears {count} times, expected 9."

        return True, "Valid"

    def to_kociemba_string(self) -> str:
        """Convert current colors to a 54-char string (URFDLB face order)."""
        # Map current center colors to their face codes
        color_to_face = {
            self.state["U"][4]: "U",
            self.state["R"][4]: "R",
            self.state["F"][4]: "F",
            self.state["D"][4]: "D",
            self.state["L"][4]: "L",
            self.state["B"][4]: "B",
        }
        order = ["U", "R", "F", "D", "L", "B"]
        kociemba_str = ""
        for face in order:
            for color in self.state[face]:
                kociemba_str += color_to_face[color]
        return kociemba_str

    @classmethod
    def from_kociemba_string(cls, s: str) -> "CubeState":
        """Create a CubeState from a 54-char URFDLB Kociemba face string."""
        if len(s) != 54:
            raise InvalidCubeError("Kociemba string must be exactly 54 chars.")
        cube = cls()
        order = ["U", "R", "F", "D", "L", "B"]
        idx = 0
        for face in order:
            for i in range(9):
                face_char = s[idx]
                cube.state[face][i] = FACE_DEFAULTS[face_char]
                idx += 1
        return cube

    def apply_move(self, move: str) -> None:
        """Mutate state by applying one WCA move string."""
        # Delegates to MoveExecutor, evaluated at runtime so order is OK
        MoveExecutor().apply_move(self, move)

    def reset(self) -> None:
        """Restore the solved state."""
        self.state = {
            face: [color] * 9 for face, color in FACE_DEFAULTS.items()
        }
        logger.info("CubeState reset to solved.")

    def copy(self) -> "CubeState":
        """Return a deep copy of this CubeState."""
        new_cube = CubeState()
        new_cube.state = copy.deepcopy(self.state)
        return new_cube


# ========================================================================
# 4. KociembaBridge
# ========================================================================

class KociembaBridge:
    """Wraps kociemba.solve(); converts results to a list of WCA move strings."""

    def solve(self, cube: CubeState) -> List[str]:
        """Compute the solution, returning a list of WCA move strings."""
        is_valid, msg = cube.is_valid()
        if not is_valid:
            logger.error(f"Cannot solve invalid cube: {msg}")
            raise InvalidCubeError(msg)

        kociemba_str = cube.to_kociemba_string()
        logger.info(f"Initiating Kociemba solve for string: {kociemba_str}")

        try:
            solution_str = kociemba.solve(kociemba_str)
            moves = solution_str.split()
            logger.info(f"Solution found: {solution_str} ({len(moves)} moves)")
            return moves
        except Exception as e:
            logger.error(f"Kociemba solver failed: {e}")
            raise SolverEngineError(f"Engine failed to find a solution: {e}")


# ========================================================================
# 5. MoveExecutor
# ========================================================================

class MoveExecutor:
    """Applies a single WCA move string to CubeState by cycling face arrays."""
    """Implements all 18 WCA outer-layer moves: U U' U2 D D' D2 F F' F2 B B' B2 L L' L2 R R' R2."""
    # Defines cycles of 4 stickers for a clockwise 90-degree rotation.
    # Each cycle is (face, index), tracing a sticker's movement: A -> B -> C -> D -> A
    CYCLES: Dict[str, List[Tuple[Tuple[str, int], ...]]] = {
        "U": [
            (("U", 0), ("U", 2), ("U", 8), ("U", 6)),
            (("U", 1), ("U", 5), ("U", 7), ("U", 3)),
            (("F", 2), ("L", 2), ("B", 2), ("R", 2)),
            (("F", 1), ("L", 1), ("B", 1), ("R", 1)),
            (("F", 0), ("L", 0), ("B", 0), ("R", 0)),
        ],
        "D": [
            (("D", 0), ("D", 2), ("D", 8), ("D", 6)),
            (("D", 1), ("D", 5), ("D", 7), ("D", 3)),
            (("F", 6), ("R", 6), ("B", 6), ("L", 6)),
            (("F", 7), ("R", 7), ("B", 7), ("L", 7)),
            (("F", 8), ("R", 8), ("B", 8), ("L", 8)),
        ],
        "F": [
            (("F", 0), ("F", 2), ("F", 8), ("F", 6)),
            (("F", 1), ("F", 5), ("F", 7), ("F", 3)),
            (("U", 6), ("R", 0), ("D", 2), ("L", 8)),
            (("U", 7), ("R", 3), ("D", 1), ("L", 5)),
            (("U", 8), ("R", 6), ("D", 0), ("L", 2)),
        ],
        "B": [
            (("B", 0), ("B", 2), ("B", 8), ("B", 6)),
            (("B", 1), ("B", 5), ("B", 7), ("B", 3)),
            (("U", 2), ("L", 0), ("D", 6), ("R", 8)),
            (("U", 1), ("L", 3), ("D", 7), ("R", 5)),
            (("U", 0), ("L", 6), ("D", 8), ("R", 2)),
        ],
        "L": [
            (("L", 0), ("L", 2), ("L", 8), ("L", 6)),
            (("L", 1), ("L", 5), ("L", 7), ("L", 3)),
            (("U", 0), ("F", 0), ("D", 0), ("B", 8)),
            (("U", 3), ("F", 3), ("D", 3), ("B", 5)),
            (("U", 6), ("F", 6), ("D", 6), ("B", 2)),
        ],
        "R": [
            (("R", 0), ("R", 2), ("R", 8), ("R", 6)),
            (("R", 1), ("R", 5), ("R", 7), ("R", 3)),
            (("U", 8), ("B", 0), ("D", 8), ("F", 8)),
            (("U", 5), ("B", 3), ("D", 5), ("F", 5)),
            (("U", 2), ("B", 6), ("D", 2), ("F", 2)),
        ],
    }

    def apply_move(self, cube: CubeState, move: str) -> None:
        """Rotate the 3D cube arrays according to the move string."""
        if len(move) == 0 or move[0] not in self.CYCLES:
            raise MoveError(f"Unknown move: {move}")

        base = move[0]
        if len(move) == 1:
            times = 1
        elif move[1] == "'":
            times = 3
        elif move[1] == "2":
            times = 2
        else:
            raise MoveError(f"Invalid move suffix: {move}")

        cycles = self.CYCLES[base]
        for _ in range(times):
            old_state = copy.deepcopy(cube.state)
            for cycle in cycles:
                a, b, c, d = cycle
                # A -> B -> C -> D -> A
                cube.state[b[0]][b[1]] = old_state[a[0]][a[1]]
                cube.state[c[0]][c[1]] = old_state[b[0]][b[1]]
                cube.state[d[0]][d[1]] = old_state[c[0]][c[1]]
                cube.state[a[0]][a[1]] = old_state[d[0]][d[1]]

        logger.debug(f"Applied move {move}")

    def get_affected_stickers(self, move: str) -> List[Tuple[str, int]]:
        """Return a list of (face, index) affected by a move."""
        if not move or move[0] not in self.CYCLES:
            return []
        affected = set()
        for cycle in self.CYCLES[move[0]]:
            for sticker in cycle:
                affected.add(sticker)
        # Add center sticker explicitly
        affected.add((move[0], 4))
        return list(affected)

    @classmethod
    def validate_all_moves(cls) -> None:
        """Run a self-test of all 18 WCA moves + inverses to ensure correctness."""
        state = CubeState()
        executor = cls()
        faces = ["U", "D", "F", "B", "L", "R"]
        for f in faces:
            for suffix, inv in [("", "'"), ("'", ""), ("2", "2")]:
                move = f + suffix
                inverse = f + inv
                state.reset()
                executor.apply_move(state, move)
                executor.apply_move(state, inverse)
                if not state.solved():
                    logger.critical(f"Move validation failed for {move} -> {inverse}")
                    raise RuntimeError(f"Engine corrupt: {move} fails validation.")
                logger.debug(f"PASS: {move} -> {inverse}")
        logger.info("All startup validations passed")


# ========================================================================
# 6. SolutionPlayer
# ========================================================================

class SolutionPlayer:
    """Owns the ordered move list and current step index for Playback mode."""

    def __init__(self) -> None:
        self.moves: List[str] = []
        self.step_index: int = 0
        self._history: List[CubeState] = []
        self._current_move: Optional[str] = None

    def load(self, moves: List[str], initial_state: CubeState) -> None:
        """Load a sequence of moves and save the starting state."""
        self.moves = moves
        self.step_index = 0
        self._history = [initial_state.copy()]
        self._current_move = None
        logger.info(f"Solution loaded with {len(self.moves)} steps.")

    def step_forward(self, current_state: CubeState) -> Optional[str]:
        """Apply the next move, saving state to history."""
        if self.is_at_end:
            return None
        move = self.moves[self.step_index]
        self._history.append(current_state.copy())
        current_state.apply_move(move)
        self._current_move = move
        self.step_index += 1
        logger.info(f"Step forward: {move} ({self.progress})")
        return move

    def step_back(self, current_state: CubeState) -> Optional[str]:
        """Undo the last move by popping from history."""
        if self.is_at_start:
            return None
        self.step_index -= 1
        previous_state = self._history.pop()
        # Restore state dict values without breaking object reference
        current_state.state = previous_state.state
        # For highlight purposes, calculate what the inverse move would be
        prev_move = self.moves[self.step_index]
        inv_move = prev_move[0] + ("" if len(prev_move)>1 and prev_move[1]=="'" else "'" if len(prev_move)==1 else "2")
        self._current_move = inv_move
        logger.info(f"Step back: {inv_move} ({self.progress})")
        return inv_move

    def clear(self) -> None:
        """Clear the current solution."""
        self.moves = []
        self.step_index = 0
        self._history = []
        self._current_move = None
        logger.debug("Solution cleared.")

    @property
    def is_at_end(self) -> bool:
        return self.step_index >= len(self.moves)

    @property
    def is_at_start(self) -> bool:
        return self.step_index <= 0

    @property
    def current_move(self) -> Optional[str]:
        return self._current_move

    @property
    def progress(self) -> str:
        if not self.moves:
            return "0 / 0"
        return f"{self.step_index} / {len(self.moves)}"


# ========================================================================
# 7. Cube3DRenderer
# ========================================================================

class ClickTarget:
    """Stores data for sticker hit-testing."""
    def __init__(self, face: str, idx: int, z_depth: float, poly: List[Tuple[float, float]]):
        self.face = face
        self.idx = idx
        self.z_depth = z_depth
        self.poly = poly

class Cube3DRenderer:
    """Handles PyGame / 3-D projection logic for the Rubik's Cube using NumPy."""

    def __init__(self) -> None:
        # Default view: Up, Front, Right faces visible
        self.yaw: float = -math.pi / 4
        self.pitch: float = -math.pi / 6
        self.scale: float = 1.0

        self.faces: List[str] = ["U", "D", "F", "B", "L", "R"]
        self.click_targets: List[ClickTarget] = []

        # base_points shape: (6 faces, 9 stickers, 4 corners, 3 coordinates)
        self.base_points: np.ndarray = np.zeros((6, 9, 4, 3))
        self._init_3d_geometry()

        self.font_move = pygame.font.SysFont(None, 64)

    def _init_3d_geometry(self) -> None:
        """
        Analytically compute local 3D coordinates for all 54 stickers.
        Cube is centered at (0,0,0). +X is Right, +Y is Down, +Z is Front (toward viewer).
        """
        S = STICKER_PX
        G = GAP_PX
        offset = S + G
        grid = [-offset, 0, offset]

        def make_face(normal_axis: int, normal_dir: int,
                      right_axis: int, right_dir: int,
                      down_axis: int, down_dir: int) -> np.ndarray:
            pts = np.zeros((9, 4, 3))
            for i in range(9):
                row, col = divmod(i, 3)
                center = [0.0, 0.0, 0.0]
                center[right_axis] = grid[col] * right_dir
                center[down_axis] = grid[row] * down_dir
                center[normal_axis] = (offset * 1.5) * normal_dir

                # 4 corners counter-clockwise
                corners_2d = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
                for c_idx, (dx, dy) in enumerate(corners_2d):
                    corner = list(center)
                    corner[right_axis] += dx * S * right_dir
                    corner[down_axis] += dy * S * down_dir
                    pts[i, c_idx] = corner
            return pts

        # The order of faces and their orientations are critical and must match 
        # the CubeState's face definitions and the Kociemba string mapping.
        # U: Normal -Y, Right +X, Down +Z
        self.base_points[0] = make_face(1, -1,  0, 1,  2, 1)
        # D: Normal +Y, Right +X, Down -Z
        self.base_points[1] = make_face(1,  1,  0, 1,  2, -1)
        # F: Normal +Z, Right +X, Down +Y
        self.base_points[2] = make_face(2,  1,  0, 1,  1, 1)
        # B: Normal -Z, Right -X, Down +Y
        self.base_points[3] = make_face(2, -1,  0, -1, 1, 1)
        # L: Normal -X, Right -Z, Down +Y
        self.base_points[4] = make_face(0, -1,  2, 1, 1, 1)
        # R: Normal +X, Right +Z, Down +Y
        self.base_points[5] = make_face(0,  1,  2, -1,  1, 1)

    def render(self, surface: pygame.Surface, state: CubeState,
               hover_target: Optional[Tuple[str, int]],
               playback_highlight: List[Tuple[str, int]],
               current_move_text: Optional[str],
               move_text_alpha: int) -> None:
        """Projects and draws the 3D cube onto the 2D surface."""
        # 1. Math Setup & Rotation Matrix
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)

        Ry = np.array([
            [cy,  0, sy],
            [0,   1,  0],
            [-sy, 0, cy]
        ])
        Rp = np.array([
            [1,  0,   0],
            [0,  cp, -sp],
            [0,  sp,  cp]
        ])
        R_matrix = Rp @ Ry

        # 2. Vectorized Transform
        # shape: (6, 9, 4, 3)
        rotated = self.base_points @ R_matrix.T

        # Calculate Z depths of centers (axis 2 is Z, index 2)
        # +Z is toward viewer, so sort ascending to draw back-to-front
        centers_z = rotated.mean(axis=2)[..., 2]

        cx_screen = WINDOW_WIDTH // 2
        cy_screen = WINDOW_HEIGHT // 2

        # Move cube slightly left if in playback to make room for scroll box
        if playback_highlight or current_move_text:
            cx_screen -= 80

        # Flatten and sort polygons
        poly_list = []
        for f_idx in range(6):
            face = self.faces[f_idx]
            for s_idx in range(9):
                z_depth = centers_z[f_idx, s_idx]
                poly_3d = rotated[f_idx, s_idx]

                # Orthographic projection with scale
                poly_2d = []
                for pt in poly_3d:
                    px = cx_screen + pt[0] * self.scale
                    py = cy_screen + pt[1] * self.scale
                    poly_2d.append((px, py))

                poly_list.append((z_depth, face, s_idx, poly_2d))

        # Sort back to front
        poly_list.sort(key=lambda item: item[0])

        self.click_targets = []
        highlight_layer = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

        # 3. Draw Stickers
        for z_depth, face, s_idx, poly_2d in poly_list:
            color_code = state.state[face][s_idx]
            rgb = COLORS.get(color_code, (255, 0, 255))

            # Hover Brightening
            if hover_target == (face, s_idx):
                rgb = tuple(min(255, c + 40) for c in rgb)

            # Filled parallelogram
            pygame.draw.polygon(surface, rgb, poly_2d)
            # Dark Border
            pygame.draw.polygon(surface, (20, 20, 20), poly_2d, 2)

            self.click_targets.append(ClickTarget(face, s_idx, z_depth, poly_2d))

            # Render Highlights (accumulate on alpha layer)
            if (face, s_idx) in playback_highlight:
                pygame.draw.polygon(highlight_layer, HIGHLIGHT_COLOR, poly_2d)

        # 4. Apply playback overlay and text
        if playback_highlight:
            surface.blit(highlight_layer, (0, 0))

        if current_move_text and move_text_alpha > 0:
            txt_surf = self.font_move.render(current_move_text, True, (255, 255, 255))
            txt_surf.set_alpha(move_text_alpha)
            tr = txt_surf.get_rect(center=(cx_screen, WINDOW_HEIGHT // 4))
            surface.blit(txt_surf, tr)

    def get_sticker_at_pos(self, x: int, y: int) -> Optional[Tuple[str, int]]:
        """Hit-test screen coordinates against drawn polygons. Reverse order = front-to-back."""
        for target in reversed(self.click_targets):
            if self._point_in_poly(x, y, target.poly):
                return target.face, target.idx
        return None

    @staticmethod
    def _point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
        """Ray-casting algorithm for point in convex polygon."""
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside


# ========================================================================
# 8. UIOverlay
# ========================================================================

class UIOverlay:
    """Manages buttons, labels, status bar, color popup, error toasts, and move scroll box."""

    def __init__(self) -> None:
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 24)
        self.font_small = pygame.font.SysFont(None, 20)

        # Bottom Bar Buttons
        self.buttons: Dict[str, pygame.Rect] = {}
        self.hovered_button: Optional[str] = None

        # Toast State
        self.toast_msg: Optional[str] = None
        self.toast_end_time: int = 0

        # Color Popup State
        self.popup_active: bool = False
        self.popup_pos: Tuple[int, int] = (0, 0)
        self.popup_target: Optional[Tuple[str, int]] = None
        # pre-calculate circle positions relative to popup center
        self.popup_circles: List[Tuple[str, int, int]] = []
        c_radius = COLOR_POPUP_RADIUS
        c_gap = COLOR_POPUP_GAP
        total_w = 6 * (c_radius * 2) + 5 * c_gap
        start_x = -total_w // 2 + c_radius
        for i, color_code in enumerate(["W", "Y", "R", "O", "B", "G"]):
            self.popup_circles.append((color_code, start_x + i * (c_radius * 2 + c_gap), 0))

        # Scroll Box Setup
        self.scroll_y = 0
        self.box_rect = pygame.Rect(0, 0, 0, 0)

        self._update_button_layout()

    def _update_button_layout(self) -> None:
        bx = BUTTON_PAD_X
        by = WINDOW_HEIGHT - STATUS_BAR_H - BUTTON_H - 10
        self.buttons["Solve"] = pygame.Rect(bx, by, BUTTON_MIN_W, BUTTON_H)
        
        cx = WINDOW_WIDTH // 2
        self.buttons["Next"] = pygame.Rect(cx - BUTTON_MIN_W - 5, by, BUTTON_MIN_W, BUTTON_H)
        self.buttons["Back"] = pygame.Rect(cx + 5, by, BUTTON_MIN_W, BUTTON_H)
        
        rx = WINDOW_WIDTH - BUTTON_PAD_X - BUTTON_MIN_W
        self.buttons["Reset"] = pygame.Rect(rx, by, BUTTON_MIN_W, BUTTON_H)

        # Update Scroll Box dimensions
        box_w = 220
        box_h = WINDOW_HEIGHT - STATUS_BAR_H - BUTTON_H - 60
        self.box_rect = pygame.Rect(WINDOW_WIDTH - box_w - 20, 20, box_w, box_h)

    def handle_resize(self) -> None:
        self._update_button_layout()

    def show_toast(self, msg: str) -> None:
        self.toast_msg = msg
        self.toast_end_time = pygame.time.get_ticks() + int(TOAST_DURATION_S * 1000)
        logger.warning(f"Toast displayed: {msg}")

    def clear_toast(self) -> None:
        self.toast_msg = None

    def open_color_popup(self, face: str, idx: int, pos: Tuple[int, int]) -> None:
        self.popup_active = True
        self.popup_target = (face, idx)
        self.popup_pos = pos
        logger.debug(f"Opened color popup at {pos} for sticker {face}:{idx}")

    def close_color_popup(self) -> None:
        self.popup_active = False
        self.popup_target = None

    def auto_scroll(self, step_index: int, total_moves: int) -> None:
        target_y = step_index * 35
        if target_y + self.scroll_y > self.box_rect.height - 40:
            self.scroll_y = self.box_rect.height - 40 - target_y
        elif target_y + self.scroll_y < 10:
            self.scroll_y = 10 - target_y
        max_scroll = 0
        min_scroll = min(0, self.box_rect.height - (total_moves * 35 + 20))
        self.scroll_y = max(min_scroll, min(max_scroll, self.scroll_y))

    def render(self, surface: pygame.Surface, state_mode: str, cube_solved: bool, player: SolutionPlayer) -> None:
        # 1. Status Bar
        bar_rect = pygame.Rect(0, WINDOW_HEIGHT - STATUS_BAR_H, WINDOW_WIDTH, STATUS_BAR_H)
        pygame.draw.rect(surface, (30, 30, 35), bar_rect)
        
        mode_text = f"{state_mode} MODE"
        if state_mode == "PLAYBACK":
            mode_text += f" — STEP {player.progress}"
        mt = self.font_small.render(mode_text, True, (200, 200, 200))
        surface.blit(mt, (10, WINDOW_HEIGHT - STATUS_BAR_H + 6))

        solved_color = (80, 255, 80) if cube_solved else (150, 150, 150)
        solved_text = "SOLVED ✓" if cube_solved else "NOT SOLVED"
        st = self.font_small.render(solved_text, True, solved_color)
        surface.blit(st, (WINDOW_WIDTH - st.get_width() - 10, WINDOW_HEIGHT - STATUS_BAR_H + 6))

        # 2. Buttons
        for name, rect in self.buttons.items():
            enabled = True
            if name == "Solve" and state_mode != "EDIT": enabled = False
            if name in ["Next", "Back"] and state_mode != "PLAYBACK": enabled = False

            if not enabled:
                color = BUTTON_COLOR_DISABLED
            elif name == self.hovered_button:
                color = BUTTON_COLOR_HOVER
            else:
                color = BUTTON_COLOR_NORMAL

            pygame.draw.rect(surface, color, rect, border_radius=BUTTON_RADIUS)
            pygame.draw.rect(surface, (100, 100, 110), rect, width=2, border_radius=BUTTON_RADIUS)
            
            t_color = (255, 255, 255) if enabled else (150, 150, 150)
            txt = self.font.render(name, True, t_color)
            surface.blit(txt, txt.get_rect(center=rect.center))

        # 3. Interactive Solution Scroll Box
        if state_mode == "PLAYBACK" and player.moves:
            box_surf = pygame.Surface((self.box_rect.w, self.box_rect.h), pygame.SRCALPHA)
            pygame.draw.rect(box_surf, (30, 30, 35, 220), box_surf.get_rect(), border_radius=8)
            pygame.draw.rect(box_surf, (100, 100, 110), box_surf.get_rect(), width=2, border_radius=8)
            
            y_offset = 15 + self.scroll_y
            for i, move in enumerate(player.moves):
                color = (120, 120, 120)       # Gray for past moves
                if i == player.step_index - 1:
                    color = (40, 255, 80)     # Bright green for CURRENT move
                elif i == player.step_index:
                    color = (255, 255, 255)   # White for NEXT move
                
                txt = self.font.render(f"{i+1}.   {move}", True, color)
                box_surf.blit(txt, (25, y_offset))
                y_offset += 35
                
            surface.blit(box_surf, self.box_rect)

        # 4. Color Popup
        if self.popup_active:
            px, py = self.popup_pos
            bg_rect = pygame.Rect(0, 0, 6 * (COLOR_POPUP_RADIUS * 2) + 5 * COLOR_POPUP_GAP + 20, COLOR_POPUP_RADIUS * 2 + 20)
            bg_rect.center = (px, py)
            # clamp popup to screen
            bg_rect.clamp_ip(surface.get_rect())
            pygame.draw.rect(surface, (40, 40, 40), bg_rect, border_radius=10)
            pygame.draw.rect(surface, (100, 100, 100), bg_rect, width=2, border_radius=10)

            # Render circles
            for c_code, dx, dy in self.popup_circles:
                cx, cy = bg_rect.centerx + dx, bg_rect.centery + dy
                pygame.draw.circle(surface, COLORS[c_code], (cx, cy), COLOR_POPUP_RADIUS)

        # 5. Error Toast
        if self.toast_msg and pygame.time.get_ticks() < self.toast_end_time:
            tw = int(WINDOW_WIDTH * 0.8)
            th = 60
            toast_surf = pygame.Surface((tw, th), pygame.SRCALPHA)
            toast_surf.fill((220, 50, 50, 230))
            
            tt = self.font.render(self.toast_msg, True, (255, 255, 255))
            toast_surf.blit(tt, tt.get_rect(center=(tw // 2, th // 2)))
            
            surface.blit(toast_surf, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.1))
        elif self.toast_msg:
            self.toast_msg = None


# ========================================================================
# 9. InputHandler
# ========================================================================

class InputHandler:
    """Routes mouse / keyboard events to App / state controllers."""

    def __init__(self, app: "App") -> None:
        self.app = app
        self.drag_active = False

    def handle(self, event: pygame.event.Event) -> None:
        if event.type == pygame.VIDEORESIZE:
            global WINDOW_WIDTH, WINDOW_HEIGHT
            WINDOW_WIDTH, WINDOW_HEIGHT = event.w, event.h
            self.app.ui.handle_resize()
            logger.info(f"Window resized to {WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        # Any key or click dismisses toast immediately
        if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
            self.app.ui.clear_toast()

        if event.type == pygame.KEYDOWN:
            self._handle_keydown(event)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_mousedown(event)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.drag_active = False

        elif event.type == pygame.MOUSEMOTION:
            self._handle_mousemotion(event)

        elif event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            # Scroll box logic vs Zoom logic
            if self.app.mode == "PLAYBACK" and self.app.ui.box_rect.collidepoint(mx, my):
                self.app.ui.scroll_y += event.y * 25
                moves_len = len(self.app.player.moves)
                max_sc = 0
                min_sc = min(0, self.app.ui.box_rect.height - (moves_len * 35 + 20))
                self.app.ui.scroll_y = max(min_sc, min(max_sc, self.app.ui.scroll_y))
            else:
                self.app.renderer.scale += event.y * 0.1
                self.app.renderer.scale = max(0.9, min(2.0, self.app.renderer.scale))
                logger.debug(f"Zoomed to {self.app.renderer.scale:.2f}")

    def _handle_keydown(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_ESCAPE:
            if self.app.ui.popup_active:
                self.app.ui.close_color_popup()
            elif self.app.mode == "PLAYBACK":
                self.app._enter_edit_mode()
        elif event.key == pygame.K_RIGHT:
            self.app._action_next()
        elif event.key == pygame.K_LEFT:
            self.app._action_back()
        elif event.key == pygame.K_r:
            self.app._action_reset()

    def _handle_mousedown(self, event: pygame.event.Event) -> None:
        mx, my = event.pos

        # Check Color Popup
        if self.app.ui.popup_active:
            if event.button == 1:
                px, py = self.app.ui.popup_pos
                bg_rect = pygame.Rect(0, 0, 6 * (COLOR_POPUP_RADIUS * 2) + 5 * COLOR_POPUP_GAP + 20, COLOR_POPUP_RADIUS * 2 + 20)
                bg_rect.center = (px, py)
                bg_rect.clamp_ip(pygame.display.get_surface().get_rect())
                
                if bg_rect.collidepoint(mx, my):
                    # Check circles
                    for c_code, dx, dy in self.app.ui.popup_circles:
                        cx, cy = bg_rect.centerx + dx, bg_rect.centery + dy
                        if math.hypot(mx - cx, my - cy) <= COLOR_POPUP_RADIUS:
                            face, idx = self.app.ui.popup_target
                            self.app.cube.state[face][idx] = c_code
                            logger.debug(f"Set sticker {face}:{idx} to {c_code}")
                            self.app.ui.close_color_popup()
                            return
                else:
                    self.app.ui.close_color_popup()
            return

        # Check Buttons
        for name, rect in self.app.ui.buttons.items():
            if rect.collidepoint(mx, my):
                if event.button == 1:
                    if name == "Solve" and self.app.mode == "EDIT":
                        self.app._action_solve()
                    elif name == "Next" and self.app.mode == "PLAYBACK":
                        self.app._action_next()
                    elif name == "Back" and self.app.mode == "PLAYBACK":
                        self.app._action_back()
                    elif name == "Reset":
                        self.app._action_reset()
                return

        # Check Stickers
        hit = self.app.renderer.get_sticker_at_pos(mx, my)
        if hit and self.app.mode == "EDIT":
            face, idx = hit
            if event.button == 1:
                self.app.ui.open_color_popup(face, idx, (mx, my))
            elif event.button == 3:
                # Right click -> reset to default
                self.app.cube.state[face][idx] = FACE_DEFAULTS[face]
                logger.debug(f"Reset sticker {face}:{idx} to {FACE_DEFAULTS[face]}")
        elif not hit and event.button == 1:
            self.drag_active = True

    def _handle_mousemotion(self, event: pygame.event.Event) -> None:
        mx, my = event.pos
        
        # Hover button state
        self.app.ui.hovered_button = None
        for name, rect in self.app.ui.buttons.items():
            if rect.collidepoint(mx, my):
                self.app.ui.hovered_button = name

        # Hover sticker state
        if not self.app.ui.popup_active and self.app.mode == "EDIT":
            self.app.hover_target = self.app.renderer.get_sticker_at_pos(mx, my)
        else:
            self.app.hover_target = None

        # Drag rotation
        if self.drag_active and not self.app.ui.popup_active:
            dx, dy = event.rel
            self.app.renderer.yaw -= dx * 0.01
            self.app.renderer.pitch += dy * 0.01
            # clamp pitch to avoid flipping
            self.app.renderer.pitch = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, self.app.renderer.pitch))


# ========================================================================
# 10. App
# ========================================================================

class App:
    """Main loop, state transitions, threading, and rendering control."""

    def __init__(self) -> None:
        logger.info("Initializing 3x3 application...")
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("3x3 Rubik's Cube Solver")
        self.clock = pygame.time.Clock()

        # Startup Validations
        MoveExecutor.validate_all_moves()
        self._test_cube_state()

        self.cube = CubeState()
        self.renderer = Cube3DRenderer()
        self.ui = UIOverlay()
        self.input_handler = InputHandler(self)
        self.player = SolutionPlayer()
        self.bridge = KociembaBridge()
        self.move_executor = MoveExecutor()

        self.running = True
        self.mode = "EDIT"  # EDIT, SOLVING, PLAYBACK
        self.hover_target: Optional[Tuple[str, int]] = None
        
        # Threading state
        self.solve_thread: Optional[threading.Thread] = None
        self.solve_result: Optional[List[str]] = None
        self.solve_error: Optional[str] = None

        # Playback animation state
        self.move_text_timer: int = 0
        self.move_text_alpha: int = 0
        self.current_move_text: Optional[str] = None
        self.highlight_stickers: List[Tuple[str, int]] = []

    def _test_cube_state(self) -> None:
        """Self-test logic: start solved → apply 'R' → apply 'R'' → assert solved."""
        test_cube = CubeState()
        test_cube.apply_move("R")
        test_cube.apply_move("R'")
        if not test_cube.solved():
            raise RuntimeError("CubeState self-test failed.")
        logger.debug("CubeState self-test passed.")

    def _enter_edit_mode(self) -> None:
        self.mode = "EDIT"
        self.player.clear()
        self.highlight_stickers = []
        self.current_move_text = None
        logger.info("Transitioned to EDIT mode.")

    def _action_solve(self) -> None:
        is_valid, msg = self.cube.is_valid()
        if not is_valid:
            self.ui.show_toast(msg)
            return

        if self.cube.solved():
            self.ui.show_toast("Cube is already solved!")
            return

        self.mode = "SOLVING"
        self.solve_result = None
        self.solve_error = None
        cube_copy = self.cube.copy()

        def thread_target() -> None:
            try:
                moves = self.bridge.solve(cube_copy)
                self.solve_result = moves
            except Exception as e:
                self.solve_error = str(e)

        self.solve_thread = threading.Thread(target=thread_target)
        self.solve_thread.start()
        logger.info("Transitioned to SOLVING mode (background thread started).")

    def _action_next(self) -> None:
        move = self.player.step_forward(self.cube)
        if move:
            self._trigger_move_animation(move)
            self.ui.auto_scroll(self.player.step_index, len(self.player.moves))

    def _action_back(self) -> None:
        move = self.player.step_back(self.cube)
        if move:
            self._trigger_move_animation(move)
            self.ui.auto_scroll(max(0, self.player.step_index - 1), len(self.player.moves))

    def _action_reset(self) -> None:
        self.cube.reset()
        self._enter_edit_mode()
        self.ui.close_color_popup()

    def _trigger_move_animation(self, move: str) -> None:
        self.current_move_text = move
        self.move_text_timer = pygame.time.get_ticks() + 800
        self.move_text_alpha = 255
        self.highlight_stickers = self.move_executor.get_affected_stickers(move)

    def update(self) -> None:
        # Check thread results
        if self.mode == "SOLVING":
            if self.solve_error:
                self.ui.show_toast(self.solve_error)
                self.mode = "EDIT"
                logger.info("Transitioned to EDIT mode (solver failed).")
            elif self.solve_result is not None:
                self.player.load(self.solve_result, self.cube)
                self.ui.scroll_y = 0
                self.mode = "PLAYBACK"
                logger.info("Transitioned to PLAYBACK mode.")

        # Animation updates
        if self.mode == "PLAYBACK" and self.current_move_text:
            now = pygame.time.get_ticks()
            left = self.move_text_timer - now
            if left > 0:
                self.move_text_alpha = min(255, int((left / 800) * 255 * 1.5))
            else:
                self.move_text_alpha = 0
                self.highlight_stickers = []

    def draw(self) -> None:
        try:
            self.screen.fill((0, 0, 0))

            # 3D Render
            self.renderer.render(
                surface=self.screen,
                state=self.cube,
                hover_target=self.hover_target,
                playback_highlight=self.highlight_stickers,
                current_move_text=self.current_move_text,
                move_text_alpha=self.move_text_alpha
            )

            # Spinner for solving
            if self.mode == "SOLVING":
                cx, cy = WINDOW_WIDTH // 2, 80
                angle = pygame.time.get_ticks() * 0.005
                x2 = cx + math.cos(angle) * 20
                y2 = cy + math.sin(angle) * 20
                pygame.draw.line(self.screen, (255, 255, 255), (cx, cy), (x2, y2), 4)

            # Draw currently assigned color ring in popup if active
            if self.ui.popup_active and self.ui.popup_target:
                face, idx = self.ui.popup_target
                current_color = self.cube.state[face][idx]
                bg_rect = pygame.Rect(0, 0, 6 * (COLOR_POPUP_RADIUS * 2) + 5 * COLOR_POPUP_GAP + 20, COLOR_POPUP_RADIUS * 2 + 20)
                bg_rect.center = self.ui.popup_pos
                bg_rect.clamp_ip(self.screen.get_rect())
                for c_code, dx, dy in self.ui.popup_circles:
                    if c_code == current_color:
                        cx, cy = bg_rect.centerx + dx, bg_rect.centery + dy
                        pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), COLOR_POPUP_RADIUS + 2, 2)

            # UI Overlay
            self.ui.render(
                surface=self.screen,
                state_mode=self.mode,
                cube_solved=self.cube.solved(),
                player=self.player
            )

            pygame.display.flip()

        except pygame.error as e:
            logger.error(f"Render error: {e}")

    def run(self) -> None:
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    else:
                        self.input_handler.handle(event)

                self.update()
                self.draw()
                self.clock.tick(FPS_LIMIT)
        except Exception as e:
            logger.critical(f"Fatal crash in main loop: {e}", exc_info=True)
            raise
        finally:
            pygame.quit()
            logger.info("Application closed cleanly.")


# ========================================================================
# 11. if __name__ == "__main__": entry point
# ========================================================================

if __name__ == "__main__":
    app = App()
    app.run()