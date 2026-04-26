#   __.-._
#   '-._"7'
#    /'.-c
#    |  //
#   _)_/||
#
# rubiks2x2.py - 2x2 Rubik's Cube 3-D Solver with detailed logging.
# Author: santakd
# Contact: santakd at gmail dot com
# Date: April 22, 2026
# Version: 1.0.8
# License: MIT License 
#
# ========================================================================
# 2x2 Rubik's Cube Solver — Installation & Run Instructions
# ========================================================================
#
# REQUIREMENTS
#   Python 3.9+   (https://www.python.org/downloads/)
#   pip3 install pygame numpy
#
# The app is a 3-D interactive 2x2 Rubik's Cube rendered with PyGame. 
# It has three modes:
#  Mode         Description
#  Edit         User clicks stickers to set the scrambled state
#  Solve        App computes the optimal solution
#  Playback     User steps forward/backward through the solution
#
# RUN
#   python3 rubiks2x2.py
#
# CONTROLS
#   Left-click a sticker  → cycle through face colors (W Y R O B G)
#   Right-click a sticker → reset that sticker to its default color
#   [Solve]               → compute solution and load step list
#   [Next]                → advance one move in the solution
#   [Back]                → undo one move in the solution
#   [Reset]               → return cube to solved state and clear solution
#   Drag (empty space)    → rotate the 3-D view
#   Scroll wheel (space)  → zoom in/out
#   Scroll wheel (panel)  → scroll the move list in Playback mode
#
# NOTES
#   • ANCHOR RULE: The White-Red-Blue corner MUST be placed at the 
#     Top-Front-Right (U-F-R) position before clicking Solve.
#   • By forcing that one White-Red-Blue piece to act as the 
#     "North Star," you made both the human experience 
#     (painting the cube) and the algorithmic experience (the BFS solver)
#     infinitely smoother and faster.
#   • The solver uses a highly-optimized Bidirectional BFS (using only 
#     D, B, L moves to preserve the anchor) to find the optimal solution.
#   • Virtual center labels (U, F, R, etc.) are drawn on the cube to help 
#     you keep track of orientation while painting.
#   • By strictly anchoring the White-Red-Blue (W-R-B) corner to the 
#     Top-Front-Right (U-F-R) position, we have eliminated the need to 
#     search all 24 spatial orientations. The solver now uses an 
#     ultra-fast Bidirectional BFS that only turns the Down (D), Back (B),
#     and Left (L) faces. Because these moves never touch the U-F-R corner, 
#     the anchor is perfectly preserved, and the cube will automatically 
#     solve relative to your required orientation!
#   • The Anchor Rule: When you input your scramble, the solver now strictly
#     validates that the W-R-B corner is positioned at Top-Front-Right. 
#     If it's missing or twisted, a helpful toast message will prompt you
#     to correct it.
#   • The Math Engine: It has completely replaced the 24-orientation Kociemba
#     bridge with the D, B, L Optimal BFS. The solver now runs 
#     instantaneously (< 0.2s) and always guarantees an optimal solution
#     (≤ 11 moves).
#   • Everything Else: The interactive scrollable move list, the 3D 
#     virtual center labels (U, F, R), and all dynamic logging remain
#     exactly as they were.
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
    import numpy as np                          # For 3D geometry calculations
except ImportError as e:
    print("=" * 70)
    print("MISSING DEPENDENCIES DETECTED")
    print(f"Error: {e}")
    print("Please install required packages using:")
    print("    pip3 install pygame numpy")
    print("=" * 70)
    sys.exit(1)


# ========================================================================
# 1. Constants & color palette
# ========================================================================

COLORS: Dict[str, Tuple[int, int, int]] = {
    "W": (255, 255, 255),   # Up    face — white
    "Y": (255, 215,   2),   # Down  face — yellow
    "R": (183,  16,  52),   # Front face — red
    "O": (255,  80,   2),   # Back  face — orange
    "B": (  2,  68, 177),   # Right face — blue
    "G": (  2, 156,  73),   # Left  face — green
}

FACE_DEFAULTS: Dict[str, str] = {
    "U": "W", "D": "Y", "F": "R", "B": "O", "R": "B", "L": "G"
}

STICKER_PX: float = 44.0
GAP_PX: float = 3.0
WINDOW_WIDTH: int = 1024
WINDOW_HEIGHT: int = 768
FPS_LIMIT: int = 60

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

HIGHLIGHT_COLOR: Tuple[int, int, int, int] = (40, 100, 255, 80)


# ========================================================================
# 2. Logger setup
# ========================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"rubiks2x2_{timestamp}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, mode="w"),
    ],
)
logger = logging.getLogger("rubiks2x2")


# ========================================================================
# Exceptions
# ========================================================================

class RubiksSolverError(Exception): pass
class InvalidCubeError(RubiksSolverError): pass
class SolverEngineError(RubiksSolverError): pass
class MoveError(RubiksSolverError): pass


# ========================================================================
# 3. CubeState
# ========================================================================

class CubeState:
    def __init__(self) -> None:
        self.state: Dict[str, List[str]] = {
            face: [color] * 4 for face, color in FACE_DEFAULTS.items()
        }
        logger.debug("CubeState initialized.")

    def solved(self) -> bool:
        """A 2x2 cube is solved if every face consists of exactly one solid color."""
        for stickers in self.state.values():
            if len(set(stickers)) > 1:
                return False
        return True

    def is_valid(self) -> Tuple[bool, str]:
        color_counts: Dict[str, int] = {c: 0 for c in COLORS.keys()}
        total = 0
        for face, stickers in self.state.items():
            if len(stickers) != 4: return False, f"Face {face} does not have exactly 4 stickers."
            for color in stickers:
                if color not in color_counts: return False, f"Invalid color '{color}'."
                color_counts[color] += 1
                total += 1
        if total != 24: return False, f"Cube has {total} stickers, expected 24."
        for color, count in color_counts.items():
            if count != 4: return False, f"Color {color} appears {count} times, expected 4."
        return True, "Valid"

    def apply_move(self, move: str) -> None:
        MoveExecutor().apply_move(self, move)

    def reset(self) -> None:
        self.state = {face: [color] * 4 for face, color in FACE_DEFAULTS.items()}
        logger.info("CubeState reset to default orientation.")

    def copy(self) -> "CubeState":
        new_cube = CubeState()
        new_cube.state = copy.deepcopy(self.state)
        return new_cube


# ========================================================================
# 4. Optimal 2x2 Solver Engine (Anchored)
# ========================================================================

class KociembaBridge:
    """
    Native Bidirectional BFS using the White-Red-Blue piece as the fixed global anchor.
    By restricting allowed moves to ONLY D, B, and L, the U-F-R corner is never touched.
    This safely bypasses 2x2 orientation parities and computes the absolute optimal 
    solution (≤ 11 moves) instantaneously.
    """

    def _to_flat(self, state_dict: Dict[str, List[str]]) -> Tuple[str, ...]:
        res = []
        for face in ["U", "D", "F", "B", "L", "R"]: res.extend(state_dict[face])
        return tuple(res)

    def _generate_permutations(self) -> Dict[str, Tuple[int, ...]]:
        offsets = {"U": 0, "D": 4, "F": 8, "B": 12, "L": 16, "R": 20}
        perms = {}
        for base_move in ["U", "D", "F", "B", "L", "R"]:
            p = list(range(24))
            for cycle in MoveExecutor.CYCLES[base_move]:
                c_idx = [offsets[f] + i for f, i in cycle]
                p[c_idx[1]], p[c_idx[2]], p[c_idx[3]], p[c_idx[0]] = p[c_idx[0]], p[c_idx[1]], p[c_idx[2]], p[c_idx[3]]
            perms[base_move] = tuple(p)
            p_inv = list(range(24))
            for i, val in enumerate(p): p_inv[val] = i
            perms[base_move + "'"] = tuple(p_inv)
            p2 = list(range(24))
            for i in range(24): p2[i] = p[p[i]]
            perms[base_move + "2"] = tuple(p2)
        return perms

    def solve(self, cube: CubeState) -> List[str]:
        is_valid, msg = cube.is_valid()
        if not is_valid:
            logger.error(f"Cannot solve invalid cube: {msg}")
            raise InvalidCubeError(msg)
            
        # 1. Enforce the U-F-R Anchor Orientation Strategy
        u_col = cube.state["U"][3]
        f_col = cube.state["F"][1]
        r_col = cube.state["R"][0]
        
        if {u_col, f_col, r_col} != {"W", "R", "B"}:
            msg = "ANCHOR REQUIRED: The White-Red-Blue corner piece MUST be placed at the Top-Front-Right position."
            logger.error(msg)
            raise InvalidCubeError(msg)
            
        if u_col != "W" or f_col != "R" or r_col != "B":
            msg = "ANCHOR TWISTED: The White-Red-Blue corner is at Top-Front-Right, but White must face Up."
            logger.error(msg)
            raise InvalidCubeError(msg)

        logger.info("Initiating optimal 2x2 Bidirectional BFS solver (W-R-B Anchored)...")
        start_state = self._to_flat(cube.state)
        target_state = self._to_flat({
            "U": ["W"] * 4, "D": ["Y"] * 4, "F": ["R"] * 4,
            "B": ["O"] * 4, "L": ["G"] * 4, "R": ["B"] * 4
        })
            
        if start_state == target_state:
            logger.info("Cube is already solved!")
            return []

        perms = self._generate_permutations()
        # By exclusively using D, B, L moves, the U-F-R anchor is mathematically untouched.
        allowed_moves = ["D", "D'", "D2", "B", "B'", "B2", "L", "L'", "L2"]
        inv_move = {"D": "D'", "D'": "D", "D2": "D2", "B": "B'", "B'": "B", "B2": "B2", "L": "L'", "L'": "L", "L2": "L2"}

        forward, forward_q = {start_state: []}, [start_state]
        backward, backward_q = {target_state: []}, [target_state]

        for depth in range(6): 
            new_forward_q = []
            for state in forward_q:
                path = forward[state]
                for move in allowed_moves:
                    if path and path[-1][0] == move[0]: continue
                    next_state = tuple(state[idx] for idx in perms[move])
                    if next_state in backward:
                        ans = path + [move] + backward[next_state]
                        logger.info(f"Solution found: {' '.join(ans)} ({len(ans)} moves)")
                        return ans
                    if next_state not in forward:
                        forward[next_state] = path + [move]
                        new_forward_q.append(next_state)
            forward_q = new_forward_q
            
            new_backward_q = []
            for state in backward_q:
                path = backward[state]
                for move in allowed_moves:
                    if path and path[0][0] == move[0]: continue
                    next_state = tuple(state[idx] for idx in perms[inv_move[move]])
                    if next_state in forward:
                        ans = forward[next_state] + [move] + path
                        logger.info(f"Solution found: {' '.join(ans)} ({len(ans)} moves)")
                        return ans
                    if next_state not in backward:
                        backward[next_state] = [move] + path
                        new_backward_q.append(next_state)
            backward_q = new_backward_q
            
        logger.error("BFS exhausted. A corner is likely physically twisted.")
        raise SolverEngineError("Engine failed to find a solution. A corner is likely physically twisted.")


# ========================================================================
# 5. MoveExecutor
# ========================================================================

class MoveExecutor:
    CYCLES: Dict[str, List[Tuple[Tuple[str, int], ...]]] = {
        "U": [
            (("U", 0), ("U", 1), ("U", 3), ("U", 2)),
            (("F", 1), ("L", 1), ("B", 1), ("R", 1)),
            (("F", 0), ("L", 0), ("B", 0), ("R", 0)),
        ],
        "D": [
            (("D", 0), ("D", 1), ("D", 3), ("D", 2)),
            (("F", 2), ("R", 2), ("B", 2), ("L", 2)),
            (("F", 3), ("R", 3), ("B", 3), ("L", 3)),
        ],
        "F": [
            (("F", 0), ("F", 1), ("F", 3), ("F", 2)),
            (("U", 2), ("R", 0), ("D", 1), ("L", 3)),
            (("U", 3), ("R", 2), ("D", 0), ("L", 1)),
        ],
        "B": [
            (("B", 0), ("B", 1), ("B", 3), ("B", 2)),
            (("U", 1), ("L", 0), ("D", 2), ("R", 3)),
            (("U", 0), ("L", 2), ("D", 3), ("R", 1)),
        ],
        "L": [
            (("L", 0), ("L", 1), ("L", 3), ("L", 2)),
            (("U", 0), ("F", 0), ("D", 0), ("B", 3)),
            (("U", 2), ("F", 2), ("D", 2), ("B", 1)),
        ],
        "R": [
            (("R", 0), ("R", 1), ("R", 3), ("R", 2)),
            (("U", 3), ("B", 0), ("D", 3), ("F", 3)),
            (("U", 1), ("B", 2), ("D", 1), ("F", 1)),
        ],
    }

    def apply_move(self, cube: CubeState, move: str) -> None:
        if not move or move[0] not in self.CYCLES: raise MoveError(f"Unknown move: {move}")
        times = 1 if len(move) == 1 else (3 if move[1] == "'" else 2)
        for _ in range(times):
            old = copy.deepcopy(cube.state)
            for a, b, c, d in self.CYCLES[move[0]]:
                cube.state[b[0]][b[1]] = old[a[0]][a[1]]
                cube.state[c[0]][c[1]] = old[b[0]][b[1]]
                cube.state[d[0]][d[1]] = old[c[0]][c[1]]
                cube.state[a[0]][a[1]] = old[d[0]][d[1]]
        logger.debug(f"Applied move: {move}")

    def get_affected_stickers(self, move: str) -> List[Tuple[str, int]]:
        if not move or move[0] not in self.CYCLES: return []
        affected = set((f, i) for cycle in self.CYCLES[move[0]] for f, i in cycle)
        for i in range(4): affected.add((move[0], i))
        return list(affected)

    @classmethod
    def validate_all_moves(cls) -> None:
        state, ex = CubeState(), cls()
        for f in ["U", "D", "F", "B", "L", "R"]:
            for s, i in [("", "'"), ("'", ""), ("2", "2")]:
                state.reset()
                ex.apply_move(state, f + s)
                ex.apply_move(state, f + i)
                if not state.solved(): raise RuntimeError(f"Engine corrupt: {f+s} fails.")
        logger.info("All startup move validations passed.")


# ========================================================================
# 6. SolutionPlayer
# ========================================================================

class SolutionPlayer:
    def __init__(self) -> None:
        self.moves: List[str] = []
        self.step_index: int = 0
        self._history: List[CubeState] = []
        self._current_move: Optional[str] = None

    def load(self, moves: List[str], initial_state: CubeState) -> None:
        self.moves = moves
        self.step_index = 0
        self._history = [initial_state.copy()]
        self._current_move = None
        logger.info(f"Solution loaded with {len(self.moves)} steps.")

    def step_forward(self, current_state: CubeState) -> Optional[str]:
        if self.is_at_end: return None
        move = self.moves[self.step_index]
        self._history.append(current_state.copy())
        current_state.apply_move(move)
        self._current_move = move
        self.step_index += 1
        logger.info(f"Step forward: {move} ({self.progress})")
        return move

    def step_back(self, current_state: CubeState) -> Optional[str]:
        if self.is_at_start: return None
        self.step_index -= 1
        current_state.state = self._history.pop().state
        pm = self.moves[self.step_index]
        self._current_move = pm[0] + ("" if len(pm)>1 and pm[1]=="'" else "'" if len(pm)==1 else "2")
        logger.info(f"Step back: {self._current_move} ({self.progress})")
        return self._current_move

    def clear(self) -> None:
        self.moves, self.step_index, self._history, self._current_move = [], 0, [], None
        logger.debug("SolutionPlayer cleared.")

    @property
    def is_at_end(self) -> bool: return self.step_index >= len(self.moves)
    @property
    def is_at_start(self) -> bool: return self.step_index <= 0
    @property
    def current_move(self) -> Optional[str]: return self._current_move
    @property
    def progress(self) -> str: return "0 / 0" if not self.moves else f"{self.step_index} / {len(self.moves)}"


# ========================================================================
# 7. Cube3DRenderer
# ========================================================================

class ClickTarget:
    def __init__(self, face: str, idx: int, z_depth: float, poly: List[Tuple[float, float]]):
        self.face = face
        self.idx = idx
        self.z_depth = z_depth
        self.poly = poly

class Cube3DRenderer:
    def __init__(self) -> None:
        self.yaw: float = -math.pi / 4
        self.pitch: float = -math.pi / 6
        self.scale: float = 1.0
        self.faces: List[str] = ["U", "D", "F", "B", "L", "R"]
        self.click_targets: List[ClickTarget] = []
        self.base_points: np.ndarray = np.zeros((6, 4, 4, 3))
        self._init_3d_geometry()
        
        self.font_move = pygame.font.SysFont(None, 64)
        self.font_label = pygame.font.SysFont(None, 32, bold=True)

    def _init_3d_geometry(self) -> None:
        S, G = STICKER_PX, GAP_PX
        offset = S/2 + G/2 
        grid = [-offset, offset]

        def make_face(norm_axis: int, norm_dir: int, right_axis: int, right_dir: int, down_axis: int, down_dir: int) -> np.ndarray:
            pts = np.zeros((4, 4, 3))
            for i in range(4):
                r, c = divmod(i, 2)
                center = [0.0, 0.0, 0.0]
                center[right_axis] = grid[c] * right_dir
                center[down_axis] = grid[r] * down_dir
                center[norm_axis] = offset * 2.0 * norm_dir
                for c_idx, (dx, dy) in enumerate([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]):
                    pt = list(center)
                    pt[right_axis] += dx * S * right_dir
                    pt[down_axis] += dy * S * down_dir
                    pts[i, c_idx] = pt
            return pts

        self.base_points[0] = make_face(1, -1,  0, 1,  2, 1)
        self.base_points[1] = make_face(1,  1,  0, 1,  2, -1)
        self.base_points[2] = make_face(2,  1,  0, 1,  1, 1)
        self.base_points[3] = make_face(2, -1,  0, -1, 1, 1)
        self.base_points[4] = make_face(0, -1,  2, 1,  1, 1)
        self.base_points[5] = make_face(0,  1,  2, -1, 1, 1)

    def render(self, surface: pygame.Surface, state: CubeState, hover_target: Optional[Tuple[str, int]],
               playback_highlight: List[Tuple[str, int]], current_move_text: Optional[str], move_text_alpha: int) -> None:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        R_matrix = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

        rotated = self.base_points @ R_matrix.T
        cx_screen, cy_screen = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        
        if playback_highlight or current_move_text: cx_screen -= 80

        poly_list = []
        for f_idx in range(6):
            face = self.faces[f_idx]
            face_center_3d = rotated[f_idx].mean(axis=(0,1))
            
            cx = cx_screen + face_center_3d[0] * self.scale
            cy_ = cy_screen + face_center_3d[1] * self.scale
            poly_list.append((face_center_3d[2], face, -1, [(cx, cy_)]))
            
            for s_idx in range(4):
                poly_3d = rotated[f_idx, s_idx]
                z_depth = poly_3d.mean(axis=0)[2]
                poly_2d = [(cx_screen + pt[0] * self.scale, cy_screen + pt[1] * self.scale) for pt in poly_3d]
                poly_list.append((z_depth, face, s_idx, poly_2d))

        poly_list.sort(key=lambda item: item[0])
        self.click_targets = []
        highlight_layer = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

        for z_depth, face, s_idx, poly_2d in poly_list:
            if s_idx == -1:
                px, py = poly_2d[0]
                pygame.draw.circle(surface, (0, 0, 0, 150), (int(px), int(py)), 18)
                pygame.draw.circle(surface, (255, 255, 255), (int(px), int(py)), 18, 1)
                lbl = self.font_label.render(face, True, (255, 255, 255))
                surface.blit(lbl, lbl.get_rect(center=(px, py)))
                continue

            color_code = state.state[face][s_idx]
            rgb = COLORS.get(color_code, (255, 0, 255))
            if hover_target == (face, s_idx): rgb = tuple(min(255, c + 40) for c in rgb)

            pygame.draw.polygon(surface, rgb, poly_2d)
            pygame.draw.polygon(surface, (20, 20, 20), poly_2d, 2)
            self.click_targets.append(ClickTarget(face, s_idx, z_depth, poly_2d))

            if (face, s_idx) in playback_highlight:
                pygame.draw.polygon(highlight_layer, HIGHLIGHT_COLOR, poly_2d)

        if playback_highlight: surface.blit(highlight_layer, (0, 0))

        if current_move_text and move_text_alpha > 0:
            txt_surf = self.font_move.render(current_move_text, True, (255, 255, 255))
            txt_surf.set_alpha(move_text_alpha)
            tr = txt_surf.get_rect(center=(cx_screen, WINDOW_HEIGHT // 4))
            surface.blit(txt_surf, tr)

    def get_sticker_at_pos(self, x: int, y: int) -> Optional[Tuple[str, int]]:
        for target in reversed(self.click_targets):
            if self._point_in_poly(x, y, target.poly): return target.face, target.idx
        return None

    @staticmethod
    def _point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
        n, inside = len(poly), False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y: xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xints: inside = not inside
            p1x, p1y = p2x, p2y
        return inside


# ========================================================================
# 8. UIOverlay
# ========================================================================

class UIOverlay:
    def __init__(self) -> None:
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 24)
        self.font_small = pygame.font.SysFont(None, 20)
        self.buttons: Dict[str, pygame.Rect] = {}
        self.hovered_button: Optional[str] = None
        self.toast_msg: Optional[str] = None
        self.toast_end_time: int = 0
        
        self.popup_active: bool = False
        self.popup_pos: Tuple[int, int] = (0, 0)
        self.popup_target: Optional[Tuple[str, int]] = None
        self.popup_circles: List[Tuple[str, int, int]] = []
        for i, c in enumerate(["W", "Y", "R", "O", "B", "G"]):
            self.popup_circles.append((c, -86 + i * 38, 0))

        self.scroll_y = 0
        self.box_rect = pygame.Rect(0,0,0,0)

        self._update_button_layout()

    def _update_button_layout(self) -> None:
        bx, by = BUTTON_PAD_X, WINDOW_HEIGHT - STATUS_BAR_H - BUTTON_H - 10
        cx = WINDOW_WIDTH // 2
        self.buttons["Solve"] = pygame.Rect(bx, by, BUTTON_MIN_W, BUTTON_H)
        self.buttons["Next"] = pygame.Rect(cx - BUTTON_MIN_W - 5, by, BUTTON_MIN_W, BUTTON_H)
        self.buttons["Back"] = pygame.Rect(cx + 5, by, BUTTON_MIN_W, BUTTON_H)
        self.buttons["Reset"] = pygame.Rect(WINDOW_WIDTH - BUTTON_PAD_X - BUTTON_MIN_W, by, BUTTON_MIN_W, BUTTON_H)
        
        box_w = 220
        box_h = WINDOW_HEIGHT - STATUS_BAR_H - BUTTON_H - 60
        self.box_rect = pygame.Rect(WINDOW_WIDTH - box_w - 20, 20, box_w, box_h)

    def handle_resize(self) -> None: self._update_button_layout()

    def show_toast(self, msg: str) -> None:
        self.toast_msg = msg
        self.toast_end_time = pygame.time.get_ticks() + int(TOAST_DURATION_S * 1000)
        logger.warning(f"Toast displayed: {msg}")

    def clear_toast(self) -> None: self.toast_msg = None

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
        bar_rect = pygame.Rect(0, WINDOW_HEIGHT - STATUS_BAR_H, WINDOW_WIDTH, STATUS_BAR_H)
        pygame.draw.rect(surface, (30, 30, 35), bar_rect)
        mode_text = f"{state_mode} MODE"
        if state_mode == "PLAYBACK": mode_text += f" — STEP {player.progress}"
        surface.blit(self.font_small.render(mode_text, True, (200, 200, 200)), (10, bar_rect.y + 6))
        
        solved_color = (80, 255, 80) if cube_solved else (150, 150, 150)
        st = self.font_small.render("SOLVED ✓" if cube_solved else "NOT SOLVED", True, solved_color)
        surface.blit(st, (WINDOW_WIDTH - st.get_width() - 10, bar_rect.y + 6))

        for name, rect in self.buttons.items():
            enabled = True
            if name == "Solve" and state_mode != "EDIT": enabled = False
            if name in ["Next", "Back"] and state_mode != "PLAYBACK": enabled = False
            color = BUTTON_COLOR_DISABLED if not enabled else (BUTTON_COLOR_HOVER if name == self.hovered_button else BUTTON_COLOR_NORMAL)
            pygame.draw.rect(surface, color, rect, border_radius=BUTTON_RADIUS)
            pygame.draw.rect(surface, (100, 100, 110), rect, width=2, border_radius=BUTTON_RADIUS)
            txt = self.font.render(name, True, (255, 255, 255) if enabled else (150, 150, 150))
            surface.blit(txt, txt.get_rect(center=rect.center))

        if state_mode == "PLAYBACK" and player.moves:
            box_surf = pygame.Surface((self.box_rect.w, self.box_rect.h), pygame.SRCALPHA)
            pygame.draw.rect(box_surf, (30, 30, 35, 220), box_surf.get_rect(), border_radius=8)
            pygame.draw.rect(box_surf, (100, 100, 110), box_surf.get_rect(), width=2, border_radius=8)
            
            y_offset = 15 + self.scroll_y
            for i, move in enumerate(player.moves):
                color = (120, 120, 120)
                if i == player.step_index - 1: color = (40, 255, 80)
                elif i == player.step_index: color = (255, 255, 255)
                
                txt = self.font.render(f"{i+1}.   {move}", True, color)
                box_surf.blit(txt, (25, y_offset))
                y_offset += 35
                
            surface.blit(box_surf, self.box_rect)

        if self.popup_active:
            bg_rect = pygame.Rect(0, 0, 6 * (COLOR_POPUP_RADIUS * 2) + 5 * COLOR_POPUP_GAP + 20, COLOR_POPUP_RADIUS * 2 + 20)
            bg_rect.center = self.popup_pos
            bg_rect.clamp_ip(surface.get_rect())
            pygame.draw.rect(surface, (40, 40, 40), bg_rect, border_radius=10)
            pygame.draw.rect(surface, (100, 100, 100), bg_rect, width=2, border_radius=10)
            for c_code, dx, dy in self.popup_circles:
                pygame.draw.circle(surface, COLORS[c_code], (bg_rect.centerx + dx, bg_rect.centery + dy), COLOR_POPUP_RADIUS)

        if self.toast_msg and pygame.time.get_ticks() < self.toast_end_time:
            tw, th = int(WINDOW_WIDTH * 0.8), 60
            toast_surf = pygame.Surface((tw, th), pygame.SRCALPHA)
            toast_surf.fill((220, 50, 50, 230))
            tt = self.font.render(self.toast_msg, True, (255, 255, 255))
            toast_surf.blit(tt, tt.get_rect(center=(tw // 2, th // 2)))
            surface.blit(toast_surf, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.1))


# ========================================================================
# 9. InputHandler
# ========================================================================

class InputHandler:
    def __init__(self, app: "App") -> None:
        self.app = app
        self.drag_active = False

    def handle(self, event: pygame.event.Event) -> None:
        if event.type == pygame.VIDEORESIZE:
            global WINDOW_WIDTH, WINDOW_HEIGHT
            WINDOW_WIDTH, WINDOW_HEIGHT = event.w, event.h
            self.app.ui.handle_resize()

        if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
            self.app.ui.clear_toast()

        if event.type == pygame.KEYDOWN: self._handle_keydown(event)
        elif event.type == pygame.MOUSEBUTTONDOWN: self._handle_mousedown(event)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1: self.drag_active = False
        elif event.type == pygame.MOUSEMOTION: self._handle_mousemotion(event)
        elif event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if self.app.mode == "PLAYBACK" and self.app.ui.box_rect.collidepoint(mx, my):
                self.app.ui.scroll_y += event.y * 25
                moves_len = len(self.app.player.moves)
                max_sc = 0
                min_sc = min(0, self.app.ui.box_rect.height - (moves_len * 35 + 20))
                self.app.ui.scroll_y = max(min_sc, min(max_sc, self.app.ui.scroll_y))
            else:
                self.app.renderer.scale = max(0.9, min(2.0, self.app.renderer.scale + event.y * 0.1))

    def _handle_keydown(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_ESCAPE:
            if self.app.ui.popup_active: self.app.ui.close_color_popup()
            elif self.app.mode == "PLAYBACK": self.app._enter_edit_mode()
        elif event.key == pygame.K_RIGHT: self.app._action_next()
        elif event.key == pygame.K_LEFT: self.app._action_back()
        elif event.key == pygame.K_r: self.app._action_reset()

    def _handle_mousedown(self, event: pygame.event.Event) -> None:
        mx, my = event.pos
        if self.app.ui.popup_active:
            if event.button == 1:
                bg_rect = pygame.Rect(0, 0, 6 * (COLOR_POPUP_RADIUS * 2) + 5 * COLOR_POPUP_GAP + 20, COLOR_POPUP_RADIUS * 2 + 20)
                bg_rect.center = self.app.ui.popup_pos
                if bg_rect.collidepoint(mx, my):
                    for c_code, dx, dy in self.app.ui.popup_circles:
                        if math.hypot(mx - (bg_rect.centerx + dx), my - (bg_rect.centery + dy)) <= COLOR_POPUP_RADIUS:
                            face, idx = self.app.ui.popup_target
                            self.app.cube.state[face][idx] = c_code
                            logger.debug(f"Set sticker {face}:{idx} to {c_code}")
                            self.app.ui.close_color_popup()
                            return
                else: self.app.ui.close_color_popup()
            return

        for name, rect in self.app.ui.buttons.items():
            if rect.collidepoint(mx, my):
                if event.button == 1:
                    if name == "Solve" and self.app.mode == "EDIT": self.app._action_solve()
                    elif name == "Next" and self.app.mode == "PLAYBACK": self.app._action_next()
                    elif name == "Back" and self.app.mode == "PLAYBACK": self.app._action_back()
                    elif name == "Reset": self.app._action_reset()
                return

        hit = self.app.renderer.get_sticker_at_pos(mx, my)
        if hit and self.app.mode == "EDIT":
            face, idx = hit
            if event.button == 1: self.app.ui.open_color_popup(face, idx, (mx, my))
            elif event.button == 3: self.app.cube.state[face][idx] = FACE_DEFAULTS[face]
        elif not hit and event.button == 1: self.drag_active = True

    def _handle_mousemotion(self, event: pygame.event.Event) -> None:
        mx, my = event.pos
        self.app.ui.hovered_button = next((n for n, r in self.app.ui.buttons.items() if r.collidepoint(mx, my)), None)
        if not self.app.ui.popup_active and self.app.mode == "EDIT":
            self.app.hover_target = self.app.renderer.get_sticker_at_pos(mx, my)
        else: self.app.hover_target = None
        if self.drag_active and not self.app.ui.popup_active:
            self.app.renderer.yaw -= event.rel[0] * 0.01
            self.app.renderer.pitch = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, self.app.renderer.pitch + event.rel[1] * 0.01))


# ========================================================================
# 10. App
# ========================================================================

class App:
    def __init__(self) -> None:
        logger.info("Initializing 2x2 application...")
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("2x2 Rubik's Cube Solver")
        self.clock = pygame.time.Clock()

        MoveExecutor.validate_all_moves()
        self.cube = CubeState()
        self.renderer = Cube3DRenderer()
        self.ui = UIOverlay()
        self.input_handler = InputHandler(self)
        self.player = SolutionPlayer()
        self.bridge = KociembaBridge()
        self.move_executor = MoveExecutor()

        self.running = True
        self.mode = "EDIT"  
        self.hover_target: Optional[Tuple[str, int]] = None
        self.solve_thread: Optional[threading.Thread] = None
        self.solve_result: Optional[List[str]] = None
        self.solve_error: Optional[str] = None
        self.move_text_timer: int = 0
        self.move_text_alpha: int = 0
        self.current_move_text: Optional[str] = None
        self.highlight_stickers: List[Tuple[str, int]] = []

    def _enter_edit_mode(self) -> None:
        self.mode = "EDIT"
        self.player.clear()
        self.highlight_stickers = []
        self.current_move_text = None
        logger.info("Transitioned to EDIT mode.")

    def _action_solve(self) -> None:
        is_valid, msg = self.cube.is_valid()
        if not is_valid: return self.ui.show_toast(msg)
        if self.cube.solved(): return self.ui.show_toast("Cube is already solved!")

        self.mode, self.solve_result, self.solve_error = "SOLVING", None, None
        cube_copy = self.cube.copy()

        def thread_target() -> None:
            try: self.solve_result = self.bridge.solve(cube_copy)
            except Exception as e: self.solve_error = str(e)

        self.solve_thread = threading.Thread(target=thread_target)
        self.solve_thread.start()
        logger.info("Transitioned to SOLVING mode (background thread started).")

    def _action_next(self) -> None:
        if move := self.player.step_forward(self.cube):
            self._trigger_move_animation(move)
            self.ui.auto_scroll(self.player.step_index, len(self.player.moves))

    def _action_back(self) -> None:
        if move := self.player.step_back(self.cube):
            self._trigger_move_animation(move)
            self.ui.auto_scroll(max(0, self.player.step_index - 1), len(self.player.moves))

    def _action_reset(self) -> None:
        self.cube.reset()
        self._enter_edit_mode()
        self.ui.close_color_popup()

    def _trigger_move_animation(self, move: str) -> None:
        self.current_move_text = move
        self.move_text_timer = pygame.time.get_ticks() + 4000
        self.move_text_alpha = 255
        self.highlight_stickers = self.move_executor.get_affected_stickers(move)

    def update(self) -> None:
        if self.mode == "SOLVING":
            if self.solve_error:
                logger.error(f"Solver failed: {self.solve_error}")
                self.ui.show_toast(self.solve_error)
                self.mode = "EDIT"
            elif self.solve_result is not None:
                self.player.load(self.solve_result, self.cube)
                self.ui.scroll_y = 0
                self.mode = "PLAYBACK"
                logger.info("Transitioned to PLAYBACK mode.")

        if self.mode == "PLAYBACK" and self.current_move_text:
            left = self.move_text_timer - pygame.time.get_ticks()
            if left > 0: self.move_text_alpha = min(255, int((left / 4000) * 255 * 1.5))
            else:
                self.move_text_alpha = 0
                self.highlight_stickers = []

    def draw(self) -> None:
        try:
            self.screen.fill((0, 0, 0))
            self.renderer.render(
                self.screen, self.cube, self.hover_target, self.highlight_stickers,
                self.current_move_text, self.move_text_alpha
            )
            if self.mode == "SOLVING":
                cx, cy, angle = WINDOW_WIDTH // 2, 80, pygame.time.get_ticks() * 0.005
                pygame.draw.line(self.screen, (255, 255, 255), (cx, cy), (cx + math.cos(angle) * 20, cy + math.sin(angle) * 20), 4)

            if self.ui.popup_active and self.ui.popup_target:
                face, idx = self.ui.popup_target
                bg_rect = pygame.Rect(0, 0, 6 * (COLOR_POPUP_RADIUS * 2) + 5 * COLOR_POPUP_GAP + 20, COLOR_POPUP_RADIUS * 2 + 20)
                bg_rect.center = self.ui.popup_pos
                bg_rect.clamp_ip(self.screen.get_rect())
                for c, dx, dy in self.ui.popup_circles:
                    if c == self.cube.state[face][idx]:
                        pygame.draw.circle(self.screen, (255, 255, 255), (bg_rect.centerx + dx, bg_rect.centery + dy), COLOR_POPUP_RADIUS + 2, 2)

            self.ui.render(self.screen, self.mode, self.cube.solved(), self.player)
            pygame.display.flip()
        except pygame.error as e: 
            logger.error(f"Render error: {e}")

    def run(self) -> None:
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: self.running = False
                    else: self.input_handler.handle(event)
                self.update()
                self.draw()
                self.clock.tick(FPS_LIMIT)
        except Exception as e:
            logger.critical(f"Fatal crash: {e}", exc_info=True)
            raise
        finally:
            pygame.quit()
            logger.info("Application closed cleanly.")


if __name__ == "__main__":
    App().run()