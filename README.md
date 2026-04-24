## Rubik's Cube

![Rubik's Cube](https://github.com/santakd/rubikscube/blob/main/rubikscube.gif)



At first glance, the Rubik’s Cube is deceptively simple—a colorful, plastic cube that fits neatly in the palm of your hand. Yet, beneath its six brightly colored faces lies a labyrinth of mathematical complexity that has captivated millions. Invented in 1974 by Ernő Rubik, a Hungarian architecture professor, the "Magic Cube" wasn’t originally designed as a toy. It was created as a tactile teaching tool to help his students understand three-dimensional geometry and the movement of independent parts. In fact, when Rubik scrambled his own invention for the first time, it took him a full month to figure out how to put it back together.

The genius of the puzzle lies in its brilliant physical architecture. While it appears to be a solid block of 54 colored squares, it is actually composed of 26 smaller cubes—known as "cubies"—interlocking tightly around a hidden, six-pronged central pivot. This ingenious core allows any given face to rotate smoothly without the entire structure collapsing. The pieces are strictly categorized into centers (which dictate the final color of the face and never move relative to one another), edges (with two colors), and corners (with three colors). When you turn the cube, you aren't just moving flat stickers; you are navigating solid, three-dimensional geometric pieces through space.

The true challenge of the Rubik’s Cube becomes apparent the moment you make your first few turns. Order instantly dissolves into chaotic entropy. Mathematically, a standard 3x3x3 Rubik’s Cube can be arranged into exactly 43,252,003,274,489,856,000 (over 43 quintillion) different configurations. To put that staggering number into perspective: if you had one standard-sized cube for every possible state and laid them end-to-end, the line would stretch across the entire observable universe. If you made one turn per second, trying to guess the solution, the sun would burn out long before you finished. You simply cannot stumble upon the solution by accident.

The cube was built to symbolize symmetry, but it threw Rubik a curve: It was also a puzzle. Even a few twists made it difficult to return the small cubes to their starting positions. It was “surprising and deeply emotional,” Rubik tells Smithsonian, with “an inherent element of problem-solving that brought with it complexity, difficulty and experiential value". 

This mathematical vastness creates the central paradox that makes the puzzle so utterly fascinating. It is a battle against chaos that cannot be won with brute force; it requires logic, spatial awareness, and algorithmic thinking. The cruelest trick of the Rubik's Cube is that to fix one piece, you must temporarily destroy the progress you have already made. Solving it demands foresight. You must execute precise sequences of moves—algorithms—that briefly dismantle your solved sections just to maneuver a single new piece into place, before perfectly reversing the sequence to restore order.

Why does a middle-aged plastic puzzle with one right combination and 43 quintillion wrong ones still seduce in our digital age? Because it “talks to human universals” while remaining “languageless,” says Rubik. Mostly though, its appeal is “part of the mystery of the Cube itself.”

Despite its 43 quintillion permutations, mathematicians and software engineers have proven a concept known as "God’s Number"—the awe-inspiring revelation that every single scrambled state, no matter how tangled, is no more than 20 moves away from being perfectly solved. Fifty years after its creation, the Rubik’s Cube remains the ultimate monument to human curiosity. It is a physical manifestation of chaos and order, continually challenging us to look at a mess of fractured colors and believe that, through patience and intellect, we can put things right again.

This repo contains the solver for a standard [3x3](https://github.com/santakd/rubikscube/blob/main/rubiks3x3.py) cube and also a [2x2](https://github.com/santakd/rubikscube/blob/main/rubiks2x2.py) pocket cube.

### ⭐ Like it? Star it!

If you find this project interesting, please give it a star — it helps others discover it too.

---

### The Kociemba Algorithm

Herbert Kociemba’s algorithm is absolutely fascinating. It is the gold standard for software-based Rubik’s Cube solvers and is the mathematical backbone that proved **"God's Number"** (the maximum number of moves required to solve *any* Rubik's Cube state is 20).

Here is a breakdown of how the algorithm works conceptually, followed by exactly how it is integrated into the Python application.

---

### 1. How the Kociemba Algorithm Works (The Theory)

A Rubik’s Cube has about **43 quintillion** possible states. You cannot brute-force a solution. To solve this, Kociemba invented the **Two-Phase Algorithm** in 1992. 

Instead of trying to solve the cube in one massive leap, the algorithm breaks the problem down into two smaller, highly optimized steps:

**Phase 1: "Restrict the move set"**
Normally, you can turn any face 90 degrees: `U, D, L, R, F, B`. 
In Phase 1, the algorithm looks for a sequence of moves to get the cube into a specific mathematical subgroup (a restricted state). In this state:
* All corners are oriented correctly (no twisted corners).
* All edges are oriented correctly (no flipped edges).
* The four "middle slice" edges belong to the middle slice.
Once the cube is in this state, it can be fully solved using *only* 180-degree turns for the vertical faces (`R2, L2, F2, B2`) and standard turns for the top/bottom (`U, D`).

**Phase 2: "Solve the rest"**
Using *only* that restricted move set (`U, D, R2, L2, F2, B2`), the algorithm searches for the sequence that actually solves the cube. Because the state space is now infinitely smaller, this search is incredibly fast.

**The Magic (Iterative Deepening):**
If Phase 1 takes 12 moves and Phase 2 takes 15 moves, the total is 27 moves. Kociemba's algorithm doesn't stop there. It goes back and says, *"What if I take 13 moves in Phase 1? Does that allow me to do Phase 2 in only 5 moves?"* It rapidly searches these trade-offs using massive pre-computed lookup tables (pruning tables) until it finds a solution that is ≤ 20 moves.

---

### 2. How We Integrated It (The Practice)

In the app, we used the `kociemba` Python package, which is a highly optimized implementation of this algorithm. However, Kociemba's engine is extremely strict. It doesn't know what "Red" or "Blue" is—it only knows mathematical coordinates. 

Here is how the app safely bridges the visual UI with the raw math engine:

#### A. Translation: Colors → Faces (`CubeState.to_kociemba_string()`)
If you look at the cube in your hands, the engine demands a 54-character string representing the stickers in exactly this order: **U**p, **R**ight, **F**ront, **D**own, **L**eft, **B**ack.

Because the user can change sticker colors in our "Edit" mode, we don't assume White is Up. Instead, the code looks at the **center sticker** of every face. 
```python
color_to_face = {
    self.state["U"][4]: "U",  # Whatever color is in the middle of the Up face is "U"
    self.state["R"][4]: "R",
    # ...
}
```
It then loops through all 54 stickers. If it sees a Red sticker, and Red is the center of the Front face, it outputs an `F`. This results in a string that looks like `UUUUUUUUURRRRRRRRRFFFFFFFFF...` (if solved).

#### B. Validation (Crucial Step)
If you send a physically impossible cube string to the `kociemba` library (e.g., a cube with 10 red stickers, or a corner twisted out of reality), the C-engine will either throw an ugly exception or get stuck in an infinite loop. 

That’s why, before touching the bridge, the app calls `CubeState.is_valid()`. It strictly ensures exactly 54 stickers exist, exactly 9 of each color exist, and the centers haven't moved. If validation fails, we intercept it and trigger the red UI toast.

#### C. Asynchronous Execution (`App._action_solve()`)
While the Two-Phase algorithm is fast, a difficult scramble might take 0.5 to 1.5 seconds to compute. 
If we simply called `kociemba.solve()` inside our main PyGame loop, the entire window would freeze—you wouldn't even be able to move your mouse.

To fix this, we fire off a background thread:
```python
def thread_target() -> None:
    try:
        moves = self.bridge.solve(cube_copy)
        self.solve_result = moves
    except Exception as e:
        self.solve_error = str(e)

self.solve_thread = threading.Thread(target=thread_target)
self.solve_thread.start()
```
While that thread is thinking, the PyGame main loop continues running at 60 FPS, allowing us to render the rotating white "Loading Spinner" in the top-middle of the screen.

#### D. Parsing and Playback (`SolutionPlayer`)
When the thread finishes, Kociemba returns a single string like:
`"R2 U' F B2 R D' ..."`

The `KociembaBridge` splits this by spaces into a Python list of strings. We pass this list to the `SolutionPlayer`. When you click **[Next]**, the app grabs the current move from the list (e.g., `U'`) and passes it to the `MoveExecutor`, which mathematically rotates our 3D NumPy arrays to match reality, while the UI flashes the blue translucent highlight over the affected stickers!

---
