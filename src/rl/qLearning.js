/* ============================================================
 * Q-LEARNING — FLAPPY BIRD ADAPTER
 * ------------------------------------------------------------
 * Adapted for continuous state space using discretization.
 * ============================================================ */

/* ------------------------------------------------------------
 * ACTIONS
 * ------------------------------------------------------------ */
export const ACTION_FLAP = 1;
export const ACTION_IDLE = 0;
export const ACTIONS = [ACTION_IDLE, ACTION_FLAP];

/* ------------------------------------------------------------
 * Q-TABLE
 * ------------------------------------------------------------
 * lazy initialization for sparse states
 * ------------------------------------------------------------ */
export let Q = {};

/* ------------------------------------------------------------
 * HYPERPARAMETERS
 * ------------------------------------------------------------ */
export const alpha = 0.2;   // Learning Rate
export const gamma = 1.0;   // Discount Factor (Propagate rewards fully)
export let epsilon = 0.3; // Exploration Rate (Start high for training)
export const EPSILON_MIN = 0.01;
export const EPSILON_DECAY = 0.98;

// Initial epsilon helper
export function setEpsilon(val) {
  epsilon = val;
}

/* ------------------------------------------------------------
 * STATE DISCRETIZATION
 * ------------------------------------------------------------
 * We map continuous (dx, dy) to a discrete string key.
 * 
 * dx: Horizontal distance to next pipe center
 * dy: Vertical distance to next pipe center (birdY - pipeY)
 * ------------------------------------------------------------ */

// Grid resolution
export const DIST_X = 20; // 50px buckets for X
export const DIST_Y = 20; // 40px buckets for Y

export function getStateKey(dx, dy) {
  // Discretize
  // We limit the range to avoid infinite states, though in game bounds are fixed.
  const x = Math.floor(dx / DIST_X);
  const y = Math.floor(dy / DIST_Y);

  return `${x},${y}`;
}

/* ------------------------------------------------------------
 * Q-TABLE MANAGEMENT
 * ------------------------------------------------------------ */
export function getQ(state) {
  if (!Q[state]) {
    Q[state] = {};
    for (const action of ACTIONS) {
      Q[state][action] = 0;
    }
  }
  return Q[state];
}

/* ------------------------------------------------------------
 * ACTION SELECTION (E-GREEDY)
 * ------------------------------------------------------------ */
export function chooseAction(state) {
  // Explore
  if (Math.random() < epsilon) {
    // Bias towards IDLE because FLAP is strong
    // 5% chance to flap, 95% idle
    return Math.random() < 0.05 ? ACTION_FLAP : ACTION_IDLE;
  }

  // Exploit
  const qValues = getQ(state);

  // Check if values are equal (initially 0) -> Random choice to break tie
  if (qValues[ACTION_IDLE] === qValues[ACTION_FLAP]) {
    return ACTIONS[Math.floor(Math.random() * ACTIONS.length)];
  }

  return qValues[ACTION_FLAP] > qValues[ACTION_IDLE]
    ? ACTION_FLAP
    : ACTION_IDLE;
}

/* ------------------------------------------------------------
 * UPDATE (BELLMAN)
 * ------------------------------------------------------------ */
export function updateQ(state, action, reward, nextState) {
  const qValues = getQ(state);
  const currentQ = qValues[action];

  const nextQValues = getQ(nextState);
  const maxNextQ = Math.max(nextQValues[ACTION_IDLE], nextQValues[ACTION_FLAP]);

  qValues[action] = currentQ + alpha * (reward + gamma * maxNextQ - currentQ);
}

/* ------------------------------------------------------------
 * PERSISTENCE
 * ------------------------------------------------------------ */
export function saveBrain(gen) {
  const data = {
    Q: Q,
    epsilon: epsilon,
    generation: gen || 1
  };
  localStorage.setItem('flappy_brain', JSON.stringify(data));
  console.log('Brain saved! States:', Object.keys(Q).length, 'Gen:', data.generation);
}

export function loadBrain() {
  const data = localStorage.getItem('flappy_brain');
  if (data) {
    const parsed = JSON.parse(data);
    if (parsed.Q) {
      Q = parsed.Q;
      epsilon = parsed.epsilon || epsilon;
      const gen = parsed.generation || 1;
      console.log('Brain loaded! States:', Object.keys(Q).length, 'Epsilon:', epsilon, 'Gen:', gen);
      return { success: true, generation: gen };
    }
  }
  return { success: false, generation: 1 };
}

export function resetBrain() {
  Q = {};
  epsilon = 0.1;
  saveBrain();
}

export function decayEpsilon() {
  if (epsilon > EPSILON_MIN) {
    epsilon *= EPSILON_DECAY;
    if (epsilon < EPSILON_MIN) epsilon = EPSILON_MIN;
  }
}
