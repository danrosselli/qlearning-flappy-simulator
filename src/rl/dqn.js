/* ============================================================
 * DEEP Q-NETWORK — FLAPPY BIRD ADAPTER
 * ------------------------------------------------------------
 * Uses TensorFlow.js to implement DQN with a neural network.
 * States are continuous [dx, dy, velY], no discretization.
 * ============================================================ */

import * as tf from '@tensorflow/tfjs';

/* ------------------------------------------------------------
 * ACTIONS
 * ------------------------------------------------------------ */
export const ACTION_FLAP = 1;
export const ACTION_IDLE = 0;
export const ACTIONS = [ACTION_IDLE, ACTION_FLAP];

/* ------------------------------------------------------------
 * HYPERPARAMETERS
 * ------------------------------------------------------------ */
export const gamma = 0.99; // Discount Factor
export let epsilon = 0.5; // Exploration Rate
export const EPSILON_MIN = 0.001;
export const EPSILON_DECAY = 0.9995;
export const BATCH_SIZE = 64;
export const REPLAY_BUFFER_SIZE = 30000;
export const TARGET_UPDATE_FREQ = 500; // Atualiza target network com menos frequência
export const LEARNING_RATE = 0.002; // Learning rate menor para convergência suave
export const TRAIN_THROTTLE = 2; // Treina a cada N passos para evitar sobrecarga

/* ------------------------------------------------------------
 * REPLAY BUFFER
 * ------------------------------------------------------------ */
class ReplayBuffer {
  constructor(maxSize) {
    this.maxSize = maxSize;
    this.buffer = [];
  }

  add(state, action, reward, nextState, done) {
    const transition = { state, action, reward, nextState, done };
    if (this.buffer.length >= this.maxSize) {
      this.buffer.shift();
    }
    this.buffer.push(transition);
  }

  sampleRandomBasic(batchSize) {
    const batch = [];
    for (let i = 0; i < batchSize; i++) {
      const idx = Math.floor(Math.random() * this.buffer.length);
      batch.push(this.buffer[idx]);
    }
    return batch;
  }

  sampleLastPrioritizedAndRandom(batchSize) {
    const batch = [];
    const recentPercentage = 0.25;  // 25% do batchSize pra recentes (ajuste aqui: 0.25 pra 25%)
    const numRecent = Math.floor(batchSize * recentPercentage);  // Dinâmico: ex: 128*0.25=32
    const numRandom = batchSize - numRecent;  // Resto aleatório: ex: 128-32=96

    // Parte 1: Últimas numRecent recentes (mais nova primeiro: reverse do tail)
    const recentStart = Math.max(0, this.buffer.length - numRecent);
    for (let i = this.buffer.length - 1; i >= recentStart; i--) {
      batch.push(this.buffer[i]);  // Adiciona da mais nova pra mais antiga
    }

    // Parte 2: numRandom aleatórias do buffer TODO (com possível repetição, mas baixa chance)
    if (numRandom > 0 && this.buffer.length > 0) {
      for (let i = 0; i < numRandom; i++) {
        const idx = Math.floor(Math.random() * this.buffer.length);
        batch.push(this.buffer[idx]);
      }
    }

    // Shuffle Opcional: Mistura o batch todo pra não enviesar o fit() (recomendado)
    for (let i = batch.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [batch[i], batch[j]] = [batch[j], batch[i]];  // Swap simples
    }

    return batch;
  }

  sampleRewardPrioritized(batchSize) {
    if (this.buffer.length === 0) return [];

    const epsilon = 1.0;  // Adiciona pra evitar p=0 em rewards baixos
    const absRewards = this.buffer.map(t => Math.abs(t.reward) + epsilon);  // |r| + ε
    const total = absRewards.reduce((a, b) => a + b, 0);
    const probs = absRewards.map(r => r / total);  // Normaliza pra [0,1]

    // Cumsum pra weighted random (simples, sem binary search pra velocidade)
    const cumsum = []; let sum = 0;
    for (let p of probs) {
      sum += p;
      cumsum.push(sum);
    }

    const batch = [];
    for (let i = 0; i < batchSize; i++) {
      const rand = Math.random();
      // Encontra idx via binary search (otimizado: O(log N) em vez de O(N))
      let low = 0, high = cumsum.length - 1;
      while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        if (cumsum[mid] >= rand) {
          high = mid - 1;
        } else {
          low = mid + 1;
        }
      }
      const idx = low;
      batch.push(this.buffer[idx]);
    }

    // Shuffle opcional pra diversidade
    for (let i = batch.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [batch[i], batch[j]] = [batch[j], batch[i]];
    }

    return batch;
  }

  size() {
    return this.buffer.length;
  }
}

/* ------------------------------------------------------------
 * DQN AGENT
 * ------------------------------------------------------------ */
export class DQNAgent {
  constructor() {
    // Force WebGL backend for better performance and stability
    tf.setBackend('webgl').then(() => console.log('TensorFlow.js backend: WebGL')).catch(err => console.warn('Failed to set WebGL:', err));
    this.model = this.createModel();
    this.targetModel = this.createModel();
    this.targetModel.setWeights(this.model.getWeights());
    this.replayBuffer = new ReplayBuffer(REPLAY_BUFFER_SIZE);
    this.stepCount = 0;
    this.trainingInProgress = false; // Flag para evitar treinos concorrentes
  }

  createModel() {
    const model = tf.sequential();

    model.add(tf.layers.dense({ units: 256, activation: 'relu', inputShape: [7] }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 2, activation: 'linear' }));

    model.compile({
      optimizer: tf.train.adam(LEARNING_RATE),
      loss: 'meanSquaredError'
    });

    return model;
  }

  async chooseAction(state) {
    if (Math.random() < epsilon) {
      return Math.random() < 0.2 ? ACTION_FLAP : ACTION_IDLE;
    }
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state], [1, 7]);
      const qValues = this.model.predict(stateTensor);
      const action = qValues.argMax(1).dataSync()[0];
      return action;
    });
  }

  async train() {
    // CORREÇÃO: Incrementa stepCount SEMPRE, no início, para evitar congelamento
    this.stepCount++;

    // Throttle: Treina só a cada TRAIN_THROTTLE passos (agora stepCount já avançou)
    if (this.stepCount % TRAIN_THROTTLE !== 0) {
      // Opcional: Log para debug (comente se quiser silenciar)
      // console.log('Step skipped:', this.stepCount);
      return;
    }
    if (this.replayBuffer.size() < BATCH_SIZE) return;
    if (this.trainingInProgress) {
      console.log('Training skipped: Already in progress');
      return;
    }

    this.trainingInProgress = true;

    const batch = this.replayBuffer.sampleLastPrioritizedAndRandom(BATCH_SIZE);

    const states = [];
    const actions = [];
    const rewards = [];
    const nextStates = [];
    const dones = [];

    batch.forEach(transition => {
      states.push(transition.state);
      actions.push(transition.action);
      rewards.push(transition.reward);
      nextStates.push(transition.nextState);
      dones.push(transition.done ? 1 : 0);
    });

    // Create input tensors
    const stateTensor = tf.tensor2d(states, [BATCH_SIZE, 7]);
    const nextStateTensor = tf.tensor2d(nextStates, [BATCH_SIZE, 7]);
    const rewardTensor = tf.tensor1d(rewards);
    const doneTensor = tf.tensor1d(dones);

    try {
      // Compute targets inside tidy
      const targets = tf.tidy(() => {
        const nextQValues = this.targetModel.predict(nextStateTensor);
        const maxNextQ = nextQValues.max(1);
        const notDone = tf.logicalNot(tf.cast(doneTensor, 'bool'));
        const targetQ = rewardTensor.add(maxNextQ.mul(gamma).mul(notDone));
        const qValues = this.model.predict(stateTensor);
        const qValuesArray = qValues.arraySync();
        const targetQArray = targetQ.arraySync();
        for (let i = 0; i < BATCH_SIZE; i++) {
          qValuesArray[i][actions[i]] = targetQArray[i];
        }
        const targets = tf.tensor2d(qValuesArray, [BATCH_SIZE, 2]);
        nextQValues.dispose();
        maxNextQ.dispose();
        targetQ.dispose();
        qValues.dispose();
        return targets;
      });

      // Fit outside tidy
      await this.model.fit(stateTensor, targets, {
        epochs: 1,
        batchSize: BATCH_SIZE,
        verbose: 0
      });

      this.decayEpsilon();

      targets.dispose();
      stateTensor.dispose();
      nextStateTensor.dispose();
      rewardTensor.dispose();
      doneTensor.dispose();

      //console.log('Training completed. Step:', this.stepCount, 'Buffer size:', this.replayBuffer.size());

    } catch (error) {
      console.error('Training error:', error);
    } finally {
      this.trainingInProgress = false; // Sempre libera a flag
    }

    // CORREÇÃO: Check de target update após incremento (agora sempre executa)
    if (this.stepCount % TARGET_UPDATE_FREQ === 0) {
      this.targetModel.setWeights(this.model.getWeights());
    }
  }

  async saveBrain(generation) {
    await this.model.save('localstorage://flappy-dqn');
    localStorage.setItem('flappy_dqn_metadata', JSON.stringify({
      epsilon,
      generation
    }));
    console.log('DQN Brain saved! Gen:', generation);
  }

  async loadBrain() {
    try {
      const model = await tf.loadLayersModel('localstorage://flappy-dqn');
      this.model = model;
      // CORREÇÃO: Recompile o modelo carregado para restaurar optimizer/loss
      this.model.compile({
        optimizer: tf.train.adam(LEARNING_RATE),
        loss: 'meanSquaredError'
      });
      this.targetModel = this.createModel();
      this.targetModel.setWeights(model.getWeights());
      const metadata = JSON.parse(localStorage.getItem('flappy_dqn_metadata'));
      if (metadata) {
        epsilon = metadata.epsilon;
        const generation = metadata.generation;
        console.log('DQN Brain loaded! Epsilon:', epsilon, 'Gen:', generation);
        return { success: true, generation };
      }
    } catch (e) {
      console.log('No DQN brain found');
    }
    return { success: false, generation: 1 };
  }

  decayEpsilon() {
    if (epsilon > EPSILON_MIN) {
      epsilon *= EPSILON_DECAY;
      if (epsilon < EPSILON_MIN) epsilon = EPSILON_MIN;
    }
  }

  getQValues(state) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state], [1, 7]);
      const qValues = this.model.predict(stateTensor);
      const qArray = qValues.dataSync();
      return { [ACTION_IDLE]: qArray[0], [ACTION_FLAP]: qArray[1] };
    });
  }
}

export function resetBrain() {
  localStorage.removeItem('flappy-dqn');
  localStorage.removeItem('flappy_dqn_metadata');
  epsilon = 0.1;
  console.log('DQN Brain reset');
}