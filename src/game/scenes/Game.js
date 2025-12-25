import Phaser from 'phaser';
import {
	ACTIONS, ACTION_FLAP, ACTION_IDLE,
	chooseAction, updateQ, getStateKey,
	saveBrain, loadBrain, resetBrain, decayEpsilon,
	epsilon
} from '../../rl/qLearning';

export class Game extends Phaser.Scene {
	constructor() {
		super('Game');
		this.gameOver = false;

		// RL State
		this.generation = 1;
		this.highScore = 0;

		// Last step memory
		this.lastState = null;
		this.lastAction = null;
	}

	preload() {
		// use the sprites folder assets (Flappy sprites)
		this.load.setPath('assets/sprites');
		this.load.image('bg', 'background-day.png');

		// bird frames (use blue bird by default)
		this.load.image('bird_up', 'bluebird-upflap.png');
		this.load.image('bird_mid', 'bluebird-midflap.png');
		this.load.image('bird_down', 'bluebird-downflap.png');

		// pipes (green and red variants)
		this.load.image('pipeGreen', 'pipe-green.png');
		this.load.image('pipeRed', 'pipe-red.png');
	}

	create() {
		// Attempt to load existing brain
		if (this.generation === 1) {
			const loaded = loadBrain();
			if (loaded.success) {
				this.generation = loaded.generation;
			}
		}

		// ensure state resets on restart
		this.gameOver = false;
		this.score = 0;
		this.lastState = null;
		this.lastAction = null;
		this.zones = [];

		// rewards
		this.bonusReward = 0; // reward por passar o pipe
		this.proximityReward = 0;  // reward por proximidade

		// background
		const bg = this.add.image(this.scale.width / 2, this.scale.height / 2, 'bg');
		bg.setDisplaySize(this.scale.width, this.scale.height);

		// bird (use bluebird frames)
		this.bird = this.physics.add.sprite(120, this.scale.height / 2, 'bird_mid');
		this.bird.setGravityY(1000);
		this.bird.setCollideWorldBounds(true);

		// create flap animation from individual bluebird frames (only once)
		if (!this.anims.exists('fly')) {
			this.anims.create({
				key: 'fly',
				frames: [
					{ key: 'bird_up' },
					{ key: 'bird_mid' },
					{ key: 'bird_down' }
				],
				frameRate: 10,
				repeat: -1
			});
		}
		this.bird.play('fly');
		// set bird display size to native 34x24 (user provided) then double it
		this.bird.setDisplaySize(68, 48);
		this.bird.setBodySize(this.bird.width - 10, this.bird.height - 10);

		// input (Manual override REMOVED per user request)
		// this.input.on('pointerdown', this.manualFlap, this);
		// this.input.keyboard.on('keydown-SPACE', this.manualFlap, this);
		// this.input.keyboard.on('keydown-UP', this.manualFlap, this);

		// pipes group
		this.pipes = this.physics.add.group();

		// score
		this.scoreText = this.add.text(16, 16, 'Score: 0', {
			fontSize: '32px',
			fill: '#fff',
			stroke: '#000',
			strokeThickness: 4,
		});

		// HUD
		this.hudText = this.add.text(16, 50, '', {
			fontSize: '18px',
			fill: '#ff0',
			stroke: '#000',
			strokeThickness: 3,
		});

		// ensure score is drawn above pipes
		this.scoreText.setDepth(1000);
		this.hudText.setDepth(1000);

		// spawn pipes
		this.pipeTimer = this.time.addEvent({
			delay: 2500,
			callback: this.addPipeRow,
			callbackScope: this,
			loop: true,
		});

		// collisions
		this.physics.add.collider(this.bird, this.pipes, this.hitPipe, null, this);
	}

	update() {
		if (this.gameOver) return;

		// --------------------------------------------------------
		// RL AGENT LOGIC
		// --------------------------------------------------------

		// 1. Observe State
		const closestPipe = this.getClosestPipe();
		let dx, dy;

		// Calcular distância horizontal e vertical dos pipes mais próximos
		if (closestPipe) {
			dx = closestPipe.x - this.bird.x;
			dy = this.bird.y - closestPipe.body.center.y; // Zone center Y is correct gap center
		} else {
			// No pipes yet, mock distant pipe
			dx = 1000;
			dy = 0;
		}

		// Calcular reward de proximidade (gradiente full exponencial: + centro → - fora)
		if (closestPipe) {
			const gap = closestPipe.height;  // Altura real do gap
			const halfGap = gap / 2;
			const absDy = Math.abs(dy);

			// Exponencial: Pico no centro, decay suave pra negativo
			const scale = halfGap * 1.5;  // Escala ajustada ao gap (maior = decay mais lento)
			const absDyNorm = absDy / scale;
			this.proximityReward = 1.0 * Math.exp(-absDyNorm * 2) - 0.5;  // +0.5 max → ~-0.5 min

			// Opcional: Cap no negativo pra não punir demais
			this.proximityReward = Math.max(this.proximityReward, -0.5);
		} else {
			this.proximityReward = 0;
		}

		const currentState = getStateKey(dx, dy);

		// 2. Update Q-Table (Learn from previous step)
		// Reward for staying alive this frame: +1
		// Plus any bonus (pipe passed)
		if (this.lastState !== null && this.lastAction !== null) {
			const reward = 1 + this.bonusReward + this.proximityReward;  // + novo termo!
			updateQ(this.lastState, this.lastAction, reward, currentState);
		}
		this.bonusReward = 0;

		// 3. Choose Action
		const action = chooseAction(currentState);

		// 4. Perform Action
		let actionStr = "IDLE";
		if (action === ACTION_FLAP) {
			actionStr = "FLAP";
			if (this.bird.body.velocity.y > -20) {
				this.flap();
			}
		}

		// 5. Store for next frame
		this.lastState = currentState;
		this.lastAction = action;

		// --------------------------------------------------------
		// GAME PHYSICS & CLEANUP
		// --------------------------------------------------------

		if (this.bird.angle < 20) {
			this.bird.angle += 1;
		}

		this.pipes.getChildren().forEach((pipe) => {
			const w = (pipe.displayWidth !== undefined) ? pipe.displayWidth : (pipe.width || 0);
			if (pipe.x + w < 0) pipe.destroy();
		});

		// Clean up zones
		if (this.zones) {
			for (let i = this.zones.length - 1; i >= 0; i--) {
				const zone = this.zones[i];
				if (zone.x + zone.width < 0) {
					zone.destroy();
					this.zones.splice(i, 1);
				}
			}
		}

		if (this.bird.y > this.scale.height || this.bird.y < 0) {
			this.hitPipe(); // Treat boundary hit as pipe hit
		}

		// Update HUD
		const dxDisp = Math.floor(dx);
		const dyDisp = Math.floor(dy);
		this.hudText.setText(
			`Gen: ${this.generation}\n` +
			`High: ${this.highScore}\n` +
			`Epsilon: ${epsilon.toFixed(4)}\n` +
			`DX: ${dxDisp} | DY: ${dyDisp}\n` +
			`Prox: ${this.proximityReward.toFixed(2)}\n` +
			`State: ${currentState}\n` +
			`Action: ${actionStr}`
		);
	}

	manualFlap() {
		// Allows human intervention (optional)
		this.flap();
	}

	flap() {
		if (this.gameOver) return;
		this.bird.setVelocityY(-350);
		this.bird.angle = -20;
	}

	getClosestPipe() {
		let closest = null;
		let minDist = Infinity;

		// Ensure zones array exists
		if (!this.zones) return null;

		this.zones.forEach(zone => {
			// Check if zone is valid (not destroyed) and active
			if (zone.active && (zone.x + zone.width / 2) > (this.bird.x)) {
				const dist = zone.x - this.bird.x;
				if (dist < minDist) {
					minDist = dist;
					closest = zone;
				}
			}
		});

		return closest;
	}

	addPipeRow() {
		// make the gap a bit larger to reduce difficulty
		const gap = Phaser.Math.Between(200, 410);
		const centerY = Phaser.Math.Between(150, this.scale.height - 150);
		const x = this.scale.width + 50;

		// pick a random pipe color (green or red) and use as mouth
		const pipeKey = Phaser.Math.Between(0, 1) === 0 ? 'pipeGreen' : 'pipeRed';
		// top mouth (flip vertically so mouth faces downwards)
		const top = this.pipes.create(x, centerY - gap / 2, pipeKey).setOrigin(0, 1);
		top.body.allowGravity = false;
		top.setImmovable(true);
		top.setVelocityX(-200);
		top.setFlipY(true);
		// ensure mouth shows full pipe texture height (original 52x320) scaled 2x -> 104x640
		top.setDisplaySize(104, 640);

		// create a single invisible physics body that fills from y=0 to the mouth top
		const topMouthBottomY = centerY - gap / 2; // mouth bottom (since origin 0,1)
		const topMouthHeight = top.displayHeight;
		const topMouthTopY = topMouthBottomY - topMouthHeight;
		const topBodyHeight = Math.max(0, Math.floor(topMouthTopY));
		if (topBodyHeight > 0) {
			const topBody = this.add.rectangle(x + top.displayWidth / 2, topMouthTopY / 2, top.displayWidth, topBodyHeight);
			topBody.setOrigin(0.5, 0.5);
			this.physics.add.existing(topBody);
			topBody.body.setAllowGravity(false);
			topBody.body.setImmovable(true);
			topBody.body.setVelocityX(-200);
			topBody.setVisible(false);
			this.pipes.add(topBody);
		}

		// bottom mouth (mouth faces upward, use same pipeKey or randomize)
		const bottom = this.pipes.create(x, centerY + gap / 2, pipeKey).setOrigin(0, 0);
		bottom.body.allowGravity = false;
		bottom.setImmovable(true);
		bottom.setVelocityX(-200);
		bottom.setDisplaySize(104, 640);

		// create a single invisible physics body that fills from mouth bottom to bottom of screen
		const bottomMouthTopY = centerY + gap / 2; // mouth top (origin 0,0)
		const bottomMouthHeight = bottom.displayHeight;
		const startY = bottomMouthTopY + bottomMouthHeight;
		const bottomBodyHeight = Math.max(0, this.scale.height - startY);
		if (bottomBodyHeight > 0) {
			const bottomBodyCenterY = startY + bottomBodyHeight / 2;
			const bottomBody = this.add.rectangle(x + bottom.displayWidth / 2, bottomBodyCenterY, bottom.displayWidth, bottomBodyHeight);
			bottomBody.setOrigin(0.5, 0.5);
			this.physics.add.existing(bottomBody);
			bottomBody.body.setAllowGravity(false);
			bottomBody.body.setImmovable(true);
			bottomBody.body.setVelocityX(-200);
			bottomBody.setVisible(false);
			this.pipes.add(bottomBody);
		}

		// scoring zone
		const zone = this.add.zone(x + top.displayWidth / 2, centerY, 2, gap);
		this.physics.world.enable(zone);
		zone.body.setVelocityX(-200);
		zone.body.allowGravity = false;
		zone.scored = false;

		// Track zones for RL
		if (!this.zones) this.zones = [];
		this.zones.push(zone);

		this.physics.add.overlap(this.bird, zone, (bird, z) => {
			if (!z.scored) {
				z.scored = true;
				this.score++;
				this.scoreText.setText('Score: ' + this.score);
				this.bonusReward = 100; // Reward for passing pipe!
			}
		});
	}

	hitPipe() {
		if (this.gameOver) return;
		this.gameOver = true;

		// --------------------------------------------------------
		// RL EPISODE END
		// --------------------------------------------------------
		const deathReward = -100;

		// Terminal update
		// The mock 'nextState' doesn't matter much here since we crashed,
		// but we can pass current state again.
		// Or if we want to be precise, "Dead" state.
		if (this.lastState !== null && this.lastAction !== null) {
			// We use 'lastState' (where we took the fatal action)
			// and update it with a huge negative reward.
			updateQ(this.lastState, this.lastAction, deathReward, this.lastState);
		}

		this.highScore = Math.max(this.highScore, this.score);
		this.generation++;
		decayEpsilon();
		saveBrain(this.generation);

		this.endGame();
	}

	endGame() {
		// stop spawning pipes
		if (this.pipeTimer) {
			try { this.pipeTimer.remove(false); } catch (e) { }
			this.pipeTimer = null;
		}
		// freeze existing pipes visually by zeroing their velocity (don't clear them)
		if (this.pipes) {
			try { this.pipes.setVelocityX(0); } catch (e) { }
		}

		// stop bird motion and gravity
		if (this.bird && this.bird.body) {
			try {
				this.bird.setVelocity(0);
				this.bird.body.setAllowGravity(false);
			} catch (e) { }
		}

		// pause physics and animations so the scene is frozen until restart
		try { this.physics.pause(); } catch (e) { }
		try { this.anims.pauseAll(); } catch (e) { }

		this.add
			.text(this.scale.width / 2, this.scale.height / 2 - 40, 'Game Over', {
				fontSize: '64px',
				fill: '#fff',
				stroke: '#000',
				strokeThickness: 6,
			})
			.setOrigin(0.5)
			.setDepth(1000);

		// Auto restart for training
		this.time.delayedCall(500, () => {
			this.anims.resumeAll();
			this.physics.resume();
			this.scene.restart();
		});
	}
}
