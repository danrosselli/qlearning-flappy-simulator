import Phaser from 'phaser';
import {
	ACTIONS, ACTION_FLAP, ACTION_IDLE,
	DQNAgent, resetBrain, epsilon
} from '../../rl/dqn';

export class Game extends Phaser.Scene {
	constructor() {
		super('Game');
		this.gameOver = false;
		this.generation = 1;
		this.highScore = 0;
		this.lastState = null;
		this.lastAction = null;
		this.agent = new DQNAgent();
	}

	preload() {
		this.load.setPath('assets/sprites');
		this.load.image('bg', 'background-day.png');
		this.load.image('bird_up', 'bluebird-upflap.png');
		this.load.image('bird_mid', 'bluebird-midflap.png');
		this.load.image('bird_down', 'bluebird-downflap.png');
		this.load.image('pipeGreen', 'pipe-green.png');
		this.load.image('pipeRed', 'pipe-red.png');
	}

	async create() {
		if (this.generation === 1) {
			const loaded = await this.agent.loadBrain();
			if (loaded.success) {
				this.generation = loaded.generation;
			}
		}

		this.gameOver = false;
		this.score = 0;
		this.lastState = null;
		this.lastAction = null;
		this.zones = [];
		this.bonusReward = 0;
		this.proximityReward = 0;

		const bg = this.add.image(this.scale.width / 2, this.scale.height / 2, 'bg');
		bg.setDisplaySize(this.scale.width, this.scale.height);

		this.bird = this.physics.add.sprite(120, this.scale.height / 2, 'bird_mid');
		this.bird.setGravityY(1000);
		this.bird.setCollideWorldBounds(true);

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
		this.bird.setDisplaySize(68, 48);
		this.bird.setBodySize(this.bird.width - 10, this.bird.height - 10);

		this.pipes = this.physics.add.group();

		this.scoreText = this.add.text(16, 16, 'Score: 0', {
			fontSize: '32px',
			fill: '#fff',
			stroke: '#000',
			strokeThickness: 4,
		});

		this.hudText = this.add.text(16, 50, '', {
			fontSize: '18px',
			fill: '#ff0',
			stroke: '#000',
			strokeThickness: 3,
		});

		this.scoreText.setDepth(1000);
		this.hudText.setDepth(1000);

		this.pipeTimer = this.time.addEvent({
			delay: 2500,
			callback: this.addPipeRow,
			callbackScope: this,
			loop: true,
		});

		this.physics.add.collider(this.bird, this.pipes, this.hitPipe, null, this);
	}

	async update() {
		if (this.gameOver) return;

		// 1. Observar Estado
		const closestPipe = this.getClosestPipe();
		let dx, dy, velY, gapHeight;
		if (closestPipe) {
			dx = closestPipe.x - this.bird.x;
			dy = this.bird.y - closestPipe.body.center.y;
			velY = this.bird.body.velocity.y;
			gapHeight = closestPipe.height || 300;  // Default se null (use gap médio)
		} else {
			dx = 1000; dy = 0; velY = 0; gapHeight = 300;  // Default neutro
		}
		const currentState = [dx / 1000, dy / 400, velY / 1000, gapHeight / 400];  // Novo: + gap norm ~[0.5,1.0]

		// 2. Calcular Recompensa de Proximidade
		if (closestPipe) {
			const gap = closestPipe.height;
			const halfGap = gap / 2;
			const absDy = Math.abs(dy);
			const scale = halfGap * 1.5;
			const absDyNorm = absDy / scale;
			const velPenalty = Math.min(0, -velY / 350) * 0.2;  // Penaliza queda rápida (-0.2 max)
			this.proximityReward = (1.0 * Math.exp(-absDyNorm * 2) - 0.5) + velPenalty;
			this.proximityReward = Math.max(this.proximityReward, -0.5);
		} else {
			this.proximityReward = 0;
		}

		// 3. Armazenar Transição
		if (this.lastState !== null && this.lastAction !== null) {
			const reward = 1 + this.bonusReward + this.proximityReward;
			this.agent.replayBuffer.add(
				this.lastState,
				this.lastAction,
				reward,
				currentState,
				this.gameOver
			);
			await this.agent.train();
		}
		this.bonusReward = 0;

		// 4. Escolher Ação
		const action = await this.agent.chooseAction(currentState);

		// 5. Executar Ação
		let actionStr = "IDLE";
		if (action === ACTION_FLAP) {
			actionStr = "FLAP";
			if (this.bird.body.velocity.y > -100) {
				this.flap();
			}
		}

		// 6. Armazenar para Próximo Frame
		this.lastState = currentState;
		this.lastAction = action;

		// 7. Física e Limpeza
		if (this.bird.angle < 20) {
			this.bird.angle += 1;
		}

		this.pipes.getChildren().forEach((pipe) => {
			const w = (pipe.displayWidth !== undefined) ? pipe.displayWidth : (pipe.width || 0);
			if (pipe.x + w < 0) pipe.destroy();
		});

		if (this.zones) {
			for (let i = this.zones.length - 1; i >= 0; i--) {
				const zone = this.zones[i];
				if (zone.x + zone.width < 0) {
					zone.destroy();
					this.zones.splice(i, 1);
				}
			}
		}

		if (this.bird.y > this.scale.height + 50 || this.bird.y < -50) {
			this.hitPipe();
		}

		// 8. Atualizar HUD
		const qValues = this.agent.getQValues(currentState);
		this.hudText.setText(
			`Gen: ${this.generation}\n` +
			`High: ${this.highScore}\n` +
			`Epsilon: ${epsilon.toFixed(4)}\n` +
			`DX: ${Math.floor(dx)}\n` +
			`DY: ${Math.floor(dy)}\n` +
			`VelY: ${Math.floor(velY)}\n` +
			`Gap: ${Math.floor(gapHeight)}\n` +
			`Prox: ${this.proximityReward.toFixed(2)}\n` +
			`Q-Idle: ${qValues[ACTION_IDLE].toFixed(2)}\n` +
			`Q-Flap: ${qValues[ACTION_FLAP].toFixed(2)}\n` +
			`Action: ${actionStr}`
		);

	}

	flap() {
		if (this.gameOver) return;
		this.bird.setVelocityY(-350);
		this.bird.angle = -20;
	}

	getClosestPipe() {
		let closest = null;
		let minDist = Infinity;

		if (!this.zones) return null;

		this.zones.forEach(zone => {
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
		const gap = Phaser.Math.Between(200, 410);
		const centerY = Phaser.Math.Between(150, this.scale.height - 150);
		const x = this.scale.width + 50;

		const pipeKey = Phaser.Math.Between(0, 1) === 0 ? 'pipeGreen' : 'pipeRed';
		const top = this.pipes.create(x, centerY - gap / 2, pipeKey).setOrigin(0, 1);
		top.body.allowGravity = false;
		top.setImmovable(true);
		top.setVelocityX(-200);
		top.setFlipY(true);
		top.setDisplaySize(104, 640);

		const topMouthBottomY = centerY - gap / 2;
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

		const bottom = this.pipes.create(x, centerY + gap / 2, pipeKey).setOrigin(0, 0);
		bottom.body.allowGravity = false;
		bottom.setImmovable(true);
		bottom.setVelocityX(-200);
		bottom.setDisplaySize(104, 640);

		const bottomMouthTopY = centerY + gap / 2;
		const bottomMouthHeight = bottom.displayHeight;
		const startY = bottomMouthTopY + bottomMouthHeight;
		const bottomBodyHeight = Math.max(0, this.scale.height - startY);
		if (bottomBodyHeight > 0) {
			const bottomBodyCenterY = startY + bottomBodyHeight / 2;
			const bottomBody = this.add.rectangle(x + bottom.displayWidth / 2, bottomBodyCenterY, bottom.displayWidth, bottomBodyHeight);
			bottomBody.setOrigin(0.5, 0.5);
			this.physics.add.existing(bottomBody);
			bottomBody.body.setAllowGravity(false);
			topBody.body.setImmovable(true);
			bottomBody.body.setVelocityX(-200);
			bottomBody.setVisible(false);
			this.pipes.add(bottomBody);
		}

		const zone = this.add.zone(x + top.displayWidth / 2, centerY, 2, gap);
		this.physics.world.enable(zone);
		zone.body.setVelocityX(-200);
		zone.body.allowGravity = false;
		zone.scored = false;

		if (!this.zones) this.zones = [];
		this.zones.push(zone);

		this.physics.add.overlap(this.bird, zone, (bird, z) => {
			if (!z.scored) {
				z.scored = true;
				this.score++;
				this.scoreText.setText('Score: ' + this.score);
				this.bonusReward = 15;
			}
		});
	}

	async hitPipe() {
		if (this.gameOver) return;
		this.gameOver = true;

		const deathReward = -50;
		if (this.lastState !== null && this.lastAction !== null) {
			const reward = deathReward;
			const terminalState = [0, 0, 0, 0];
			this.agent.replayBuffer.add(
				this.lastState,
				this.lastAction,
				reward,
				terminalState,
				true
			);
			await this.agent.train();
		}

		this.highScore = Math.max(this.highScore, this.score);
		this.generation++;
		await this.agent.saveBrain(this.generation);

		this.endGame();
	}

	endGame() {
		if (this.pipeTimer) {
			try { this.pipeTimer.remove(false); } catch (e) { }
			this.pipeTimer = null;
		}
		if (this.pipes) {
			try { this.pipes.setVelocityX(0); } catch (e) { }
		}
		if (this.bird && this.bird.body) {
			try {
				this.bird.setVelocity(0);
				this.bird.body.setAllowGravity(false);
			} catch (e) { }
		}
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

		this.time.delayedCall(500, () => {
			this.anims.resumeAll();
			this.physics.resume();
			this.scene.restart();
		});
	}
}