import OvercookedBoardRenderer from "./board-renderer";

export default class OvercookedTrajectoryReplay {
    constructor({
        container_id,
        trajectory,
        start_grid,
        TIMESTEP = 180,
        cook_time = 20,
        init_orders = null,
        focus_step = null,
        focus_pause_ms = 1000,
        completion_callback = () => { },
        focus_callback = () => { },
        DELIVERY_REWARD = 20
    }) {
        this.container_id = container_id;
        this.trajectory = trajectory;
        this.start_grid = start_grid;
        this.TIMESTEP = TIMESTEP;
        this.cook_time = cook_time;
        this.init_orders = init_orders;
        this.focus_step = focus_step;
        this.focus_pause_ms = focus_pause_ms;
        this.completion_callback = completion_callback;
        this.focus_callback = focus_callback;
        this.focus_triggered = false;
        this.pause_until = 0;
        this.closed = false;

        this.observations = trajectory.ep_states[0];
        this.rewards = trajectory.ep_rewards[0].concat();
        this.total_timesteps = this.observations.length;
        this.cur_gameloop = 0;
        this.renderer = new OvercookedBoardRenderer({
            container_id: this.container_id,
            trial: {
                layout_grid: start_grid,
                mode: "replay",
                human_player_index: null,
                target_agent_index: 0
            }
        });
    }

    init() {
        this.renderer.init();
        this.cumulativeScores = [];
        let running = 0;
        for (let i = 0; i < this.rewards.length; i += 1) {
            running += this.rewards[i];
            this.cumulativeScores.push(running);
        }

        this.gameloop = setInterval(() => {
            this.tick();
        }, this.TIMESTEP);
    }

    tick() {
        if (this.closed) {
            return;
        }
        if (this.pause_until && Date.now() < this.pause_until) {
            return;
        }
        if (this.cur_gameloop >= this.total_timesteps) {
            this.close();
            return;
        }

        const state = this.observations[this.cur_gameloop];
        this.renderer.renderState(state, {
            mode: "replay",
            score: this.cumulativeScores[this.cur_gameloop] || 0,
            time_left: Math.max(this.total_timesteps - this.cur_gameloop, 0),
            step_count: this.cur_gameloop
        });

        if (!this.focus_triggered && this.focus_step !== null && this.cur_gameloop >= this.focus_step) {
            this.focus_triggered = true;
            this.pause_until = Date.now() + this.focus_pause_ms;
            this.focus_callback({
                focus_step: this.focus_step,
                current_step: this.cur_gameloop
            });
        }

        this.cur_gameloop += 1;
    }

    close() {
        if (this.closed) {
            return;
        }
        this.closed = true;
        if (typeof this.gameloop !== "undefined") {
            clearInterval(this.gameloop);
        }
        this.renderer.close();
        this.completion_callback();
    }
}
