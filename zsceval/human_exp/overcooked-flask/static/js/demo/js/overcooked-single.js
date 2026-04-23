import $ from "jquery";

import OvercookedBoardRenderer from "./board-renderer";

const DEFAULT_ACTION_IDX = 4;

function actionLabel(actionIdx, actionOptions) {
    for (let i = 0; i < actionOptions.length; i += 1) {
        if (Number(actionOptions[i].value) === Number(actionIdx)) {
            return actionOptions[i].label;
        }
    }
    return String(actionIdx);
}

function syncPost(url, payload) {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url, false);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.send(JSON.stringify(payload));
    return xhr.responseText ? JSON.parse(xhr.responseText) : {};
}

export default class OvercookedSessionTask {
    constructor({
        container_id,
        control_id,
        user_info,
        trial,
        start_payload,
        action_options,
        TIMESTEP = 200,
        completion_callback = () => { },
        probe_request_callback = () => { },
        probe_result_callback = () => { }
    }) {
        this.container_id = container_id;
        this.control_id = control_id;
        this.user_info = user_info;
        this.trial = trial;
        this.start_payload = start_payload || {};
        this.action_options = action_options || [];
        this.TIMESTEP = TIMESTEP;
        this.completion_callback = completion_callback;
        this.probe_request_callback = probe_request_callback;
        this.probe_result_callback = probe_result_callback;
        this.human_player_index = trial.human_player_index;
        this.ai_player_indices = trial.ai_player_indices || [];
        this.target_agent_index = trial.target_agent_index || 0;
        this.max_time = trial.max_time || 0;
        this.mode = trial.mode;
        this.probe_config = trial.probe || { enabled: false };
        this.probe_records = [];
        this.probe_used = false;
        this.pending_probe = null;
        this.awaiting_probe_resume = false;
        this.closed = false;
        this.pendingHumanActionIdx = DEFAULT_ACTION_IDX;
        this.score = 0;
        this.time_left = trial.max_time || 0;
        this.cur_gameloop = 0;
        this.state = null;
        this.renderer = new OvercookedBoardRenderer({
            container_id: this.container_id,
            trial: this.trial
        });
    }

    init() {
        this.renderer.init();
        const initialRuntime = this.start_payload.runtime || {};
        this.applyRuntimeUpdate(initialRuntime);

        if (this.human_player_index !== null && this.human_player_index !== undefined) {
            this.activate_response_listener();
        }

        this.gameloop = setInterval(() => {
            this.tick();
        }, this.TIMESTEP);
    }

    tick() {
        if (this.closed || this.awaiting_probe_resume) {
            return;
        }
        const response = syncPost("/step_trial", {
            trial_id: this.trial.id,
            human_action_idx: this.pendingHumanActionIdx,
            user_info: this.user_info
        });
        this.pendingHumanActionIdx = DEFAULT_ACTION_IDX;

        if (!response.status) {
            this.close();
            return;
        }

        if (response.probe_pending) {
            this.pending_probe = response.probe;
            this.awaiting_probe_resume = true;
            this.disable_response_listener();
            this.applyRuntimeUpdate(response.runtime);
            this.probe_request_callback({
                trial_id: this.trial.id,
                mode: this.mode,
                target_agent_index: this.target_agent_index,
                game_loop: this.pending_probe.probe_game_loop,
                time_left: this.pending_probe.time_left
            });
            return;
        }

        this.applyRuntimeUpdate(response.runtime);
        if (response.done || this.time_left <= 0) {
            this.close();
        }
    }

    submitProbe(answerIdx, extraData) {
        if (!this.pending_probe) {
            return null;
        }
        const response = syncPost("/submit_probe", {
            user_info: this.user_info,
            trial_id: this.trial.id,
            selected_action_idx: Number(answerIdx),
            extra: extraData
        });
        const record = response.probe_record;
        if (!record) {
            return null;
        }
        this.probe_records = this.probe_records.concat([record]);
        this.probe_used = true;
        this.probe_result_callback(record);
        return record;
    }

    resumeAfterProbe() {
        if (!this.pending_probe) {
            return;
        }
        const response = syncPost("/resume_trial_after_probe", {
            user_info: this.user_info,
            trial_id: this.trial.id
        });
        if (!response.status) {
            this.close();
            return;
        }
        this.pending_probe = null;
        this.awaiting_probe_resume = false;
        this.applyRuntimeUpdate(response.runtime);
        if (response.done || this.time_left <= 0) {
            this.close();
        }
    }

    applyRuntimeUpdate(runtime) {
        if (!runtime) {
            return;
        }
        this.state = runtime.state || this.state;
        this.score = runtime.score || 0;
        this.cur_gameloop = runtime.step_count || 0;
        this.time_left = runtime.time_left || 0;
        this.probe_records = runtime.probe_records || this.probe_records;
        if (this.state) {
            this.renderer.renderState(this.state, {
                mode: this.mode,
                score: this.score,
                time_left: this.time_left,
                step_count: this.cur_gameloop
            });
        }

        if (this.human_player_index !== null && this.human_player_index !== undefined) {
            this.activate_response_listener();
        }
    }

    activate_response_listener() {
        if (this.human_player_index === null || this.human_player_index === undefined) {
            return;
        }
        this.disable_response_listener();
        $(document).on("keydown.overcooked-human", (e) => {
            let actionIdx = null;
            switch (e.which) {
                case 37:
                    actionIdx = 3;
                    break;
                case 38:
                    actionIdx = 0;
                    break;
                case 39:
                    actionIdx = 2;
                    break;
                case 40:
                    actionIdx = 1;
                    break;
                case 32:
                    actionIdx = 5;
                    break;
                default:
                    return;
            }
            e.preventDefault();
            this.pendingHumanActionIdx = actionIdx;
            this.disable_response_listener();
        });
    }

    disable_response_listener() {
        $(document).off("keydown.overcooked-human");
    }

    buildSummary() {
        return {
            trial_id: this.trial.id,
            mode: this.mode,
            score: this.score,
            total_steps: this.cur_gameloop,
            probes: this.probe_records,
            primary_probe: this.probe_records.length ? this.probe_records[0] : null
        };
    }

    close() {
        if (this.closed) {
            return;
        }
        this.closed = true;
        if (typeof this.gameloop !== "undefined") {
            clearInterval(this.gameloop);
        }
        this.disable_response_listener();
        this.renderer.close();

        this.completion_callback({
            summary: this.buildSummary()
        });
    }
}
