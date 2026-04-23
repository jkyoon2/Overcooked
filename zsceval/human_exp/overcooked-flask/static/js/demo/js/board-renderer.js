import $ from "jquery";

function terrainClass(symbol) {
    switch (symbol) {
        case "X":
            return "counter";
        case "O":
            return "onion";
        case "T":
            return "tomato";
        case "D":
            return "dish";
        case "P":
            return "pot";
        case "S":
            return "serve";
        default:
            return "floor";
    }
}

function terrainLabel(symbol) {
    switch (symbol) {
        case "O":
            return "Onion";
        case "T":
            return "Tomato";
        case "D":
            return "Dish";
        case "P":
            return "Pot";
        case "S":
            return "Serve";
        case "X":
            return "Counter";
        default:
            return "";
    }
}

function orientationLabel(orientation) {
    const key = Array.isArray(orientation) ? orientation.join(",") : String(orientation);
    switch (key) {
        case "0,-1":
            return "N";
        case "0,1":
            return "S";
        case "1,0":
            return "E";
        case "-1,0":
            return "W";
        default:
            return "-";
    }
}

function countIngredients(ingredients) {
    const counts = { onion: 0, tomato: 0 };
    (ingredients || []).forEach((ingredient) => {
        if (ingredient && ingredient.name && Object.prototype.hasOwnProperty.call(counts, ingredient.name)) {
            counts[ingredient.name] += 1;
        }
    });
    return counts;
}

function soupLabel(objectState) {
    const counts = countIngredients(objectState._ingredients || []);
    const size = counts.onion + counts.tomato;
    if (objectState.is_ready) {
        return "Soup Ready";
    }
    if (objectState.is_cooking) {
        const tick = objectState.cooking_tick !== undefined ? objectState.cooking_tick : objectState._cooking_tick;
        return "Soup Cook " + String(tick);
    }
    return "Soup Prep " + String(size);
}

function objectLabel(objectState) {
    switch (objectState.name) {
        case "onion":
            return "Onion";
        case "tomato":
            return "Tomato";
        case "dish":
            return "Dish";
        case "soup":
            return soupLabel(objectState);
        default:
            return objectState.name || "Object";
    }
}

function carriedLabel(heldObject) {
    if (!heldObject) {
        return "";
    }
    switch (heldObject.name) {
        case "onion":
            return "On";
        case "tomato":
            return "Tm";
        case "dish":
            return "Ds";
        case "soup":
            return "Sp";
        default:
            return "Obj";
    }
}

export default class OvercookedBoardRenderer {
    constructor({ container_id, trial }) {
        this.container_id = container_id;
        this.trial = trial;
        this.cells = {};
    }

    init() {
        const container = $("#" + this.container_id);
        container.empty();

        this.root = $('<div class="oc-shell"></div>');
        this.hud = $('<div class="oc-hud"></div>');
        this.hud.html(
            '<div class="oc-hud-item"><span class="oc-hud-label">Mode</span><span class="oc-hud-value" data-field="mode"></span></div>' +
            '<div class="oc-hud-item"><span class="oc-hud-label">Score</span><span class="oc-hud-value" data-field="score">0</span></div>' +
            '<div class="oc-hud-item"><span class="oc-hud-label">Time Left</span><span class="oc-hud-value" data-field="time_left">0.0s</span></div>' +
            '<div class="oc-hud-item"><span class="oc-hud-label">Step</span><span class="oc-hud-value" data-field="step_count">0</span></div>'
        );

        this.grid = $('<div class="oc-grid"></div>');
        this.grid.css("grid-template-columns", "repeat(" + this.trial.layout_grid[0].length + ", minmax(0, 1fr))");

        this.trial.layout_grid.forEach((row, y) => {
            row.split("").forEach((symbol, x) => {
                const cell = $('<div class="oc-cell"></div>');
                cell.addClass("is-" + terrainClass(symbol));
                cell.append('<div class="oc-cell-terrain">' + terrainLabel(symbol) + "</div>");
                const objectLayer = $('<div class="oc-cell-objects"></div>');
                const playerLayer = $('<div class="oc-cell-players"></div>');
                cell.append(objectLayer);
                cell.append(playerLayer);
                this.grid.append(cell);
                this.cells[this.keyFor(x, y)] = {
                    objectLayer: objectLayer,
                    playerLayer: playerLayer,
                };
            });
        });

        this.legend = $(
            '<div class="oc-legend">' +
            '<span class="oc-legend-chip is-ai">AI</span>' +
            '<span class="oc-legend-chip is-human">YOU</span>' +
            '<span class="oc-legend-chip is-target">Probe Target</span>' +
            "</div>"
        );

        this.root.append(this.hud);
        this.root.append(this.grid);
        this.root.append(this.legend);
        container.append(this.root);
        this.renderMeta({
            mode: this.trial.mode,
            score: 0,
            time_left: this.trial.max_time || 0,
            step_count: 0,
        });
    }

    keyFor(x, y) {
        return String(x) + "," + String(y);
    }

    cellFor(position) {
        if (!Array.isArray(position) || position.length < 2) {
            return null;
        }
        return this.cells[this.keyFor(position[0], position[1])] || null;
    }

    playerTag(index) {
        if (index === this.trial.human_player_index) {
            return "YOU";
        }
        return "AI" + String(index);
    }

    renderMeta(meta) {
        this.hud.find('[data-field="mode"]').text(String(meta.mode || this.trial.mode || "").toUpperCase());
        this.hud.find('[data-field="score"]').text(String(meta.score || 0));
        this.hud.find('[data-field="time_left"]').text(String(Number(meta.time_left || 0).toFixed(1)) + "s");
        this.hud.find('[data-field="step_count"]').text(String(meta.step_count || 0));
    }

    clearLayers() {
        Object.keys(this.cells).forEach((key) => {
            this.cells[key].objectLayer.empty();
            this.cells[key].playerLayer.empty();
        });
    }

    renderState(state, meta) {
        if (!this.root) {
            this.init();
        }
        this.clearLayers();
        this.renderMeta(meta || {});

        (state.objects || []).forEach((objectState) => {
            const cell = this.cellFor(objectState.position);
            if (!cell) {
                return;
            }
            const chip = $('<div class="oc-badge oc-object"></div>');
            chip.text(objectLabel(objectState));
            chip.addClass("is-" + String(objectState.name || "object"));
            cell.objectLayer.append(chip);
        });

        (state.players || []).forEach((playerState, index) => {
            const cell = this.cellFor(playerState.position);
            if (!cell) {
                return;
            }
            const chip = $('<div class="oc-badge oc-player"></div>');
            const carry = carriedLabel(playerState.held_object);
            const labelParts = [this.playerTag(index), orientationLabel(playerState.orientation)];
            if (carry) {
                labelParts.push(carry);
            }
            chip.text(labelParts.join(" "));
            if (index === this.trial.human_player_index) {
                chip.addClass("is-human");
            } else {
                chip.addClass("is-ai");
            }
            if (index === this.trial.target_agent_index) {
                chip.addClass("is-target");
            }
            cell.playerLayer.append(chip);
        });
    }

    close() {
        $("#" + this.container_id).empty();
        this.root = null;
        this.cells = {};
    }
}
