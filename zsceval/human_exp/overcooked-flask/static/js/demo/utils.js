import { getFrame } from "./spa/atlas.js";

// --- API ---

export function apiPost(url, payload) {
    return window.fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: payload ? JSON.stringify(payload) : null,
    }).then((response) => response.text().then((text) => {
        if (!response.ok) {
            throw new Error(text || ("Request failed: " + response.status));
        }
        return text ? JSON.parse(text) : {};
    }));
}

// --- Session storage ---

export function readJsonStorage(key) {
    try {
        return JSON.parse(window.sessionStorage.getItem(key) || "null");
    } catch (error) {
        return null;
    }
}

export function writeJsonStorage(key, value) {
    window.sessionStorage.setItem(key, JSON.stringify(value));
}

export function clearJsonStorage(key) {
    window.sessionStorage.removeItem(key);
}

// --- Array helpers ---

export function dedupe(values) {
    return values.filter((value, index) => values.indexOf(value) === index);
}

export function removeValue(values, value) {
    return values.filter((item) => item !== value);
}

// --- Game state builders ---

export function buildPreviewState(layoutGrid) {
    const players = [null, null];
    layoutGrid.forEach((row, y) => {
        row.split("").forEach((symbol, x) => {
            if (symbol === "1" || symbol === "2") {
                players[Number(symbol) - 1] = {
                    position: [x, y],
                    orientation: [0, 1],
                    held_object: null,
                };
            }
        });
    });
    return {
        players: players.filter(Boolean),
        objects: [],
        bonus_orders: [],
        all_orders: [],
        timestep: 0,
    };
}

export function buildMockState(players, objects, timestep) {
    return {
        players: players,
        objects: objects || [],
        bonus_orders: [],
        all_orders: [],
        timestep: timestep || 0,
    };
}

export function makePlayer(position, orientation, heldObject) {
    return {
        position: position,
        orientation: orientation,
        held_object: heldObject || null,
    };
}

export function makeObject(name, position, extra) {
    const obj = { name: name, position: position };
    return Object.assign(obj, extra || {});
}

export function makeTrajId(trialId) {
    const now = new Date();
    return trialId + "_" + [
        now.getFullYear(),
        now.getMonth() + 1,
        now.getDate(),
        now.getHours(),
        now.getMinutes(),
        now.getSeconds(),
    ].join("-");
}

// --- Atlas / sprite ---

export function atlasStyle(atlas, imageUrl, frameName, size, opacity) {
    const frame = getFrame(atlas, frameName);
    const base = { width: size, height: size };
    if (opacity !== undefined) {
        base.opacity = opacity;
    }
    if (!frame) {
        return base;
    }
    const frameDef = frame.frame;
    const scale = size / frameDef.w;
    return Object.assign(base, {
        backgroundImage: 'url("' + imageUrl + '")',
        backgroundRepeat: "no-repeat",
        backgroundPosition: "-" + String(frameDef.x * scale) + "px -" + String(frameDef.y * scale) + "px",
        backgroundSize: String(atlas.meta.size.w * scale) + "px " + String(atlas.meta.size.h * scale) + "px",
        imageRendering: "pixelated",
    });
}

// --- Form helpers ---

export function fieldValueFromEvent(field, event) {
    return event.target.value;
}

export function hasValue(value) {
    if (value === null || value === undefined) {
        return false;
    }
    if (typeof value === "boolean") {
        return true;
    }
    return String(value).trim().length > 0;
}

export function normalizeChoiceValue(value) {
    if (value === true) { return "yes"; }
    if (value === false) { return "no"; }
    return value === null || value === undefined ? "" : String(value);
}

export function normalizeChoiceLabel(label, value) {
    if (typeof label === "boolean") {
        return label ? "Yes" : "No";
    }
    if (label !== null && label !== undefined && String(label).trim().length > 0) {
        return String(label);
    }
    if (value === true) { return "Yes"; }
    if (value === false) { return "No"; }
    return String(value);
}

// --- Runner state ---

export function getDefaultRunnerState(session) {
    const trialIndexByStage = {};
    session.stages.forEach((stage) => {
        if (stage.type === "trial_block" || stage.type === "tutorial_lab") {
            trialIndexByStage[stage.id] = 0;
        }
    });
    return {
        stageIndex: 0,
        trialIndexByStage: trialIndexByStage,
        completedStages: [],
        completedTrials: {},
        trialSummaries: {},
        sectionResponses: {},
    };
}

export function nextIncompleteStageIndex(session, completedStages, fallbackIndex) {
    if (typeof fallbackIndex === "number" && fallbackIndex >= 0 && fallbackIndex < session.stages.length) {
        if (completedStages.indexOf(session.stages[fallbackIndex].id) < 0) {
            return fallbackIndex;
        }
    }
    for (let index = 0; index < session.stages.length; index += 1) {
        if (completedStages.indexOf(session.stages[index].id) < 0) {
            return index;
        }
    }
    return session.stages.length;
}

export function buildRunnerStateFromServer(session, bundle, userInfo) {
    const state = getDefaultRunnerState(session);
    const progress = bundle.progress || {};
    const savedTrials = bundle.saved_trials || {};
    const savedSections = bundle.saved_sections || {};
    let hintedIndex = 0;

    Object.keys(savedSections).forEach((sectionId) => {
        state.sectionResponses[sectionId] = savedSections[sectionId];
    });

    session.stages.forEach((stage, index) => {
        if (stage.type === "trial_block" || stage.type === "tutorial_lab") {
            let completedCount = 0;
            stage.trial_ids.forEach((trialId) => {
                const trialRecord = savedTrials[trialId];
                if (trialRecord && trialRecord.finished_at) {
                    completedCount += 1;
                    state.completedTrials[trialId] = true;
                    if (trialRecord.trial_summary) {
                        state.trialSummaries[trialId] = trialRecord.trial_summary;
                    }
                }
            });
            state.trialIndexByStage[stage.id] = completedCount;
            if (completedCount >= stage.trial_ids.length && stage.trial_ids.length > 0) {
                state.completedStages = dedupe(state.completedStages.concat([stage.id]));
            }
        } else if (savedSections[stage.id]) {
            state.completedStages = dedupe(state.completedStages.concat([stage.id]));
        }
        if (progress.current_stage_id && progress.current_stage_id === stage.id) {
            hintedIndex = index;
        }
    });

    state.completedStages = dedupe((progress.completed_stage_ids || []).concat(state.completedStages));
    if (!userInfo || !userInfo.name) {
        state.stageIndex = 0;
        return state;
    }
    state.stageIndex = nextIncompleteStageIndex(session, state.completedStages, hintedIndex);
    return state;
}

// --- Stage heading ---

export function splitStageHeading(stage) {
    if (!stage) {
        return { eyebrow: "", title: "Session complete" };
    }
    const blockMatch = stage.title.match(/^(Main Block \d+):\s*(.+)$/);
    if (blockMatch) {
        return { eyebrow: blockMatch[1], title: blockMatch[2] };
    }
    return { eyebrow: "", title: stage.title };
}

// --- Demo frame builders ---

export function buildPreviewExpectation(stepIndex, state, targetAgentIndex) {
    const targetPlayer = state && state.players ? state.players[targetAgentIndex] : null;
    const startPosition = targetPlayer ? targetPlayer.position : [8, 3];
    const route = [
        [startPosition[0] - 1, startPosition[1]],
        [startPosition[0] - 2, startPosition[1]],
        [startPosition[0] - 2, startPosition[1] - 1],
    ];
    const pages = [
        { pageIndex: 0, selectedSubtaskId: "collect_tomato", pathPoints: [], confidence: "", cursorLeft: "74%", cursorTop: "42%" },
        { pageIndex: 1, selectedSubtaskId: "collect_tomato", pathPoints: route, confidence: "", cursorLeft: "64%", cursorTop: "72%" },
        { pageIndex: 2, selectedSubtaskId: "collect_tomato", pathPoints: route, confidence: 6, cursorLeft: "78%", cursorTop: "86%" },
    ];
    return pages[stepIndex % pages.length];
}

export function buildModeDemoFrames() {
    return {
        observe: [
            buildMockState([makePlayer([8, 3], [0, -1]), makePlayer([4, 3], [1, 0])], [makeObject("tomato", [5, 1])], 0),
            buildMockState([makePlayer([8, 2], [0, -1]), makePlayer([5, 3], [1, 0])], [makeObject("tomato", [5, 1])], 1),
            buildMockState([makePlayer([8, 1], [-1, 0], makeObject("dish", [0, 0])), makePlayer([6, 3], [0, -1], makeObject("tomato", [0, 0]))], [], 2),
        ],
        collaborate: [
            buildMockState([makePlayer([4, 3], [1, 0], makeObject("tomato", [0, 0])), makePlayer([8, 3], [-1, 0])], [], 0),
            buildMockState([makePlayer([5, 3], [0, -1], makeObject("tomato", [0, 0])), makePlayer([8, 2], [-1, 0])], [], 1),
            buildMockState([makePlayer([5, 3], [1, 0]), makePlayer([8, 2], [-1, 0], makeObject("dish", [0, 0]))], [makeObject("soup", [6, 2], { _ingredients: [{ name: "tomato" }, { name: "tomato" }, { name: "tomato" }], is_cooking: true })], 2),
            buildMockState([makePlayer([7, 2], [-1, 0]), makePlayer([8, 2], [-1, 0], makeObject("dish", [0, 0]))], [makeObject("soup", [6, 2], { _ingredients: [{ name: "tomato" }, { name: "tomato" }, { name: "tomato" }], is_ready: true })], 3),
            buildMockState([makePlayer([7, 3], [1, 0]), makePlayer([10, 3], [-1, 0], makeObject("soup", [0, 0]))], [], 4),
        ],
        replay: [
            buildMockState([makePlayer([7, 2], [-1, 0]), makePlayer([8, 2], [-1, 0], makeObject("dish", [0, 0]))], [makeObject("soup", [6, 2], { _ingredients: [{ name: "tomato" }, { name: "tomato" }, { name: "tomato" }], is_ready: true })], 0),
            buildMockState([makePlayer([7, 2], [0, -1], makeObject("soup", [0, 0])), makePlayer([8, 2], [-1, 0])], [], 1),
            buildMockState([makePlayer([8, 2], [0, 1], makeObject("soup", [0, 0])), makePlayer([8, 3], [-1, 0])], [], 2),
            buildMockState([makePlayer([9, 3], [1, 0], makeObject("soup", [0, 0])), makePlayer([8, 3], [-1, 0])], [], 3),
            buildMockState([makePlayer([10, 3], [1, 0]), makePlayer([8, 3], [1, 0])], [], 4),
        ],
    };
}

export function buildTutorialLoopFrames() {
    return [
        { state: buildMockState([makePlayer([4, 3], [1, 0], makeObject("tomato", [0, 0])), makePlayer([8, 3], [-1, 0])], [], 0), activeSubtaskId: "collect_tomato" },
        { state: buildMockState([makePlayer([5, 3], [0, -1], makeObject("tomato", [0, 0])), makePlayer([8, 3], [-1, 0])], [], 1), activeSubtaskId: "load_pot" },
        { state: buildMockState([makePlayer([5, 3], [1, 0]), makePlayer([8, 2], [-1, 0])], [makeObject("soup", [6, 2], { _ingredients: [{ name: "tomato" }, { name: "tomato" }, { name: "tomato" }], is_cooking: true })], 2), activeSubtaskId: "manage_pot" },
        { state: buildMockState([makePlayer([7, 2], [-1, 0]), makePlayer([8, 2], [-1, 0], makeObject("dish", [0, 0]))], [makeObject("soup", [6, 2], { _ingredients: [{ name: "tomato" }, { name: "tomato" }, { name: "tomato" }], is_ready: true })], 3), activeSubtaskId: "pickup_soup" },
        { state: buildMockState([makePlayer([7, 3], [1, 0]), makePlayer([10, 3], [-1, 0], makeObject("soup", [0, 0]))], [], 4), activeSubtaskId: "serve_soup" },
    ];
}
