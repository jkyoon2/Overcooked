import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom";

import OvercookedBoard from "./spa/board.jsx";
import ExpectationComposer from "./spa/expectation.jsx";
import {
    CHEFS_ATLAS,
    CHEFS_IMAGE_URL,
    OBJECTS_ATLAS,
    OBJECTS_IMAGE_URL,
    getFrame,
} from "./spa/atlas.js";

const TIMESTEP_MS = 220;
const PREVIEW_LOOP_MS = 1280;
const REPLAY_TIMESTEP_MS = 260;
const DEFAULT_ACTION_IDX = 4;
const DIRECTION_KEY_CODES = {
    37: 3,
    38: 0,
    39: 2,
    40: 1,
    65: 3,
    68: 2,
    83: 1,
    87: 0,
};
const INTERACT_KEY_CODES = {
    13: true,
    32: true,
};
function apiPost(url, payload) {
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

function readJsonStorage(key) {
    try {
        return JSON.parse(window.sessionStorage.getItem(key) || "null");
    } catch (error) {
        return null;
    }
}

function writeJsonStorage(key, value) {
    window.sessionStorage.setItem(key, JSON.stringify(value));
}

function clearJsonStorage(key) {
    window.sessionStorage.removeItem(key);
}

function dedupe(values) {
    return values.filter((value, index) => values.indexOf(value) === index);
}

function removeValue(values, value) {
    return values.filter((item) => item !== value);
}

function createExpectationDraft() {
    return {
        selectedSubtaskId: "",
        pathPoints: [],
        confidence: "",
    };
}

function buildPreviewState(layoutGrid) {
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

function buildMockState(players, objects, timestep) {
    return {
        players: players,
        objects: objects || [],
        bonus_orders: [],
        all_orders: [],
        timestep: timestep || 0,
    };
}

function makePlayer(position, orientation, heldObject) {
    return {
        position: position,
        orientation: orientation,
        held_object: heldObject || null,
    };
}

function makeObject(name, position, extra) {
    const obj = {
        name: name,
        position: position,
    };
    return Object.assign(obj, extra || {});
}

function atlasStyle(atlas, imageUrl, frameName, size) {
    const frame = getFrame(atlas, frameName);
    if (!frame) {
        return {
            width: size,
            height: size,
        };
    }
    const frameDef = frame.frame;
    const scale = size / frameDef.w;
    return {
        width: size,
        height: size,
        backgroundImage: 'url("' + imageUrl + '")',
        backgroundRepeat: "no-repeat",
        backgroundPosition: "-" + String(frameDef.x * scale) + "px -" + String(frameDef.y * scale) + "px",
        backgroundSize: String(atlas.meta.size.w * scale) + "px " + String(atlas.meta.size.h * scale) + "px",
        imageRendering: "pixelated",
    };
}

function AtlasSprite(props) {
    return <div className={props.className || "mini-sprite"} style={atlasStyle(props.atlas, props.imageUrl, props.frameName, props.size || 40)} />;
}

function ChefVisual(props) {
    const direction = "SOUTH";
    const hatFrame = direction + "-" + props.hatColor + "hat.png";
    const bodyFrame = direction + (props.heldSuffix || "") + ".png";
    const size = props.size || 54;
    return (
        <div className="mini-sprite" style={{ width: size, height: size }}>
            <AtlasSprite atlas={CHEFS_ATLAS} imageUrl={CHEFS_IMAGE_URL} frameName={bodyFrame} size={size} className="mini-sprite" />
            <AtlasSprite atlas={CHEFS_ATLAS} imageUrl={CHEFS_IMAGE_URL} frameName={hatFrame} size={size} className="mini-sprite mini-sprite--overlay" />
        </div>
    );
}

function IngredientStrip(props) {
    return (
        <div className="ingredient-strip">
            {props.ingredients.map((ingredient, index) => (
                <React.Fragment key={ingredient + "-" + index}>
                    <AtlasSprite
                        atlas={OBJECTS_ATLAS}
                        imageUrl={OBJECTS_IMAGE_URL}
                        frameName={ingredient === "onion" ? "onion.png" : "tomato.png"}
                        size={38}
                        className="mini-sprite"
                    />
                    {index < props.ingredients.length - 1 ? <span className="ingredient-strip__plus">+</span> : null}
                </React.Fragment>
            ))}
        </div>
    );
}

function getDefaultRunnerState(session) {
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

function nextIncompleteStageIndex(session, completedStages, fallbackIndex) {
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

function buildRunnerStateFromServer(session, bundle, userInfo) {
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

function makeTrajId(trialId) {
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

function fieldValueFromEvent(field, event) {
    if (field.type === "radio") {
        return event.target.value;
    }
    return event.target.value;
}

function hasValue(value) {
    if (value === null || value === undefined) {
        return false;
    }
    if (typeof value === "boolean") {
        return true;
    }
    return String(value).trim().length > 0;
}

function normalizeChoiceValue(value) {
    if (value === true) {
        return "yes";
    }
    if (value === false) {
        return "no";
    }
    return value === null || value === undefined ? "" : String(value);
}

function normalizeChoiceLabel(label, value) {
    if (typeof label === "boolean") {
        return label ? "Yes" : "No";
    }
    if (label !== null && label !== undefined && String(label).trim().length > 0) {
        return String(label);
    }
    if (value === true) {
        return "Yes";
    }
    if (value === false) {
        return "No";
    }
    return String(value);
}

function LikertBar(props) {
    const ticks = [];
    for (let value = props.min; value <= props.max; value += 1) {
        ticks.push(value);
    }
    return (
        <div className="likert-bar">
            <div className="likert-bar__label">{props.label}</div>
            <div className="likert-bar__track">
                {ticks.map((value) => (
                    <button
                        className={"likert-bar__tick" + (String(props.value) === String(value) ? " is-selected" : "")}
                        key={props.fieldId + "-" + value}
                        type="button"
                        onClick={() => props.onChange(value)}
                    >
                        <span>{value}</span>
                    </button>
                ))}
            </div>
            <div className="likert-bar__legend">
                <span>{props.leftLabel || ""}</span>
                <span>{props.rightLabel || ""}</span>
            </div>
        </div>
    );
}

function InputField(props) {
    const field = props.field;
    const value = normalizeChoiceValue(props.value);

    if (field.type === "radio") {
        const isInlineRow = field.options.length <= 4;
        return (
            <div className="form-field">
                <label className="form-field__label">{field.label}{field.required ? " *" : ""}</label>
                <div className={"choice-grid" + (isInlineRow ? " choice-grid--inline" : "")}>
                    {field.options.map((option) => {
                        const optionValue = normalizeChoiceValue(option.value);
                        const optionLabel = normalizeChoiceLabel(option.label, option.value);
                        return (
                        <button
                            className={"choice-pill choice-pill--segmented" + (String(value) === String(optionValue) ? " is-selected" : "")}
                            key={field.id + "-" + optionValue}
                            type="button"
                            onClick={() => props.onChange({ target: { value: optionValue } })}
                        >
                            <span className="choice-pill__label">{optionLabel}</span>
                        </button>
                        );
                    })}
                </div>
            </div>
        );
    }

    if (field.type === "scale" || field.type === "scale_bar") {
        return (
            <LikertBar
                fieldId={field.id}
                label={field.label}
                min={field.min}
                max={field.max}
                value={value}
                leftLabel={field.left_label}
                rightLabel={field.right_label}
                onChange={(nextValue) => props.onChange({ target: { value: String(nextValue) } })}
            />
        );
    }

    if (field.type === "textarea") {
        return (
            <div className="form-field">
                <label className="form-field__label">{field.label}{field.required ? " *" : ""}</label>
                <textarea className="form-textarea" value={value || ""} onChange={props.onChange} />
            </div>
        );
    }

    return (
        <div className="form-field">
            <label className="form-field__label">{field.label}{field.required ? " *" : ""}</label>
            <input className="form-input" type={field.type} min={field.min} max={field.max} value={value || ""} onChange={props.onChange} />
        </div>
    );
}

function ModalShell(props) {
    if (!props.open) {
        return null;
    }
    return (
        <div className="modal-shell">
            <div className="modal-card">
                <div className="modal-card__head">
                    <div>
                        {props.eyebrow ? <div className="panel-eyebrow">{props.eyebrow}</div> : null}
                        <h3>{props.title}</h3>
                    </div>
                </div>
                <div className="modal-card__body">{props.children}</div>
            </div>
        </div>
    );
}

function splitStageHeading(stage) {
    if (!stage) {
        return { eyebrow: "", title: "Session complete" };
    }
    const blockMatch = stage.title.match(/^(Main Block \d+):\s*(.+)$/);
    if (blockMatch) {
        return {
            eyebrow: blockMatch[1],
            title: blockMatch[2],
        };
    }
    return {
        eyebrow: "",
        title: stage.title,
    };
}

function TopProgress(props) {
    return (
        <section className="app-rail">
            <div className="app-rail__header">
                <div className="brand-block">
                    <div className="brand-badge">AI</div>
                    <div>
                        <h1>{props.session.title}</h1>
                        <p>{props.session.subtitle}</p>
                    </div>
                </div>
            </div>
            <div className="stage-progress" style={{ gridTemplateColumns: "repeat(" + String(props.session.stages.length) + ", minmax(0, 1fr))" }}>
                {props.session.stages.map((stage, index) => {
                    const isComplete = props.runnerState.completedStages.indexOf(stage.id) >= 0;
                    const isCurrent = props.runnerState.stageIndex === index;
                    return <div className={"stage-progress__segment" + (isCurrent ? " is-current" : "") + (isComplete ? " is-complete" : "")} key={stage.id} />;
                })}
            </div>
        </section>
    );
}

function StageHeader(props) {
    const heading = splitStageHeading(props.stage);
    const body = props.stage && props.stage.body && props.stage.body.length
        ? props.stage.body[0]
        : (props.stage && props.stage.type === "survey" ? "Answer a few short questions, then continue." : "Continue when you are ready.");
    return (
        <header className="stage-header">
            <div>
                {heading.eyebrow ? <div className="panel-eyebrow">{heading.eyebrow}</div> : null}
                <h2>{heading.title}</h2>
                <p>{body}</p>
            </div>
        </header>
    );
}

function PreviewCursor(props) {
    return (
        <div
            className={"preview-cursor" + (props.isClicking ? " is-clicking" : "")}
            style={{
                left: props.left,
                top: props.top,
            }}
        />
    );
}

function buildPreviewExpectation(stepIndex, state, targetAgentIndex) {
    const targetPlayer = state && state.players ? state.players[targetAgentIndex] : null;
    const startPosition = targetPlayer ? targetPlayer.position : [8, 3];
    const route = [
        [startPosition[0] - 1, startPosition[1]],
        [startPosition[0] - 2, startPosition[1]],
        [startPosition[0] - 2, startPosition[1] - 1],
    ];
    const pages = [
        {
            pageIndex: 0,
            selectedSubtaskId: "collect_tomato",
            pathPoints: [],
            confidence: "",
            cursorLeft: "74%",
            cursorTop: "42%",
        },
        {
            pageIndex: 1,
            selectedSubtaskId: "collect_tomato",
            pathPoints: route,
            confidence: "",
            cursorLeft: "64%",
            cursorTop: "72%",
        },
        {
            pageIndex: 2,
            selectedSubtaskId: "collect_tomato",
            pathPoints: route,
            confidence: 6,
            cursorLeft: "78%",
            cursorTop: "86%",
        },
    ];
    return pages[stepIndex % pages.length];
}

function LiveProbePreview(props) {
    const previewState = buildPreviewExpectation(props.stepIndex, props.state, props.targetAgentIndex);
    return (
        <div className="preview-probe">
            <div className="preview-probe__modal preview-probe__modal--live">
                <div className="preview-probe__composer">
                    <ExpectationComposer
                        readOnly
                        compact
                        previewSubtaskLimit={4}
                        sketchTileSize={24}
                        forcedPageIndex={previewState.pageIndex}
                        title={props.title}
                        probeIndex={1}
                        probeTotal={3}
                        prompt="What do you expect from the AI chef?"
                        sketchPrompt="Sketch the route you expected."
                        confidencePrompt="How confident are you?"
                        subtaskOptions={props.subtaskOptions}
                        layoutGrid={props.layoutGrid}
                        state={props.state}
                        targetAgentIndex={props.targetAgentIndex}
                        selectedSubtaskId={previewState.selectedSubtaskId}
                        pathPoints={previewState.pathPoints}
                        confidence={previewState.confidence}
                        onSubmit={() => {}}
                    />
                </div>
            </div>
            <PreviewCursor left={previewState.cursorLeft} top={previewState.cursorTop} isClicking />
        </div>
    );
}

function ObserveHeroPreview(props) {
    const [cursor, setCursor] = useState(0);
    useEffect(() => {
        const timerId = window.setInterval(() => {
            setCursor((current) => current + 1);
        }, PREVIEW_LOOP_MS);
        return function cleanup() {
            window.clearInterval(timerId);
        };
    }, []);
    const frameIndex = cursor % props.frames.length;
    const currentState = props.frames[frameIndex];
    const previewState = buildPreviewExpectation(cursor, currentState, props.trial.target_agent_index);
    const previewTrial = Object.assign({}, props.trial, {
        show_target_highlight: false,
    });

    return (
        <div className="observe-intro-preview">
            <div className="observe-intro-preview__board">
                <OvercookedBoard
                    layoutGrid={props.layoutGrid}
                    state={currentState}
                    trial={previewTrial}
                    tileSize={props.tileSize || (props.showProbe ? 44 : 56)}
                />
            </div>
            {props.showProbe ? (
                <div className="observe-intro-preview__modal">
                    <div className="modal-card modal-card--preview">
                        <div className="modal-card__head">
                            <div>
                                <div className="panel-eyebrow">Probe</div>
                                <h3>What do you expect next?</h3>
                            </div>
                        </div>
                        <div className="modal-card__body">
                            <ExpectationComposer
                                readOnly
                                compact
                                previewSubtaskLimit={2}
                                sketchTileSize={20}
                                forcedPageIndex={0}
                                title="Report what you expect from the AI chef"
                                probeIndex={1}
                                probeTotal={props.trial.probe ? props.trial.probe.count : 3}
                                prompt={props.trial.probe ? props.trial.probe.prompt : "What do you expect from the AI chef?"}
                                sketchPrompt={props.trial.probe ? props.trial.probe.sketch_prompt : "Sketch the route you expected."}
                                confidencePrompt={props.trial.probe ? props.trial.probe.confidence_prompt : "How confident are you?"}
                                subtaskOptions={props.subtaskOptions}
                                layoutGrid={props.layoutGrid}
                                state={currentState}
                                targetAgentIndex={props.trial.target_agent_index}
                                selectedSubtaskId={previewState.selectedSubtaskId}
                                pathPoints={previewState.pathPoints}
                                confidence={previewState.confidence}
                                onSubmit={() => {}}
                            />
                        </div>
                    </div>
                </div>
            ) : null}
        </div>
    );
}

function AnimatedSceneCard(props) {
    const [cursor, setCursor] = useState(0);
    useEffect(() => {
        const timerId = window.setInterval(() => {
            setCursor((current) => current + 1);
        }, PREVIEW_LOOP_MS);
        return function cleanup() {
            window.clearInterval(timerId);
        };
    }, []);
    const frameIndex = cursor % props.frames.length;
    const showPopup = props.showProbe && (cursor % 4 !== 0);
    const previewTrial = Object.assign({}, props.trial, {
        show_target_highlight: Boolean(props.showHighlight),
    });
    return (
        <div className="mode-card">
            {!props.hideCopy ? (
                <div className="mode-card__copy">
                    <h3>{props.title}</h3>
                    {props.description ? <p>{props.description}</p> : null}
                </div>
            ) : null}
            <div className="mode-card__preview">
                {props.previewBadge ? <div className="mode-card__badge">{props.previewBadge}</div> : null}
                {props.previewVisual ? <div className="mode-card__asset">{props.previewVisual}</div> : null}
                <OvercookedBoard layoutGrid={props.layoutGrid} state={props.frames[frameIndex]} trial={previewTrial} />
                {showPopup ? (
                    <LiveProbePreview
                        stepIndex={cursor}
                        title={props.probeTitle || "What do you expect next?"}
                        layoutGrid={props.layoutGrid}
                        state={props.frames[frameIndex]}
                        targetAgentIndex={props.trial.target_agent_index}
                        subtaskOptions={props.subtaskOptions}
                    />
                ) : null}
            </div>
        </div>
    );
}

function LoopingBoardPreview(props) {
    const [cursor, setCursor] = useState(0);
    const frames = props.frames || [];
    useEffect(() => {
        if (!frames.length) {
            return undefined;
        }
        const timerId = window.setInterval(() => {
            setCursor((current) => current + 1);
        }, props.intervalMs || PREVIEW_LOOP_MS);
        return function cleanup() {
            window.clearInterval(timerId);
        };
    }, [frames.length, props.intervalMs]);
    const frameEntry = frames.length ? frames[cursor % frames.length] : null;
    const currentState = frameEntry && frameEntry.state ? frameEntry.state : frameEntry;
    const activeStepId = frameEntry && frameEntry.activeSubtaskId ? frameEntry.activeSubtaskId : "";
    const previewTrial = Object.assign({}, props.trial || {}, {
        show_target_highlight: Boolean(props.showHighlight),
    });

    return (
        <div className={"looping-board-preview" + (props.compact ? " is-compact" : "")}>
            <div className="looping-board-preview__board">
                <OvercookedBoard
                    layoutGrid={props.layoutGrid}
                    state={currentState}
                    trial={previewTrial}
                    tileSize={props.tileSize || undefined}
                />
            </div>
            {props.steps && props.steps.length ? (
                <div className="looping-board-preview__steps">
                    {props.steps.map((step) => (
                        <div
                            className={"looping-board-preview__step" + (step.id === activeStepId ? " is-active" : "")}
                            key={step.id}
                        >
                            {step.label}
                        </div>
                    ))}
                </div>
            ) : null}
            {props.caption ? <p className="looping-board-preview__caption">{props.caption}</p> : null}
        </div>
    );
}

function ModePlaybackCard(props) {
    return (
        <div className="mode-card mode-card--playback">
            <div className="mode-card__copy">
                <h3>{props.title}</h3>
                <p>{props.description}</p>
            </div>
            <div className="mode-card__asset-stage mode-card__asset-stage--playback">
                <LoopingBoardPreview
                    compact
                    frames={props.frames}
                    layoutGrid={props.layoutGrid}
                    trial={props.trial}
                    caption={props.caption}
                />
            </div>
        </div>
    );
}

function buildModeDemoFrames(layoutGrid) {
    return {
        observe: [
            buildMockState([makePlayer([8, 3], [0, -1]), makePlayer([4, 3], [1, 0])], [makeObject("tomato", [5, 1])], 0),
            buildMockState([makePlayer([8, 2], [0, -1]), makePlayer([5, 3], [1, 0])], [makeObject("tomato", [5, 1])], 1),
            buildMockState([makePlayer([8, 1], [-1, 0], makeObject("dish", [0, 0])), makePlayer([6, 3], [0, -1], makeObject("tomato", [0, 0]))], [], 2),
        ],
        collaborate: [
            buildMockState([makePlayer([4, 3], [1, 0], makeObject("tomato", [0, 0])), makePlayer([8, 3], [-1, 0])], [], 0),
            buildMockState([makePlayer([5, 3], [0, -1], makeObject("tomato", [0, 0])), makePlayer([8, 2], [-1, 0])], [], 1),
            buildMockState([makePlayer([6, 3], [0, -1]), makePlayer([8, 2], [-1, 0], makeObject("dish", [0, 0]))], [makeObject("soup", [6, 2], { _ingredients: [{ name: "tomato" }, { name: "tomato" }, { name: "tomato" }], is_cooking: true })], 2),
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

function buildTutorialLoopFrames(layoutGrid) {
    return [
        {
            state: buildMockState([makePlayer([4, 3], [1, 0], makeObject("tomato", [0, 0])), makePlayer([8, 3], [-1, 0])], [], 0),
            activeSubtaskId: "collect_tomato",
        },
        {
            state: buildMockState([makePlayer([5, 3], [0, -1], makeObject("tomato", [0, 0])), makePlayer([8, 3], [-1, 0])], [], 1),
            activeSubtaskId: "load_pot",
        },
        {
            state: buildMockState([makePlayer([6, 3], [0, -1]), makePlayer([8, 2], [-1, 0])], [makeObject("soup", [6, 2], { _ingredients: [{ name: "tomato" }, { name: "tomato" }, { name: "tomato" }], is_cooking: true })], 2),
            activeSubtaskId: "manage_pot",
        },
        {
            state: buildMockState([makePlayer([7, 2], [-1, 0]), makePlayer([8, 2], [-1, 0], makeObject("dish", [0, 0]))], [makeObject("soup", [6, 2], { _ingredients: [{ name: "tomato" }, { name: "tomato" }, { name: "tomato" }], is_ready: true })], 3),
            activeSubtaskId: "pickup_soup",
        },
        {
            state: buildMockState([makePlayer([7, 3], [1, 0]), makePlayer([10, 3], [-1, 0], makeObject("soup", [0, 0]))], [], 4),
            activeSubtaskId: "serve_soup",
        },
    ];
}

function ObserveSceneAsset() {
    return (
        <div className="mode-scene mode-scene--observe">
            <div className="mode-scene__spotlight">
                <ChefVisual hatColor="orange" size={60} />
            </div>
            <div className="mode-scene__connector" />
            <div className="mode-scene__stack">
                <div className="mode-scene__pair">
                    <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="tomato.png" size={42} className="mini-sprite" />
                    <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="dish.png" size={42} className="mini-sprite" />
                    <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="soup-tomato-dish.png" size={42} className="mini-sprite" />
                </div>
            <div className="mode-scene__caption">Watch the AI chef.</div>
            </div>
        </div>
    );
}

function CollaborateSceneAsset() {
    return (
        <div className="mode-scene mode-scene--collaborate">
            <div className="mode-scene__pair">
                <ChefVisual hatColor="gray" size={58} />
                <div className="mode-scene__plus">+</div>
                <ChefVisual hatColor="orange" size={58} />
            </div>
            <div className="mode-scene__pair">
                <IngredientStrip ingredients={["tomato", "tomato", "onion"]} />
                <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="soup-tomato-dish.png" size={46} className="mini-sprite" />
            </div>
            <div className="mode-scene__caption">Cook side by side with the AI chef.</div>
        </div>
    );
}

function ReplaySceneAsset() {
    return (
        <div className="mode-scene mode-scene--replay">
            <div className="mode-scene__pair">
                <ChefVisual hatColor="gray" size={56} />
                <div className="mode-scene__rewind">↺</div>
                <ChefVisual hatColor="orange" size={56} />
            </div>
            <div className="mode-scene__path">
                <span />
                <span />
                <span />
                <span />
            </div>
            <div className="mode-scene__caption">Replay the collaborated scene and annotate what you expected.</div>
        </div>
    );
}

function ModeAssetCard(props) {
    return (
        <div className="mode-card mode-card--asset-only">
            <div className="mode-card__copy">
                <h3>{props.title}</h3>
                <p>{props.description}</p>
            </div>
            <div className="mode-card__asset-stage">
                {props.asset}
            </div>
        </div>
    );
}

function WelcomeStage(props) {
    const demos = useMemo(() => buildModeDemoFrames(props.session.layout_grid), [props.session.layout_grid]);
    const observeTrial = props.session.trials.observe_1 || props.session.trials.tutorial_team;
    return (
        <div className="stage-stack">
            <div className="hero-task-grid">
                <div className="hero-task-card">
                    <div className="hero-task-card__preview">
                        <ObserveHeroPreview
                            frames={demos.observe}
                            layoutGrid={props.session.layout_grid}
                            trial={observeTrial}
                            subtaskOptions={props.session.subtask_options}
                            showProbe={false}
                            tileSize={48}
                        />
                    </div>
                    <h3>Cook with a AI chef</h3>
                    <p>Cook onion/tomato soup and deliver it with a AI chef</p>
                </div>
                <div className="hero-task-card">
                    <div className="hero-task-card__preview">
                        <ObserveHeroPreview
                            frames={demos.observe}
                            layoutGrid={props.session.layout_grid}
                            trial={observeTrial}
                            subtaskOptions={props.session.subtask_options}
                            showProbe
                        />
                    </div>
                    <h3>Submit your expected strategy of the AI chef</h3>
                    <p>Choose the subtask you expect for the AI chef and sketch the route</p>
                </div>
            </div>
            <div className="stage-actions">
                <button className="primary-button" type="button" onClick={() => props.onContinue({ seen: true })}>Start briefing</button>
            </div>
        </div>
    );
}

function ModeOverviewStage(props) {
    const demos = useMemo(() => buildModeDemoFrames(props.session.layout_grid), [props.session.layout_grid]);
    const collaborateTrial = props.session.trials.collaborate_1 || props.session.trials.tutorial_team || props.session.trials.observe_1;
    const replayTrial = props.session.trials.replay_1 || collaborateTrial;
    return (
        <div className="stage-stack">
            <div className="mode-grid">
                <ModePlaybackCard
                    title="Collaborate"
                    description="Cook with the AI chef and answer the probe."
                    layoutGrid={props.session.layout_grid}
                    frames={demos.collaborate}
                    trial={collaborateTrial}
                    caption="Live play"
                />
                <ModePlaybackCard
                    title="Replay + Annotation"
                    description="Replay the recent scene and annotate what you expected."
                    layoutGrid={props.session.layout_grid}
                    frames={demos.replay}
                    trial={replayTrial}
                    caption="Replay"
                />
            </div>
            <div className="stage-actions">
                <button className="primary-button" type="button" onClick={() => props.onContinue({ reviewed_modes: true })}>Continue</button>
            </div>
        </div>
    );
}

function ConsentStage(props) {
    const initialValues = {};
    props.intakeForm.sections.forEach((section) => {
        section.fields.forEach((field) => {
            initialValues[field.id] = props.userInfo && Object.prototype.hasOwnProperty.call(props.userInfo, field.id)
                ? normalizeChoiceValue(props.userInfo[field.id])
                : "";
        });
    });
    initialValues.consent = false;
    const [values, setValues] = useState(initialValues);
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState("");

    function updateField(field, event) {
        const nextValues = Object.assign({}, values);
        nextValues[field.id] = fieldValueFromEvent(field, event);
        setValues(nextValues);
        setError("");
    }

    function submit() {
        const missing = [];
        props.intakeForm.sections.forEach((section) => {
            section.fields.forEach((field) => {
                if (field.required && !hasValue(values[field.id])) {
                    missing.push(field.label);
                }
            });
        });
        if (!values.consent) {
            missing.push("Consent");
        }
        if (missing.length) {
            setError("Please complete all required items.");
            return;
        }
        setError("");
        setSubmitting(true);
        apiPost("/create_questionnaire_before_game", values).then(() => {
            writeJsonStorage("before_game", values);
            return Promise.resolve(props.onSubmit(values));
        }).catch((submitError) => {
            setSubmitting(false);
            setError(String(submitError.message || submitError));
        });
    }

    return (
        <div className="stage-stack">
            <section className="panel-card panel-card--scroll">
                <div className="panel-stack">
                    {props.intakeForm.sections.map((section) => (
                        <div className="section-block" key={section.id}>
                            <div className="section-block__title">{section.title}</div>
                            {section.fields.map((field) => (
                                <InputField key={field.id} field={field} value={values[field.id]} onChange={(event) => updateField(field, event)} />
                            ))}
                        </div>
                    ))}
                    <label className={"toggle-card" + (values.consent ? " is-selected" : "")}>
                        <input
                            type="checkbox"
                            checked={Boolean(values.consent)}
                            onChange={() => {
                                setValues(Object.assign({}, values, { consent: !values.consent }));
                                setError("");
                            }}
                        />
                        <span className={"toggle-card__check" + (values.consent ? " is-checked" : "")}>{values.consent ? "✓" : ""}</span>
                        <span>I agree to participate in this study.</span>
                    </label>
                    {error ? <div className="callout callout--danger">{error}</div> : null}
                    <div className="stage-actions">
                        <button className="primary-button" type="button" disabled={submitting} onClick={submit}>{submitting ? "Saving..." : "Continue"}</button>
                    </div>
                </div>
            </section>
        </div>
    );
}

function OnboardingStage(props) {
    const guideSteps = [
        { id: "move", title: "Move", body: "Use the arrow keys to move your chef." },
        { id: "interact", title: "Interact", body: "Press the space bar to pick up, place, plate, or serve the object (tomato, onion, dish, soup)." },
        { id: "probe", title: "Probe", body: "When a probe appears, choose the AI chef's next subtask, draw the route, and rate your confidence." },
    ];
    const previewSteps = [
        { id: "collect_tomato", label: "Collect tomato" },
        { id: "load_pot", label: "Load pot" },
        { id: "manage_pot", label: "Manage pot" },
        { id: "pickup_soup", label: "Pick up soup" },
        { id: "serve_soup", label: "Serve soup" },
    ];
    const demoFrames = useMemo(() => buildTutorialLoopFrames(props.session.layout_grid), [props.session.layout_grid]);
    const probeFrames = useMemo(() => demoFrames.map((frame) => frame.state), [demoFrames]);
    const [guideIndex, setGuideIndex] = useState(-1);
    const activeGuide = guideSteps[guideIndex] || null;

    function openGuide() {
        setGuideIndex(0);
    }

    function finishOnboarding() {
        props.onContinue({
            reviewed_tutorial: true,
        });
    }

    return (
        <div className="stage-stack">
            <section className="panel-card">
                <div className="panel-stack">
                    <div className="tutorial-preview-shell">
                        <div className="tutorial-preview-shell__header">
                            <div>
                                <div className="panel-eyebrow">Before practice</div>
                                <h3>Watch one full cooking loop</h3>
                            </div>
                            <p>The active subtask lights up as the team moves through the recipe.</p>
                        </div>
                        <LoopingBoardPreview
                            frames={demoFrames}
                            steps={previewSteps}
                            layoutGrid={props.session.layout_grid}
                            trial={{ human_player_index: 1, target_agent_index: 0, show_target_highlight: false }}
                            caption="Gray chef: you. Orange chef: AI teammate."
                        />
                        <div className="key-row">
                            <div className="key-chip">Arrow keys: move</div>
                            <div className="key-chip">Space: interact</div>
                        </div>
                    </div>
                    <div className="stage-actions">
                        <button className="primary-button" type="button" onClick={openGuide}>Next</button>
                    </div>
                </div>
            </section>
            <ModalShell open={guideIndex >= 0 && guideIndex < guideSteps.length} eyebrow="Tutorial" title={activeGuide ? activeGuide.title : "Tutorial"}>
                <div className="panel-stack">
                    <p>{activeGuide ? activeGuide.body : ""}</p>
                    <div className={"guide-visual" + (activeGuide && activeGuide.id === "probe" ? " guide-visual--board" : "")}>
                        {activeGuide && activeGuide.id === "move" ? (
                            <div className="key-demo">
                                <div className="keycap">↑</div>
                                <div className="keycap">←</div>
                                <div className="keycap">↓</div>
                                <div className="keycap">→</div>
                            </div>
                        ) : null}
                        {activeGuide && activeGuide.id === "interact" ? (
                            <div className="guide-visual__interact">
                                <div className="key-demo">
                                    <div className="keycap">Space</div>
                                </div>
                                <div className="ingredient-strip">
                                    <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="tomato.png" size={34} className="mini-sprite" />
                                    <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="onion.png" size={34} className="mini-sprite" />
                                    <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="dish.png" size={34} className="mini-sprite" />
                                    <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="soup-tomato-dish.png" size={34} className="mini-sprite" />
                                </div>
                            </div>
                        ) : null}
                        {activeGuide && activeGuide.id === "probe" ? (
                            <ObserveHeroPreview
                                frames={probeFrames}
                                layoutGrid={props.session.layout_grid}
                                trial={{ human_player_index: 1, target_agent_index: 0, probe: { count: 3, prompt: "What do you expect from the AI chef?", sketch_prompt: "Sketch the route you expected.", confidence_prompt: "How confident are you?" }, show_target_highlight: false }}
                                subtaskOptions={props.session.subtask_options}
                                showProbe
                            />
                        ) : null}
                    </div>
                    <div className="stage-actions">
                        {guideIndex < guideSteps.length - 1 ? (
                            <button className="primary-button" type="button" onClick={() => setGuideIndex(guideIndex + 1)}>Next</button>
                        ) : (
                            <button className="primary-button" type="button" onClick={finishOnboarding}>Start tutorial</button>
                        )}
                    </div>
                </div>
            </ModalShell>
        </div>
    );
}

function TrialStats(props) {
    const runtime = props.runtime || {};
    const probes = runtime.probe_records || [];
    return (
        <div className="stats-grid">
            <div className="stat-card"><span>Score</span><strong>{runtime.score || 0}</strong></div>
            <div className="stat-card"><span>Time left</span><strong>{runtime.time_left || 0}s</strong></div>
            <div className="stat-card"><span>Steps</span><strong>{runtime.step_count || 0}</strong></div>
            <div className="stat-card"><span>Probes</span><strong>{probes.length}</strong></div>
        </div>
    );
}

function PostTrialRatingModal(props) {
    const [values, setValues] = useState(props.initialValues || {});
    useEffect(() => {
        setValues(props.initialValues || {});
    }, [props.initialValues, props.trial.id]);

    function update(field, event) {
        const nextValues = Object.assign({}, values);
        nextValues[field.id] = fieldValueFromEvent(field, event);
        setValues(nextValues);
    }

    const complete = props.trial.post_trial_questions.every((question) => String(values[question.id] || "").trim());
    return (
        <ModalShell open={props.open} eyebrow="Trial rating" title="How did this trial feel?">
            <div className="panel-stack">
                {props.trial.post_trial_questions.map((question) => (
                    <InputField key={question.id} field={question} value={values[question.id] || ""} onChange={(event) => update(question, event)} />
                ))}
                <div className="stage-actions">
                    <button className="primary-button" type="button" disabled={!complete} onClick={() => props.onSubmit(values)}>Save trial rating</button>
                </div>
            </div>
        </ModalShell>
    );
}

function useBufferedInput(enabled) {
    const heldDirectionsRef = useRef([]);
    const interactBufferRef = useRef(0);

    useEffect(() => {
        if (!enabled) {
            return undefined;
        }
        function clearInputs() {
            heldDirectionsRef.current = [];
            interactBufferRef.current = 0;
        }
        function onKeyDown(event) {
            const directionAction = DIRECTION_KEY_CODES[event.which];
            if (directionAction !== undefined) {
                heldDirectionsRef.current = removeValue(heldDirectionsRef.current, directionAction).concat([directionAction]);
                event.preventDefault();
                return;
            }
            if (INTERACT_KEY_CODES[event.which]) {
                interactBufferRef.current = Math.min(interactBufferRef.current + 2, 4);
                event.preventDefault();
            }
        }
        function onKeyUp(event) {
            const directionAction = DIRECTION_KEY_CODES[event.which];
            if (directionAction === undefined) {
                return;
            }
            heldDirectionsRef.current = removeValue(heldDirectionsRef.current, directionAction);
        }
        document.addEventListener("keydown", onKeyDown);
        document.addEventListener("keyup", onKeyUp);
        window.addEventListener("blur", clearInputs);
        return function cleanup() {
            document.removeEventListener("keydown", onKeyDown);
            document.removeEventListener("keyup", onKeyUp);
            window.removeEventListener("blur", clearInputs);
        };
    }, [enabled]);

    function consumeAction() {
        if (interactBufferRef.current > 0) {
            interactBufferRef.current -= 1;
            return 5;
        }
        if (heldDirectionsRef.current.length) {
            return heldDirectionsRef.current[heldDirectionsRef.current.length - 1];
        }
        return DEFAULT_ACTION_IDX;
    }

    return {
        consumeAction: consumeAction,
    };
}

function InteractiveTrialRunner(props) {
    const [runtime, setRuntime] = useState(null);
    const [status, setStatus] = useState("booting");
    const [error, setError] = useState("");
    const [probe, setProbe] = useState(null);
    const [probeSubmitted, setProbeSubmitted] = useState(false);
    const [probeDraftKey, setProbeDraftKey] = useState(0);
    const [finishPayload, setFinishPayload] = useState(null);
    const statusRef = useRef("booting");
    const input = useBufferedInput(props.trial.human_player_index !== null && props.trial.human_player_index !== undefined);

    useEffect(() => {
        statusRef.current = status;
    }, [status]);

    useEffect(() => {
        let mounted = true;
        apiPost("/start_trial", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
        }).then((startPayload) => {
            if (!mounted) {
                return;
            }
            setRuntime(startPayload.runtime);
            setStatus("running");
        }).catch((startError) => {
            if (!mounted) {
                return;
            }
            setError(String(startError.message || startError));
            setStatus("error");
        });
        return function cleanup() {
            mounted = false;
        };
    }, [props.trial.id, props.userInfo]);

    useEffect(() => {
        if (status !== "running") {
            return undefined;
        }
        let cancelled = false;
        let timerId = null;

        function schedule(delay) {
            timerId = window.setTimeout(stepLoop, delay);
        }

        function stepLoop() {
            if (cancelled || statusRef.current !== "running") {
                return;
            }
            const startedAt = window.performance.now();
            apiPost("/step_trial", {
                user_info: props.userInfo,
                trial_id: props.trial.id,
                human_action_idx: input.consumeAction(),
            }).then((response) => {
                if (cancelled) {
                    return;
                }
                if (response.probe_pending) {
                    setRuntime(response.runtime);
                    setProbe(response.probe);
                    setProbeSubmitted(false);
                    setProbeDraftKey((current) => current + 1);
                    setStatus("probe");
                    return;
                }
                setRuntime(response.runtime);
                if (response.done) {
                    finishTrial();
                    return;
                }
                schedule(Math.max(0, TIMESTEP_MS - (window.performance.now() - startedAt)));
            }).catch((stepError) => {
                if (cancelled) {
                    return;
                }
                setError(String(stepError.message || stepError));
                setStatus("error");
            });
        }

        schedule(0);
        return function cleanup() {
            cancelled = true;
            window.clearTimeout(timerId);
        };
    }, [status, props.trial.id, props.userInfo, input]);

    function finishTrial() {
        setStatus("finishing");
        apiPost("/finish_episode", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
            traj_id: makeTrajId(props.trial.id),
            summary: { client: "react-spa" },
        }).then((payload) => {
            setFinishPayload(payload);
            setStatus("rating");
        }).catch((finishError) => {
            setError(String(finishError.message || finishError));
            setStatus("error");
        });
    }

    function submitProbe(expectation) {
        apiPost("/submit_probe", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
            selected_subtask_id: expectation.selectedSubtaskId,
            extra: {
                expected_path: expectation.expectedPath,
                confidence: expectation.confidence,
                start_position: expectation.startPosition,
                input_mode: "step_wizard",
            },
        }).then((response) => {
            return apiPost("/save_trial_data", {
                user_info: props.userInfo,
                trial_id: props.trial.id,
                updates: {
                    latest_probe: response.probe_record,
                },
            });
        }).then(() => {
            setProbeSubmitted(true);
        }).catch((submitError) => {
            setError(String(submitError.message || submitError));
            setStatus("error");
        });
    }

    function resumeAfterProbe() {
        apiPost("/resume_trial_after_probe", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
        }).then((response) => {
            setRuntime(response.runtime);
            setProbe(null);
            setProbeSubmitted(false);
            if (response.done) {
                finishTrial();
                return;
            }
            setStatus("running");
        }).catch((resumeError) => {
            setError(String(resumeError.message || resumeError));
            setStatus("error");
        });
    }

    function submitTrialRating(values) {
        const mergedSummary = Object.assign({}, finishPayload.trial_summary, { post_trial_rating: values });
        apiPost("/save_trial_data", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
            updates: {
                post_trial_rating: values,
                trial_summary: mergedSummary,
            },
        }).then(() => {
            props.onComplete(mergedSummary, finishPayload.trajectory);
        }).catch((ratingError) => {
            setError(String(ratingError.message || ratingError));
            setStatus("error");
        });
    }

    return (
        <div className="trial-screen">
            <section className="panel-card panel-card--board">
                <div className="trial-hero">
                    <div className="panel-eyebrow">{props.trial.title}</div>
                    <h3>{props.trial.mode === "observe" ? "Focus on the highlighted AI chef" : "Cook with the AI chef"}</h3>
                    <p>{props.trial.instruction}</p>
                    {props.trial.mode === "observe" ? <div className="callout">Watch the chef with the bold green focus box.</div> : null}
                </div>
                <TrialStats runtime={runtime || {}} />
                <OvercookedBoard
                    layoutGrid={props.trial.layout_grid}
                    state={runtime ? runtime.state : null}
                    trial={Object.assign({}, props.trial, { show_target_highlight: props.trial.mode === "observe" })}
                />
                {error ? <div className="callout callout--danger">{error}</div> : null}
            </section>

            <ModalShell open={status === "probe"} eyebrow="Probe" title={probeSubmitted ? "Expectation saved" : "What do you expect next?"}>
                {!probeSubmitted ? (
                    <ExpectationComposer
                        key={"probe-" + probeDraftKey}
                        title="Report what you expect from the AI chef"
                        probeIndex={probe ? probe.probe_index : 1}
                        probeTotal={probe ? probe.probe_total : props.trial.probe.count}
                        prompt={props.trial.probe.prompt}
                        sketchPrompt={props.trial.probe.sketch_prompt}
                        confidencePrompt={props.trial.probe.confidence_prompt}
                        subtaskOptions={props.subtaskOptions}
                        layoutGrid={props.trial.layout_grid}
                        state={runtime ? runtime.state : null}
                        targetAgentIndex={props.trial.target_agent_index}
                        onSubmit={submitProbe}
                    />
                ) : (
                    <div className="panel-stack">
                        <p>Now watch what the AI chef actually does.</p>
                        <div className="stage-actions">
                            <button className="primary-button" type="button" onClick={resumeAfterProbe}>Resume</button>
                        </div>
                    </div>
                )}
            </ModalShell>

            <PostTrialRatingModal
                open={status === "rating" && Boolean(finishPayload)}
                trial={props.trial}
                initialValues={{}}
                onSubmit={submitTrialRating}
            />
        </div>
    );
}

function ReplayTrialRunner(props) {
    const [trajectory, setTrajectory] = useState(null);
    const [summary, setSummary] = useState(null);
    const [cursor, setCursor] = useState(0);
    const [phase, setPhase] = useState("booting");
    const [activeProbe, setActiveProbe] = useState(null);
    const [probeResponses, setProbeResponses] = useState([]);
    const [probeSubmitted, setProbeSubmitted] = useState(false);
    const [probeDraftKey, setProbeDraftKey] = useState(0);
    const [finishPayload, setFinishPayload] = useState(null);
    const [error, setError] = useState("");
    const finishingRef = useRef(false);

    useEffect(() => {
        let mounted = true;
        apiPost("/start_trial", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
        }).then(() => {
            if (!mounted) {
                return;
            }
            if (props.cachedTrajectory) {
                setTrajectory(props.cachedTrajectory.trajectory);
                setSummary(props.cachedTrajectory.summary);
                setPhase("playing");
                return;
            }
            apiPost("/load_trial_trajectory", {
                user_info: props.userInfo,
                trial_id: props.trial.source_trial_id,
            }).then((response) => {
                if (!mounted) {
                    return;
                }
                setTrajectory(response.trajectory);
                setSummary(response.trial_record ? response.trial_record.trial_summary : null);
                setPhase("playing");
            });
        }).catch((loadError) => {
            if (!mounted) {
                return;
            }
            setError(String(loadError.message || loadError));
            setPhase("error");
        });
        return function cleanup() {
            mounted = false;
        };
    }, [props.trial.id, props.trial.source_trial_id, props.userInfo, props.cachedTrajectory]);

    function getReplayProbePlan() {
        const observations = trajectory && trajectory.ep_states ? (trajectory.ep_states[0] || []) : [];
        if (!observations.length) {
            return [];
        }
        if (summary && summary.probes && summary.probes.length) {
            return summary.probes.slice(0, props.trial.probe.count).map((probeRecord, index) => Object.assign({}, probeRecord, { probe_index: index + 1 }));
        }
        return [1, 2, 3].map((value) => ({
            probe_index: value,
            probe_game_loop: Math.max(1, Math.floor((observations.length * value) / 4)),
        }));
    }

    useEffect(() => {
        if (phase !== "playing" || !trajectory) {
            return undefined;
        }
        const replayPlan = getReplayProbePlan();
        const observations = trajectory.ep_states[0] || [];
        const timerId = window.setInterval(() => {
            setCursor((previous) => {
                const nextProbe = replayPlan[probeResponses.length];
                if (nextProbe && previous >= nextProbe.probe_game_loop) {
                    setActiveProbe(nextProbe);
                    setProbeSubmitted(false);
                    setProbeDraftKey((current) => current + 1);
                    setPhase("probe");
                    return previous;
                }
                const next = previous + 1;
                if (next >= observations.length) {
                    window.clearInterval(timerId);
                    setPhase("finishing");
                    return previous;
                }
                return next;
            });
        }, REPLAY_TIMESTEP_MS);
        return function cleanup() {
            window.clearInterval(timerId);
        };
    }, [phase, trajectory, probeResponses.length]);

    useEffect(() => {
        if (phase !== "finishing" || finishingRef.current || !summary) {
            return;
        }
        finishingRef.current = true;
        const summaryPayload = {
            source_trial_id: props.trial.source_trial_id,
            source_summary: summary,
            replay_expectations: probeResponses,
        };
        apiPost("/save_trial_data", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
            updates: summaryPayload,
        }).then(() => apiPost("/finish_episode", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
            traj_id: makeTrajId(props.trial.id),
            summary: summaryPayload,
        })).then((payload) => {
            setFinishPayload(payload);
            setPhase("rating");
        }).catch((finishError) => {
            setError(String(finishError.message || finishError));
            setPhase("error");
        });
    }, [phase, props.trial.id, props.trial.source_trial_id, props.userInfo, probeResponses, summary]);

    function submitReplayProbe(expectation) {
        const record = {
            probe_index: activeProbe ? activeProbe.probe_index : probeResponses.length + 1,
            source_probe_game_loop: activeProbe ? activeProbe.probe_game_loop : cursor,
            source_actual_subtask_id: activeProbe ? activeProbe.actual_subtask_id : null,
            source_actual_subtask_label: activeProbe ? activeProbe.actual_subtask_label : null,
            selected_subtask_id: expectation.selectedSubtaskId,
            expected_path: expectation.expectedPath,
            confidence: expectation.confidence,
            response_timestamp: Date.now(),
        };
        setProbeResponses((current) => current.concat([record]));
        setProbeSubmitted(true);
    }

    function resumeAfterReplayProbe() {
        setActiveProbe(null);
        setProbeSubmitted(false);
        setPhase("playing");
    }

    function submitTrialRating(values) {
        const mergedSummary = Object.assign({}, finishPayload.trial_summary, { post_trial_rating: values });
        apiPost("/save_trial_data", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
            updates: {
                post_trial_rating: values,
                trial_summary: mergedSummary,
            },
        }).then(() => {
            props.onComplete(mergedSummary, null);
        }).catch((ratingError) => {
            setError(String(ratingError.message || ratingError));
            setPhase("error");
        });
    }

    const activeState = trajectory && trajectory.ep_states && trajectory.ep_states[0] ? trajectory.ep_states[0][cursor] : null;

    return (
        <div className="trial-screen">
            <section className="panel-card panel-card--board">
                <div className="trial-hero">
                    <div className="panel-eyebrow">{props.trial.title}</div>
                    <h3>Replay the earlier scene</h3>
                    <p>{props.trial.instruction}</p>
                </div>
                <TrialStats runtime={{ score: summary ? summary.score : 0, time_left: trajectory && trajectory.ep_states ? Math.max((trajectory.ep_states[0] || []).length - cursor, 0) : 0, step_count: cursor, probe_records: probeResponses }} />
                <OvercookedBoard
                    layoutGrid={props.trial.layout_grid}
                    state={activeState}
                    trial={Object.assign({}, props.trial, { show_target_highlight: true })}
                />
                {phase === "playing" ? <div className="callout">Watch the replay. It will pause three times and ask what you expected.</div> : null}
                {error ? <div className="callout callout--danger">{error}</div> : null}
            </section>

            <ModalShell open={phase === "probe"} eyebrow="Replay probe" title={probeSubmitted ? "Expectation saved" : "What did you expect here?"}>
                {!probeSubmitted ? (
                    <ExpectationComposer
                        key={"replay-probe-" + probeDraftKey}
                        title="Report what you expected from the AI chef at this moment"
                        probeIndex={activeProbe ? activeProbe.probe_index : probeResponses.length + 1}
                        probeTotal={props.trial.probe.count}
                        prompt={props.trial.probe.prompt}
                        sketchPrompt={props.trial.probe.sketch_prompt}
                        confidencePrompt={props.trial.probe.confidence_prompt}
                        subtaskOptions={props.subtaskOptions}
                        layoutGrid={props.trial.layout_grid}
                        state={activeState}
                        targetAgentIndex={props.trial.target_agent_index}
                        onSubmit={submitReplayProbe}
                    />
                ) : (
                    <div className="panel-stack">
                        <p>Your expectation was saved. Resume replay to compare it with the actual continuation.</p>
                        <div className="stage-actions">
                            <button className="primary-button" type="button" onClick={resumeAfterReplayProbe}>Resume replay</button>
                        </div>
                    </div>
                )}
            </ModalShell>

            <PostTrialRatingModal open={phase === "rating" && Boolean(finishPayload)} trial={props.trial} initialValues={{}} onSubmit={submitTrialRating} />
        </div>
    );
}

function TutorialStage(props) {
    const trialIndex = props.runnerState.trialIndexByStage[props.stage.id] || 0;
    const trialId = props.stage.trial_ids[trialIndex];
    const trial = trialId ? props.session.trials[trialId] : null;
    const [selectedRecipeId, setSelectedRecipeId] = useState((props.initialValues && props.initialValues.team_goal_recipe_id) || props.session.recipe_options[0].id);
    const [savedSetup, setSavedSetup] = useState(Boolean(props.initialValues && props.initialValues.team_goal_recipe_id));
    const [started, setStarted] = useState(false);
    const [briefIndex, setBriefIndex] = useState(-1);
    const [error, setError] = useState("");

    useEffect(() => {
        setStarted(false);
        setBriefIndex(-1);
    }, [trialId]);

    const briefingSteps = useMemo(() => {
        if (!trial) {
            return [];
        }
        if (trial.id === "tutorial_solo") {
            return [
                {
                    title: "You are the gray chef",
                    body: "First, practice the kitchen alone without help from the AI chef.",
                    visual: (
                        <div className="role-card__visual">
                            <ChefVisual hatColor="gray" />
                            <span className="ingredient-strip__plus">→</span>
                            <IngredientStrip ingredients={["tomato", "tomato", "tomato"]} />
                        </div>
                    ),
                },
                {
                    title: "Your team goal",
                    body: "The recipe you choose here will be used for the tutorial kitchen.",
                    visual: <IngredientStrip ingredients={(props.session.recipe_options.find((recipe) => recipe.id === selectedRecipeId) || props.session.recipe_options[0]).ingredients} />,
                },
                {
                    title: "Solo practice flow",
                    body: "Bring 3 ingredients to the pot, wait 20 seconds for cooking, pick up the soup with a dish, and deliver it.",
                },
            ];
        }
        return [
            {
                title: "Now cook with the AI chef",
                body: "The orange chef is the AI teammate you will collaborate with and predict during the study.",
                visual: (
                    <div className="role-card__visual">
                        <ChefVisual hatColor="gray" />
                        <span className="ingredient-strip__plus">+</span>
                        <ChefVisual hatColor="orange" />
                    </div>
                ),
            },
            {
                title: "Shared goal",
                body: "Keep working toward the selected recipe together while the AI chef helps in the same kitchen.",
                visual: <IngredientStrip ingredients={(props.session.recipe_options.find((recipe) => recipe.id === selectedRecipeId) || props.session.recipe_options[0]).ingredients} />,
            },
            {
                title: "Probe workflow",
                body: "A practice probe will appear. Choose the AI chef's next subtask, draw the expected route, rate your confidence, then watch the actual behavior.",
            },
        ];
    }, [trial, props.session.recipe_options, selectedRecipeId]);

    function persistSetup() {
        return apiPost("/save_session_section", {
            user_info: props.userInfo,
            section_id: "tutorial_setup",
            data: {
                team_goal_recipe_id: selectedRecipeId,
            },
        }).then(() => {
            setSavedSetup(true);
        });
    }

    function begin() {
        persistSetup().then(() => {
            setBriefIndex(0);
        }).catch((saveError) => {
            setError(String(saveError.message || saveError));
        });
    }

    if (!trial) {
        return (
            <div className="stage-stack">
                <div className="callout">Tutorial complete. Continue to EEG setup.</div>
                <div className="stage-actions">
                    <button className="primary-button" type="button" onClick={() => props.onComplete({ tutorial_complete: true, team_goal_recipe_id: selectedRecipeId })}>Continue</button>
                </div>
            </div>
        );
    }

    if (started) {
        return (
            <InteractiveTrialRunner
                trial={trial}
                userInfo={props.userInfo}
                subtaskOptions={props.session.subtask_options}
                onComplete={(summary, trajectory) => {
                    setStarted(false);
                    props.onTrialComplete(trial, summary, trajectory);
                }}
            />
        );
    }

    return (
        <div className="stage-stack">
            <section className="panel-card">
                <div className="role-grid">
                    <div className="role-card role-card--human">
                        <div className="panel-eyebrow">You</div>
                        <div className="role-card__visual">
                            <ChefVisual hatColor="gray" />
                        </div>
                        <h3>Gray chef</h3>
                        <p>You control the gray chef.</p>
                    </div>
                    <div className="role-card role-card--ai">
                        <div className="panel-eyebrow">AI chef</div>
                        <div className="role-card__visual">
                            <ChefVisual hatColor="orange" />
                        </div>
                        <h3>Orange-highlighted chef</h3>
                        <p>The AI chef is the one you observe and predict during probes.</p>
                    </div>
                </div>
                <div className="section-block">
                    <div className="section-block__title">Set the team goal</div>
                    <div className="recipe-grid">
                        {props.session.recipe_options.map((recipe) => (
                            <button
                                className={"recipe-card" + (selectedRecipeId === recipe.id ? " is-selected" : "")}
                                key={recipe.id}
                                type="button"
                                onClick={() => setSelectedRecipeId(recipe.id)}
                            >
                                <div className="recipe-card__visual">
                                    <IngredientStrip ingredients={recipe.ingredients} />
                                </div>
                                <div className="recipe-card__short">{recipe.short_label}</div>
                                <strong>{recipe.label}</strong>
                                <span>{recipe.points} points</span>
                            </button>
                        ))}
                    </div>
                </div>
                <div className="section-block">
                    <div className="section-block__title">How tutorial works</div>
                    <div className="tutorial-steps">
                        <div>1. Practice cooking alone first.</div>
                        <div>2. Practice once more with the AI chef.</div>
                        <div>3. Try the probe workflow before the main session.</div>
                    </div>
                </div>
                {error ? <div className="callout callout--danger">{error}</div> : null}
                <div className="stage-actions">
                    <button className="primary-button" type="button" onClick={begin}>{savedSetup ? "Start next tutorial trial" : "Save goal and start tutorial"}</button>
                </div>
            </section>
            <section className="panel-card">
                <OvercookedBoard layoutGrid={props.session.layout_grid} state={props.previewState} trial={{ human_player_index: 1, target_agent_index: 0, show_target_highlight: false }} label="Tutorial" description="Practice the kitchen before the main session." />
            </section>
            <ModalShell open={briefIndex >= 0 && briefIndex < briefingSteps.length} eyebrow="Tutorial" title={briefingSteps[briefIndex] ? briefingSteps[briefIndex].title : "Tutorial"}>
                <div className="panel-stack">
                    {briefingSteps[briefIndex] && briefingSteps[briefIndex].visual ? <div className="guide-visual">{briefingSteps[briefIndex].visual}</div> : null}
                    <p>{briefingSteps[briefIndex] ? briefingSteps[briefIndex].body : ""}</p>
                    <div className="stage-actions">
                        {briefIndex < briefingSteps.length - 1 ? (
                            <button className="primary-button" type="button" onClick={() => setBriefIndex(briefIndex + 1)}>Next</button>
                        ) : (
                            <button className="primary-button" type="button" onClick={() => {
                                setBriefIndex(briefingSteps.length);
                                setStarted(true);
                            }}>Start tutorial</button>
                        )}
                    </div>
                </div>
            </ModalShell>
        </div>
    );
}

function StaticStage(props) {
    return (
        <div className="stage-stack">
            <section className="panel-card">
                <div className="panel-stack">
                    {props.stage.body.map((line, index) => <p key={"static-" + index}>{line}</p>)}
                    <div className="stage-actions">
                        <button className="primary-button" type="button" onClick={() => props.onContinue({ acknowledged: true })}>Continue</button>
                    </div>
                </div>
            </section>
        </div>
    );
}

function SurveyStage(props) {
    const initialValues = Object.assign({}, props.initialValues || {});
    props.stage.questions.forEach((question) => {
        if (initialValues[question.id] === undefined) {
            initialValues[question.id] = "";
        }
    });
    const [values, setValues] = useState(initialValues);
    const [error, setError] = useState("");

    function updateField(field, event) {
        const nextValues = Object.assign({}, values);
        nextValues[field.id] = fieldValueFromEvent(field, event);
        setValues(nextValues);
    }

    function submit() {
        const missing = props.stage.questions.filter((question) => question.type !== "textarea" && !String(values[question.id] || "").trim());
        if (missing.length) {
            setError("Please answer all required questions.");
            return;
        }
        props.onSubmit(values);
    }

    return (
        <div className="stage-stack">
            <section className="panel-card panel-card--scroll">
                <div className="panel-stack">
                    {props.stage.questions.map((question) => (
                        <InputField key={question.id} field={question} value={values[question.id]} onChange={(event) => updateField(question, event)} />
                    ))}
                    {error ? <div className="callout callout--danger">{error}</div> : null}
                    <div className="stage-actions">
                        <button className="primary-button" type="button" onClick={submit}>Save and continue</button>
                    </div>
                </div>
            </section>
        </div>
    );
}

function TrialBlockStage(props) {
    const trialIndex = props.runnerState.trialIndexByStage[props.stage.id] || 0;
    const trialId = props.stage.trial_ids[trialIndex];
    const trial = trialId ? props.session.trials[trialId] : null;
    const [started, setStarted] = useState(false);

    useEffect(() => {
        setStarted(false);
    }, [props.stage.id, trialId]);

    if (!trial) {
        return (
            <div className="stage-stack">
                <div className="callout">This block is complete.</div>
                <div className="stage-actions">
                    <button className="primary-button" type="button" onClick={() => props.onAdvanceEmptyBlock()}>Continue</button>
                </div>
            </div>
        );
    }

    if (started) {
        if (trial.mode === "replay") {
            return (
                <ReplayTrialRunner
                    trial={trial}
                    userInfo={props.userInfo}
                    subtaskOptions={props.session.subtask_options}
                    cachedTrajectory={props.trajectoryCache[trial.source_trial_id] || null}
                    onComplete={(summary, trajectory) => {
                        setStarted(false);
                        props.onTrialComplete(trial, summary, trajectory);
                    }}
                />
            );
        }
        return (
            <InteractiveTrialRunner
                trial={trial}
                userInfo={props.userInfo}
                subtaskOptions={props.session.subtask_options}
                onComplete={(summary, trajectory) => {
                    setStarted(false);
                    props.onTrialComplete(trial, summary, trajectory);
                }}
            />
        );
    }

    return (
        <div className="stage-stack">
            <section className="panel-card">
                <div className="trial-start">
                    <div className="panel-eyebrow">{trial.mode.toUpperCase()}</div>
                    <h3>{trial.title}</h3>
                    <p>{trial.instruction}</p>
                    <div className="trial-start__meta">
                        <span>Trial {trialIndex + 1} / {props.stage.trial_ids.length}</span>
                        {trial.mode === "observe" ? <span>Watch the chef with the bold green focus box.</span> : null}
                    </div>
                    <div className="stage-actions">
                        <button className="primary-button" type="button" onClick={() => setStarted(true)}>Start trial</button>
                    </div>
                </div>
            </section>
            <section className="panel-card">
                <OvercookedBoard layoutGrid={trial.layout_grid} state={props.previewState} trial={Object.assign({}, trial, { show_target_highlight: trial.mode !== "collaborate" })} label={trial.title} description={trial.instruction} />
            </section>
        </div>
    );
}

function CompletionView() {
    function exit() {
        try {
            window.close();
        } catch (error) {
            window.location.href = "about:blank";
        }
    }
    return (
        <div className="completion-shell">
            <div className="panel-card completion-card">
                <h2>Session complete</h2>
                <p>Thank you for your participation.</p>
                <p>You have finished the session.</p>
                <p>Please remain seated and wait for the researcher.</p>
                <div className="stage-actions">
                    <button className="primary-button" type="button" onClick={exit}>Exit</button>
                </div>
            </div>
        </div>
    );
}

function SessionApp() {
    const storedUserInfo = readJsonStorage("before_game") || {};
    const [bootState, setBootState] = useState({ loading: true, error: "", bundle: null });
    const [userInfo, setUserInfo] = useState(storedUserInfo);
    const [runnerState, setRunnerState] = useState(null);
    const trajectoryCacheRef = useRef({});
    const previewStateRef = useRef(buildPreviewState([
        "XXXXXXXXXXXXX",
        "O   DTXTD   O",
        "XX    P    XX",
        "S   2 P 1   S",
        "XXXXXTXTXXXXX",
    ]));

    function bootstrap(nextUserInfo) {
        setBootState({ loading: true, error: "", bundle: null });
        return apiPost("/session_config", nextUserInfo && nextUserInfo.name ? { user_info: nextUserInfo } : {})
            .then((bundle) => {
                setBootState({ loading: false, error: "", bundle: bundle });
                setRunnerState(buildRunnerStateFromServer(bundle.session, bundle, nextUserInfo || {}));
            })
            .catch((error) => {
                setBootState({ loading: false, error: String(error.message || error), bundle: null });
            });
    }

    useEffect(() => {
        bootstrap(storedUserInfo);
    }, []);

    useEffect(() => {
        if (runnerState) {
            writeJsonStorage("eeg_runner_state_spa", runnerState);
        }
    }, [runnerState]);

    function persistProgress(nextRunnerState, currentStageId, payload, actorOverride) {
        const actor = actorOverride || userInfo;
        if (!actor || !actor.name) {
            return Promise.resolve();
        }
        return apiPost("/save_session_section", {
            user_info: actor,
            section_id: "client_progress",
            data: {
                runner_state: nextRunnerState,
                payload: payload || {},
            },
            progress: {
                current_stage_id: currentStageId,
                completed_stage_ids: nextRunnerState.completedStages,
            },
        });
    }

    function completeStage(stage, data, userOverride, baseRunnerState) {
        const runner = baseRunnerState || runnerState;
        const actor = userOverride || userInfo;
        const completedStages = dedupe((runner.completedStages || []).concat([stage.id]));
        const nextStageIndex = Math.min(runner.stageIndex + 1, bootState.bundle.session.stages.length);
        const nextStage = bootState.bundle.session.stages[nextStageIndex] || null;
        const nextRunnerState = {
            stageIndex: nextStageIndex,
            trialIndexByStage: Object.assign({}, runner.trialIndexByStage),
            completedStages: completedStages,
            completedTrials: Object.assign({}, runner.completedTrials),
            trialSummaries: Object.assign({}, runner.trialSummaries),
            sectionResponses: Object.assign({}, runner.sectionResponses),
        };
        nextRunnerState.sectionResponses[stage.id] = data || {};

        const savePromise = actor && actor.name ? apiPost("/save_session_section", {
            user_info: actor,
            section_id: stage.id,
            data: data || {},
            progress: {
                current_stage_id: nextStage ? nextStage.id : "session_complete",
                completed_stage_ids: completedStages,
            },
        }) : Promise.resolve();

        return savePromise.then(() => {
            setRunnerState(nextRunnerState);
            if (actor && actor.name) {
                return persistProgress(nextRunnerState, nextStage ? nextStage.id : "session_complete", data || {}, actor);
            }
            return null;
        });
    }

    function handleIntakeSubmit(values) {
        setUserInfo(values);
        return completeStage(bootState.bundle.session.stages[runnerState.stageIndex], values, values);
    }

    function handleSurveySubmit(stage, values) {
        completeStage(stage, values);
    }

    function handleTrialComplete(trial, summary, trajectory) {
        const stage = bootState.bundle.session.stages[runnerState.stageIndex];
        const currentCount = runnerState.trialIndexByStage[stage.id] || 0;
        const nextCount = currentCount + 1;
        const nextTrialIndexByStage = Object.assign({}, runnerState.trialIndexByStage);
        nextTrialIndexByStage[stage.id] = nextCount;
        const nextCompletedTrials = Object.assign({}, runnerState.completedTrials);
        nextCompletedTrials[trial.id] = true;
        const nextTrialSummaries = Object.assign({}, runnerState.trialSummaries);
        nextTrialSummaries[trial.id] = summary;
        if (trajectory) {
            trajectoryCacheRef.current[trial.id] = {
                trajectory: trajectory,
                summary: summary,
            };
        }

        const updatedRunnerState = {
            stageIndex: runnerState.stageIndex,
            trialIndexByStage: nextTrialIndexByStage,
            completedStages: runnerState.completedStages.slice(),
            completedTrials: nextCompletedTrials,
            trialSummaries: nextTrialSummaries,
            sectionResponses: Object.assign({}, runnerState.sectionResponses),
        };

        const blockFinished = nextCount >= stage.trial_ids.length;
        if (blockFinished) {
            setRunnerState(updatedRunnerState);
            completeStage(stage, { block_complete: true, last_trial_id: trial.id }, null, updatedRunnerState);
            return;
        }

        setRunnerState(updatedRunnerState);
        persistProgress(updatedRunnerState, stage.id, { current_trial_id: trial.id });
    }

    if (bootState.loading || !runnerState) {
        return (
            <div className="loading-shell">
                <div className="loading-card">
                    <div className="loader" />
                    <h2>Loading session...</h2>
                </div>
            </div>
        );
    }

    if (bootState.error) {
        return (
            <div className="loading-shell">
                <div className="loading-card loading-card--error">
                    <h2>Failed to load session</h2>
                    <p>{bootState.error}</p>
                </div>
            </div>
        );
    }

    const session = bootState.bundle.session;
    const stage = session.stages[runnerState.stageIndex] || null;
    if (!stage) {
        return <CompletionView />;
    }

    let stageBody = null;
    if (stage.type === "welcome") {
        stageBody = <WelcomeStage session={session} onContinue={(payload) => completeStage(stage, payload)} />;
    } else if (stage.type === "mode_overview") {
        stageBody = <ModeOverviewStage session={session} onContinue={(payload) => completeStage(stage, payload)} />;
    } else if (stage.type === "intake_form") {
        stageBody = <ConsentStage intakeForm={bootState.bundle.intake_form} userInfo={userInfo} onSubmit={handleIntakeSubmit} />;
    } else if (stage.type === "onboarding") {
        stageBody = <OnboardingStage stage={stage} session={session} onContinue={(payload) => completeStage(stage, payload)} />;
    } else if (stage.type === "tutorial_lab") {
        stageBody = (
            <TutorialStage
                stage={stage}
                session={session}
                userInfo={userInfo}
                previewState={previewStateRef.current}
                runnerState={runnerState}
                initialValues={runnerState.sectionResponses.tutorial_setup || runnerState.sectionResponses[stage.id] || {}}
                onTrialComplete={handleTrialComplete}
                onComplete={(payload) => completeStage(stage, payload)}
            />
        );
    } else if (stage.type === "static") {
        stageBody = <StaticStage stage={stage} onContinue={(payload) => completeStage(stage, payload)} />;
    } else if (stage.type === "survey") {
        stageBody = <SurveyStage stage={stage} initialValues={runnerState.sectionResponses[stage.id]} onSubmit={(values) => handleSurveySubmit(stage, values)} />;
    } else if (stage.type === "trial_block") {
        stageBody = (
            <TrialBlockStage
                stage={stage}
                session={session}
                userInfo={userInfo}
                runnerState={runnerState}
                previewState={previewStateRef.current}
                trajectoryCache={trajectoryCacheRef.current}
                onTrialComplete={handleTrialComplete}
                onAdvanceEmptyBlock={() => completeStage(stage, { block_complete: true })}
            />
        );
    }

    return (
        <div className="app-shell">
            <TopProgress session={session} runnerState={runnerState} />
            <main className="app-main">
                <StageHeader stage={stage} session={session} runnerState={runnerState} />
                {stageBody}
            </main>
        </div>
    );
}

ReactDOM.render(<SessionApp />, document.getElementById("app-root"));
