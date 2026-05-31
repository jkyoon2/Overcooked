import React, { useState, useEffect } from "react";
import OvercookedBoard from "../spa/board.jsx";
import ExpectationComposer from "../spa/expectation.jsx";
import { PREVIEW_LOOP_MS } from "../constants.js";
import { buildPreviewExpectation } from "../utils.js";

export default function ObserveHeroPreview(props) {
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
    const previewTrial = Object.assign({}, props.trial, { show_target_highlight: false });

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
