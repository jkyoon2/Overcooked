import React, { useState, useEffect } from "react";
import OvercookedBoard from "../spa/board.jsx";
import { PREVIEW_LOOP_MS } from "../constants.js";

export default function LoopingBoardPreview(props) {
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
                    tileSize={props.tileSize || (props.compact ? 52 : undefined)}
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
