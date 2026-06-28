import React, { useState, useEffect } from "react";
import OvercookedBoard from "../spa/board.jsx";
import { PREVIEW_LOOP_MS } from "../constants.js";
import LiveProbePreview from "./LiveProbePreview.jsx";

export default function AnimatedSceneCard(props) {
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
