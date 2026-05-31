import React from "react";
import ExpectationComposer from "../spa/expectation.jsx";
import { buildPreviewExpectation } from "../utils.js";
import PreviewCursor from "./PreviewCursor.jsx";

export default function LiveProbePreview(props) {
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
