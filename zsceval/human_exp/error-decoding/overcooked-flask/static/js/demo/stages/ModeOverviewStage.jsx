import React, { useMemo } from "react";
import { buildModeDemoFrames } from "../utils.js";
import ModePlaybackCard from "../components/ModePlaybackCard.jsx";

export default function ModeOverviewStage(props) {
    const demos = useMemo(() => buildModeDemoFrames(), []);
    const collaborateTrial = props.session.trials.collaborate_1 || props.session.trials.tutorial_team || props.session.trials.observe_1;
    const replayTrial = props.session.trials.replay_1 || collaborateTrial;
    return (
        <div className="stage-stack">
            <div className="panel-card">
                <p>This experiment has two parts.</p>
                <p>In the first part, you'll cook alongside an AI chef — picking up ingredients, loading the pot, and delivering soup together. At certain moments, a prompt will appear asking you to predict what the AI chef will do next.</p>
                <p>In the second part, we'll replay a scene from your first session. When a prompt appears, submit your expectation for what the AI chef was about to do at that moment.</p>
            </div>
            <div className="mode-grid">
                <ModePlaybackCard
                    title="Collaborate"
                    description="Cook with the AI chef. Submit your prediction when prompted."
                    layoutGrid={props.session.layout_grid}
                    frames={demos.collaborate}
                    trial={collaborateTrial}
                    caption="Live play"
                />
                <ModePlaybackCard
                    title="Replay + Annotation"
                    description="Watch the replay. Submit what you expected the AI chef to do."
                    layoutGrid={props.session.layout_grid}
                    frames={demos.replay}
                    trial={replayTrial}
                    caption="Replay"
                />
            </div>
            <div className="stage-actions">
                <button className="primary-button" type="button" onClick={() => props.onContinue({ reviewed_modes: true })}>Next</button>
            </div>
        </div>
    );
}
