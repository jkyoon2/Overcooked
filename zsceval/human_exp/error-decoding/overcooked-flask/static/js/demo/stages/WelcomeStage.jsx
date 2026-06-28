import React, { useMemo } from "react";
import { buildModeDemoFrames } from "../utils.js";
import ObserveHeroPreview from "../components/ObserveHeroPreview.jsx";

export default function WelcomeStage(props) {
    const demos = useMemo(() => buildModeDemoFrames(), []);
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
