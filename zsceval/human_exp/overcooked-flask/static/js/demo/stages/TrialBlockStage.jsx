import React, { useState, useEffect } from "react";
import OvercookedBoard from "../spa/board.jsx";
import InteractiveTrialRunner from "../components/InteractiveTrialRunner.jsx";
import ReplayTrialRunner from "../components/ReplayTrialRunner.jsx";

export default function TrialBlockStage(props) {
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
                <OvercookedBoard
                    layoutGrid={trial.layout_grid}
                    state={props.previewState}
                    trial={Object.assign({}, trial, { show_target_highlight: trial.mode !== "collaborate" })}
                    label={trial.title}
                    description={trial.instruction}
                />
            </section>
        </div>
    );
}
