import React from "react";

export default function TopProgress(props) {
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
            <div
                className="stage-progress"
                style={{ gridTemplateColumns: "repeat(" + String(props.session.stages.length) + ", minmax(0, 1fr))" }}
            >
                {props.session.stages.map((stage, index) => {
                    const isComplete = props.runnerState.completedStages.indexOf(stage.id) >= 0;
                    const isCurrent = props.runnerState.stageIndex === index;
                    return (
                        <div
                            className={"stage-progress__segment" + (isCurrent ? " is-current" : "") + (isComplete ? " is-complete" : "")}
                            key={stage.id}
                        />
                    );
                })}
            </div>
        </section>
    );
}
