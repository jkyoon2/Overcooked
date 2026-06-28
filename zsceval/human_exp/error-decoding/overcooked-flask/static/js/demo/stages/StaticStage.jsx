import React from "react";

export default function StaticStage(props) {
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
