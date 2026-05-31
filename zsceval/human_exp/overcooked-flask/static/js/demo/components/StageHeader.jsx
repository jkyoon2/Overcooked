import React from "react";
import { splitStageHeading } from "../utils.js";

export default function StageHeader(props) {
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
