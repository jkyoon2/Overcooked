import React, { useState, useEffect } from "react";
import { fieldValueFromEvent } from "../utils.js";
import ModalShell from "./ModalShell.jsx";
import InputField from "./InputField.jsx";

export default function PostTrialRatingModal(props) {
    const [values, setValues] = useState(props.initialValues || {});

    useEffect(() => {
        setValues(props.initialValues || {});
    }, [props.initialValues, props.trial.id]);

    function update(field, event) {
        const nextValues = Object.assign({}, values);
        nextValues[field.id] = fieldValueFromEvent(field, event);
        setValues(nextValues);
    }

    const complete = props.trial.post_trial_questions.every((question) => String(values[question.id] || "").trim());

    return (
        <ModalShell open={props.open} eyebrow="Trial rating" title="How did this trial feel?">
            <div className="panel-stack">
                {props.trial.post_trial_questions.map((question) => (
                    <InputField
                        key={question.id}
                        field={question}
                        value={values[question.id] || ""}
                        onChange={(event) => update(question, event)}
                    />
                ))}
                <div className="stage-actions">
                    <button className="primary-button" type="button" disabled={!complete} onClick={() => props.onSubmit(values)}>
                        Save trial rating
                    </button>
                </div>
            </div>
        </ModalShell>
    );
}
