import React, { useState } from "react";
import { fieldValueFromEvent } from "../utils.js";
import InputField from "../components/InputField.jsx";

export default function SurveyStage(props) {
    const initialValues = Object.assign({}, props.initialValues || {});
    props.stage.questions.forEach((question) => {
        if (initialValues[question.id] === undefined) {
            initialValues[question.id] = "";
        }
    });
    const [values, setValues] = useState(initialValues);
    const [error, setError] = useState("");

    function updateField(field, event) {
        const nextValues = Object.assign({}, values);
        nextValues[field.id] = fieldValueFromEvent(field, event);
        setValues(nextValues);
    }

    function submit() {
        const missing = props.stage.questions.filter((question) => question.type !== "textarea" && !String(values[question.id] || "").trim());
        if (missing.length) {
            setError("Please answer all required questions.");
            return;
        }
        props.onSubmit(values);
    }

    return (
        <div className="stage-stack">
            <section className="panel-card panel-card--scroll">
                <div className="panel-stack">
                    {props.stage.questions.map((question) => (
                        <InputField key={question.id} field={question} value={values[question.id]} onChange={(event) => updateField(question, event)} />
                    ))}
                    {error ? <div className="callout callout--danger">{error}</div> : null}
                    <div className="stage-actions">
                        <button className="primary-button" type="button" onClick={submit}>Save and continue</button>
                    </div>
                </div>
            </section>
        </div>
    );
}
