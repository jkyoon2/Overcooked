import React, { useState } from "react";
import { apiPost, writeJsonStorage, fieldValueFromEvent, hasValue, normalizeChoiceValue, normalizeChoiceLabel } from "../utils.js";

export default function ConsentStage(props) {
    const initialValues = {};
    props.intakeForm.sections.forEach((section) => {
        section.fields.forEach((field) => {
            initialValues[field.id] = props.userInfo && Object.prototype.hasOwnProperty.call(props.userInfo, field.id)
                ? normalizeChoiceValue(props.userInfo[field.id])
                : "";
        });
    });
    initialValues.consent = false;

    const [values, setValues] = useState(initialValues);
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState("");

    function updateField(field, event) {
        const nextValues = Object.assign({}, values);
        nextValues[field.id] = fieldValueFromEvent(field, event);
        setValues(nextValues);
        setError("");
    }

    function submit() {
        const missing = [];
        props.intakeForm.sections.forEach((section) => {
            section.fields.forEach((field) => {
                if (field.required && !hasValue(values[field.id])) {
                    missing.push(field.label);
                }
            });
        });
        if (!values.consent) {
            missing.push("Consent");
        }
        if (missing.length) {
            setError("Please complete all required items.");
            return;
        }
        setError("");
        setSubmitting(true);
        apiPost("/create_questionnaire_before_game", values).then(() => {
            writeJsonStorage("before_game", values);
            return Promise.resolve(props.onSubmit(values));
        }).catch((submitError) => {
            setSubmitting(false);
            setError(String(submitError.message || submitError));
        });
    }

    function renderField(field) {
        const currentValue = normalizeChoiceValue(values[field.id]);
        if (field.type === "radio") {
            return (
                <div className="ci-field" key={field.id}>
                    <div className="ci-label">{field.label}{field.required ? <span className="ci-required"> *</span> : null}</div>
                    <div className="ci-pills">
                        {field.options.map((option) => {
                            const optionValue = normalizeChoiceValue(option.value);
                            const optionLabel = normalizeChoiceLabel(option.label, option.value);
                            return (
                                <button
                                    className={"ci-pill" + (String(currentValue) === String(optionValue) ? " is-selected" : "")}
                                    key={field.id + "-" + optionValue}
                                    type="button"
                                    onClick={() => { updateField(field, { target: { value: optionValue } }); }}
                                >
                                    {optionLabel}
                                </button>
                            );
                        })}
                    </div>
                </div>
            );
        }
        if (field.type === "textarea") {
            return (
                <div className="ci-field" key={field.id}>
                    <div className="ci-label">{field.label}{field.required ? <span className="ci-required"> *</span> : null}</div>
                    <textarea className="ci-textarea" value={currentValue || ""} onChange={(event) => updateField(field, event)} />
                </div>
            );
        }
        return (
            <div className="ci-field" key={field.id}>
                <div className="ci-label">{field.label}{field.required ? <span className="ci-required"> *</span> : null}</div>
                <input className="ci-input" type={field.type} min={field.min} max={field.max} value={currentValue || ""} onChange={(event) => updateField(field, event)} />
            </div>
        );
    }

    return (
        <div className="consent-form">
            {props.intakeForm.sections.map((section, sectionIndex) => (
                <div className="ci-section" key={section.id}>
                    {sectionIndex > 0 ? <div className="ci-divider" /> : null}
                    <div className="ci-section__label">{section.title}</div>
                    <div className="ci-fields">
                        {section.fields.map((field) => renderField(field))}
                    </div>
                </div>
            ))}
            <div className="ci-divider" />
            <label className="ci-consent">
                <input
                    type="checkbox"
                    checked={Boolean(values.consent)}
                    onChange={() => {
                        setValues(Object.assign({}, values, { consent: !values.consent }));
                        setError("");
                    }}
                />
                <span className={"ci-check-box" + (values.consent ? " is-checked" : "")}>
                    {values.consent ? (
                        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                            <path d="M2 6l3 3 5-5" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                    ) : null}
                </span>
                <span className="ci-consent__text">I agree to participate in this study.</span>
            </label>
            {error ? <div className="callout callout--danger">{error}</div> : null}
            <div className="stage-actions">
                <button className="primary-button" type="button" disabled={submitting} onClick={submit}>
                    {submitting ? "Saving..." : "Next"}
                </button>
            </div>
        </div>
    );
}
