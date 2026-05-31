import React from "react";
import { normalizeChoiceValue, normalizeChoiceLabel } from "../utils.js";
import LikertBar from "./LikertBar.jsx";

export default function InputField(props) {
    const field = props.field;
    const value = normalizeChoiceValue(props.value);

    if (field.type === "radio") {
        const isInlineRow = field.options.length <= 4;
        return (
            <div className="form-field">
                <label className="form-field__label">{field.label}{field.required ? " *" : ""}</label>
                <div className={"choice-grid" + (isInlineRow ? " choice-grid--inline" : "")}>
                    {field.options.map((option) => {
                        const optionValue = normalizeChoiceValue(option.value);
                        const optionLabel = normalizeChoiceLabel(option.label, option.value);
                        return (
                            <button
                                className={"choice-pill choice-pill--segmented" + (String(value) === String(optionValue) ? " is-selected" : "")}
                                key={field.id + "-" + optionValue}
                                type="button"
                                onClick={() => props.onChange({ target: { value: optionValue } })}
                            >
                                <span className="choice-pill__label">{optionLabel}</span>
                            </button>
                        );
                    })}
                </div>
            </div>
        );
    }

    if (field.type === "scale" || field.type === "scale_bar") {
        return (
            <LikertBar
                fieldId={field.id}
                label={field.label}
                min={field.min}
                max={field.max}
                value={value}
                leftLabel={field.left_label}
                rightLabel={field.right_label}
                onChange={(nextValue) => props.onChange({ target: { value: String(nextValue) } })}
            />
        );
    }

    if (field.type === "textarea") {
        return (
            <div className="form-field">
                <label className="form-field__label">{field.label}{field.required ? " *" : ""}</label>
                <textarea className="form-textarea" value={value || ""} onChange={props.onChange} />
            </div>
        );
    }

    return (
        <div className="form-field">
            <label className="form-field__label">{field.label}{field.required ? " *" : ""}</label>
            <input className="form-input" type={field.type} min={field.min} max={field.max} value={value || ""} onChange={props.onChange} />
        </div>
    );
}
