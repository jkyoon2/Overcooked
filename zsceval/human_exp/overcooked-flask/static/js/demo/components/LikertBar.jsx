import React from "react";

export default function LikertBar(props) {
    const ticks = [];
    for (let value = props.min; value <= props.max; value += 1) {
        ticks.push(value);
    }
    return (
        <div className="likert-bar">
            <div className="likert-bar__label">{props.label}</div>
            <div className="likert-bar__track">
                {ticks.map((value) => (
                    <button
                        className={"likert-bar__tick" + (String(props.value) === String(value) ? " is-selected" : "")}
                        key={props.fieldId + "-" + value}
                        type="button"
                        onClick={() => props.onChange(value)}
                    >
                        <span>{value}</span>
                    </button>
                ))}
            </div>
            <div className="likert-bar__legend">
                <span>{props.leftLabel || ""}</span>
                <span>{props.rightLabel || ""}</span>
            </div>
        </div>
    );
}
