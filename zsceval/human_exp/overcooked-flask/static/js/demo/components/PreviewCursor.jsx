import React from "react";

export default function PreviewCursor(props) {
    return (
        <div
            className={"preview-cursor" + (props.isClicking ? " is-clicking" : "")}
            style={{ left: props.left, top: props.top }}
        />
    );
}
