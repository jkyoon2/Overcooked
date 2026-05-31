import React from "react";

export default function ModalShell(props) {
    if (!props.open) {
        return null;
    }
    return (
        <div className="modal-shell">
            <div className="modal-card">
                <div className="modal-card__head">
                    <div>
                        {props.eyebrow ? <div className="panel-eyebrow">{props.eyebrow}</div> : null}
                        <h3>{props.title}</h3>
                    </div>
                </div>
                <div className="modal-card__body">{props.children}</div>
            </div>
        </div>
    );
}
