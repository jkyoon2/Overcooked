import React from "react";

export default function ModeAssetCard(props) {
    return (
        <div className="mode-card mode-card--asset-only">
            <div className="mode-card__copy">
                <h3>{props.title}</h3>
                <p>{props.description}</p>
            </div>
            <div className="mode-card__asset-stage">
                {props.asset}
            </div>
        </div>
    );
}
