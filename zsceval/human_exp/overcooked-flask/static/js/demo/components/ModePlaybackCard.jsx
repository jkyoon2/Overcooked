import React from "react";
import LoopingBoardPreview from "./LoopingBoardPreview.jsx";

export default function ModePlaybackCard(props) {
    return (
        <div className="mode-card mode-card--playback">
            <div className="mode-card__copy">
                <h3>{props.title}</h3>
                <p>{props.description}</p>
            </div>
            <div className="mode-card__asset-stage mode-card__asset-stage--playback">
                <LoopingBoardPreview
                    compact
                    frames={props.frames}
                    layoutGrid={props.layoutGrid}
                    trial={props.trial}
                    caption={props.caption}
                />
            </div>
        </div>
    );
}
