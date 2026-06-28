import React from "react";
import ChefVisual from "./ChefVisual.jsx";

export default function ReplaySceneAsset() {
    return (
        <div className="mode-scene mode-scene--replay">
            <div className="mode-scene__pair">
                <ChefVisual hatColor="gray" size={56} />
                <div className="mode-scene__rewind">↺</div>
                <ChefVisual hatColor="orange" size={56} />
            </div>
            <div className="mode-scene__path">
                <span /><span /><span /><span />
            </div>
            <div className="mode-scene__caption">Replay the collaborated scene and annotate what you expected.</div>
        </div>
    );
}
