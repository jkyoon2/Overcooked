import React from "react";
import { OBJECTS_ATLAS, OBJECTS_IMAGE_URL } from "../spa/atlas.js";
import ChefVisual from "./ChefVisual.jsx";
import AtlasSprite from "./AtlasSprite.jsx";

export default function ObserveSceneAsset() {
    return (
        <div className="mode-scene mode-scene--observe">
            <div className="mode-scene__spotlight">
                <ChefVisual hatColor="orange" size={60} />
            </div>
            <div className="mode-scene__connector" />
            <div className="mode-scene__stack">
                <div className="mode-scene__pair">
                    <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="tomato.png" size={42} className="mini-sprite" />
                    <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="dish.png" size={42} className="mini-sprite" />
                    <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="soup-tomato-dish.png" size={42} className="mini-sprite" />
                </div>
                <div className="mode-scene__caption">Watch the AI chef.</div>
            </div>
        </div>
    );
}
