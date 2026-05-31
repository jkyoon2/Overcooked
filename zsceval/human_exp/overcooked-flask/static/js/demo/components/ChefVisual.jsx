import React from "react";
import { CHEFS_ATLAS, CHEFS_IMAGE_URL } from "../spa/atlas.js";
import AtlasSprite from "./AtlasSprite.jsx";

export default function ChefVisual(props) {
    const direction = "SOUTH";
    const hatFrame = direction + "-" + props.hatColor + "hat.png";
    const bodyFrame = direction + (props.heldSuffix || "") + ".png";
    const size = props.size || 54;
    return (
        <div className="mini-sprite" style={{ width: size, height: size }}>
            <AtlasSprite atlas={CHEFS_ATLAS} imageUrl={CHEFS_IMAGE_URL} frameName={bodyFrame} size={size} className="mini-sprite" />
            <AtlasSprite atlas={CHEFS_ATLAS} imageUrl={CHEFS_IMAGE_URL} frameName={hatFrame} size={size} className="mini-sprite mini-sprite--overlay" />
        </div>
    );
}
